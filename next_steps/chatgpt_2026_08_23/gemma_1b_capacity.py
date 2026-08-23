from __future__ import annotations

"""Provisioning, training, and evaluation for the clean same-family 1B cell."""

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import torch
from huggingface_hub import HfApi, get_token, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import ARTIFACTS, DATA, ROOT, canonical_json, read_json, read_jsonl, set_seed, sha256_file, write_json, write_jsonl
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.sft import train_sft
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention


CAPACITY_ID = "GEMMA-SAME-FAMILY-CAPACITY-001"
MODEL_ID = "google/gemma-3-1b-it"
MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
SUITE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = SUITE_ROOT / "gemma_1b_capacity_config.json"
DEFAULT_MODEL_DIR = ROOT / "models" / "base" / "google--gemma-3-1b-it"
DEFAULT_DATA = ROOT / "data" / "common-curriculum-002" / "u128-train96"
DEFAULT_PREFLIGHT = SUITE_ROOT / "generated" / "capacity" / "gemma_1b_preflight.json"
EXPECTED_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
EXPECTED_COMPARATOR = {
    "run_id": "sft-common-balanced-word-s2026-0649b4deeb",
    "backend": "transformers_peft",
    "spec_sha256": "655972a100f33ca26d0e7834f602de9856f88aed89107aed67e6426f9d8c95bc",
    "dataset_manifest_sha256": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
    "accounting_sha256": "4a46e116d8d998ec20c6f05ebd74b2cd8c4525feb52f9dad8cc22327940625b4",
    "final_adapter_sha256": "6404d0d4c06f2c2cc2dcd871fc7a6dd7b05201057d11d6be613e3188c76a2817",
}
CAPACITY_CHECKPOINTS = ("step-000150", "step-000300", "step-000450", "step-000600")
CAPACITY_SPLIT = "balanced_002_dev_32"
# This is the complete 270M development contract, applied prospectively to the
# blocked 1B cell. Passing chooses a replication candidate only; it never opens
# the locked test.
CAPACITY_GATE_THRESHOLDS = {
    "terminal_marker_compliance": {"group": "format", "op": ">=", "value": 0.99},
    "invalid_guess_rate": {"group": "legality", "op": "<=", "value": 0.01},
    "repeat_guess_rate": {"group": "legality", "op": "<=", "value": 0.10},
    "gameplay_constraint_violation_rate": {"group": "legality", "op": "<=", "value": 0.30},
    "posterior_constraint_violation_rate": {"group": "posterior", "op": "<=", "value": 0.30},
    "turn_2_posterior_constraint_violation_rate": {"group": "posterior", "op": "<", "value": 0.30},
    "singleton_answer_accuracy": {"group": "singleton", "op": ">=", "value": 0.80},
    "retention_overall_score": {"group": "retention", "op": ">=", "value": 0.20},
    "win_rate": {"group": "gameplay", "op": ">=", "value": 0.25},
}


def capacity_evaluation_policy() -> dict[str, Any]:
    """Return the frozen, pre-training all-dose development policy."""
    return {
        "schema_version": "gemma-1b-capacity-evaluation-policy-v1",
        "registration_status": "declared_before_training",
        "preregistered": True,
        "split": CAPACITY_SPLIT,
        "dev_games": 32,
        "diagnostic_items": 128,
        "checkpoints": list(CAPACITY_CHECKPOINTS),
        "prompt_variant": "explicit_feedback",
        "decoder": "greedy",
        "generation": {"do_sample": False, "max_new_tokens": 128, "use_cache": True},
        "gate_thresholds": deepcopy(CAPACITY_GATE_THRESHOLDS),
        "selection": (
            "among gate-passing doses, minimize overall then turn-2 posterior violation; "
            "maximize singleton accuracy, format compliance, retention, and win rate; "
            "then minimize gameplay violations/repeats and prefer the earlier dose"
        ),
        "singleton_correctness_mandatory": True,
        "scope": "single-seed development selection for replication only",
        "locked_test_access": False,
        "locked_test_authorized": False,
    }


def validate_capacity_config(config: dict[str, Any]) -> None:
    expected = {
        "experiment_id": CAPACITY_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_model_revision_checked_at": "2026-08-23",
        "training_backend": "transformers_peft",
        "matched_270m_comparator": EXPECTED_COMPARATOR,
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "training_rows": 512,
        "training_rows_sha256": "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f",
        "seed": 2026,
        "max_steps": 600,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "max_length": 320,
        "word_token_weight": 8.0,
        "checkpoint_steps": [150, 300, 450, 600],
        "lora": EXPECTED_LORA,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    drift = {key: (value, config.get(key)) for key, value in expected.items() if config.get(key) != value}
    unexpected = sorted(set(config) - set(expected))
    if unexpected:
        drift["unexpected_keys"] = ([], unexpected)
    if drift:
        raise ValueError(f"Gemma 1B capacity config drift: {drift}")


def audit_270m_comparator(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the exact native-PEFT 270M arm used for the scale comparison."""
    declared = config["matched_270m_comparator"]
    run_dir = ARTIFACTS / "runs" / declared["run_id"]
    required = {
        "spec.json": declared["spec_sha256"],
        "dataset_manifest.json": declared["dataset_manifest_sha256"],
        "accounting.json": declared["accounting_sha256"],
        "checkpoints/final/adapter_model.safetensors": declared["final_adapter_sha256"],
    }
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"matched 270M comparator is incomplete: {missing}")
    actual = {name: sha256_file(run_dir / name) for name in required}
    if actual != required:
        raise AssertionError("matched 270M comparator artifact hash drift")
    spec = read_json(run_dir / "spec.json")
    accounting = read_json(run_dir / "accounting.json")
    recipe_checks = {
        "method": spec.get("method") == "sft",
        "model": spec.get("model", {}).get("model_id") == "google/gemma-3-270m-it",
        "model_revision": spec.get("model", {}).get("revision") == "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
        "curriculum": spec.get("curriculum", {}).get("curriculum_id") == "COMMON-WORD-CURRICULUM-002",
        "rows": spec.get("curriculum", {}).get("rendered_sha256") == config["training_rows_sha256"],
        "seed": spec.get("seed") == config["seed"],
        "steps": spec.get("max_steps") == config["max_steps"],
        "learning_rate": spec.get("learning_rate") == config["learning_rate"],
        "batch_size": spec.get("batch_size") == config["batch_size"],
        "accumulation": spec.get("gradient_accumulation_steps") == config["gradient_accumulation_steps"],
        "max_length": spec.get("max_length") == config["max_length"],
        "word_token_weight": spec.get("word_token_weight") == config["word_token_weight"],
        "lora": spec.get("lora") == config["lora"],
        "checkpoint_steps": accounting.get("checkpoint_steps") == config["checkpoint_steps"],
        "effective_batch": accounting.get("effective_batch_size") == config["effective_batch_size"],
    }
    if not all(recipe_checks.values()):
        raise AssertionError(f"matched 270M comparator recipe drift: {[key for key, ok in recipe_checks.items() if not ok]}")
    return {
        "status": "passed",
        "run_id": declared["run_id"],
        "backend": declared["backend"],
        "artifact_hashes": actual,
        "recipe_checks": recipe_checks,
        "locked_test_access": False,
    }


def _local_model_status(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    metadata_path = model_dir / "wordle_lab_model.json"
    if not config_path.exists() or not metadata_path.exists():
        return {"available": False, "reason": "local_snapshot_missing"}
    model_config = read_json(config_path)
    metadata = read_json(metadata_path)
    checks = {
        "model_id": metadata.get("model_id") == MODEL_ID,
        "revision": metadata.get("revision") == MODEL_REVISION,
        "model_type": model_config.get("model_type") == "gemma3_text",
        "architecture": model_config.get("architectures") == ["Gemma3ForCausalLM"],
    }
    return {
        "available": all(checks.values()),
        "reason": None if all(checks.values()) else "local_snapshot_metadata_mismatch",
        "checks": checks,
        "path": str(model_dir),
    }


def capacity_preflight(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    output_path: Path | None = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    """Check the official model record and local/auth availability without exposing a token."""
    token = get_token()
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, token=token)
    local = _local_model_status(Path(model_dir))
    result = {
        "experiment_id": CAPACITY_ID,
        "model_id": MODEL_ID,
        "pinned_revision": MODEL_REVISION,
        "remote_revision": info.sha,
        "remote_revision_matches": info.sha == MODEL_REVISION,
        "gated": info.gated,
        "private": bool(info.private),
        "huggingface_authenticated": bool(token),
        "local": local,
        "ready": bool(local["available"]),
        "status": (
            "ready"
            if local["available"]
            else "blocked_missing_huggingface_auth_for_gated_model"
            if not token
            else "provisioning_required"
        ),
        "locked_test_access": False,
        "credentials_recorded": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def provision_model(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Download only the pinned licensed snapshot when an existing token is available."""
    preflight = capacity_preflight(model_dir=model_dir, output_path=None)
    if preflight["ready"]:
        return preflight
    token = get_token()
    if not token:
        preflight["download_attempted"] = False
        preflight["reason"] = "no Hugging Face token is configured; gated files cannot be downloaded"
        write_json(DEFAULT_PREFLIGHT, preflight)
        return preflight
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=Path(model_dir),
        token=token,
    )
    write_json(
        Path(model_dir) / "wordle_lab_model.json",
        {"model_id": MODEL_ID, "revision": MODEL_REVISION, "source": "huggingface_hub snapshot_download"},
    )
    result = capacity_preflight(model_dir=model_dir)
    result["download_attempted"] = True
    return result


def audit_capacity_data(config: dict[str, Any], data_dir: Path = DEFAULT_DATA) -> dict[str, Any]:
    validate_capacity_config(config)
    rows = read_jsonl(data_dir / "train.jsonl")
    manifest = read_json(data_dir / "manifest.json")
    train = set(read_json(data_dir / "train_secrets.json"))
    dev = set(read_json(data_dir / "dev_secrets.json"))
    if len(rows) != config["training_rows"] or sha256_file(data_dir / "train.jsonl") != config["training_rows_sha256"]:
        raise AssertionError("Gemma 1B capacity data does not match balanced-002")
    if manifest["rendered_sha256"] != config["training_rows_sha256"] or train & dev:
        raise AssertionError("Gemma 1B capacity split/hash audit failed")
    if any(row["source_state"]["secret_answer"] not in train for row in rows):
        raise AssertionError("Gemma 1B capacity labels are not training-only")
    return {
        "status": "passed",
        "rows": len(rows),
        "rows_sha256": config["training_rows_sha256"],
        "manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "state_manifest_sha256": sha256_file(data_dir / "state_manifest.jsonl"),
        "canonical_sha256": sha256_file(data_dir / "canonical.jsonl"),
        "universe_sha256": sha256_file(data_dir / "universe.json"),
        "train_secrets_sha256": sha256_file(data_dir / "train_secrets.json"),
        "dev_secrets_sha256": sha256_file(data_dir / "dev_secrets.json"),
        "allowed_words_sha256": sha256_file(ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt"),
        "retention_probes_sha256": sha256_file(DATA / "retention_probes_v1.jsonl"),
        "train_secrets": len(train),
        "dev_secrets": len(dev),
        "locked_test_access": False,
    }


def capacity_spec(config: dict[str, Any], data_dir: Path = DEFAULT_DATA) -> dict[str, Any]:
    audit = audit_capacity_data(config, data_dir)
    protocol = read_json(DATA / "protocol_lock.json")
    comparator = audit_270m_comparator(config)
    return {
        "experiment_id": CAPACITY_ID,
        "method": "sft",
        "training_backend": config["training_backend"],
        "representation": "common_balanced_curriculum",
        "seed": config["seed"],
        "max_steps": config["max_steps"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_batch_size": config["effective_batch_size"],
        "max_length": config["max_length"],
        "word_token_weight": config["word_token_weight"],
        "checkpoint_steps": config["checkpoint_steps"],
        "lora": config["lora"],
        "matched_270m_comparator": comparator,
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "local_path": str(DEFAULT_MODEL_DIR)},
        "data": audit,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_lock_file_sha256": sha256_file(DATA / "protocol_lock.json"),
        "evaluation": capacity_evaluation_policy(),
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }


def _tokenizer_loader():
    status = _local_model_status(DEFAULT_MODEL_DIR)
    if not status["available"]:
        raise RuntimeError("pinned Gemma 1B snapshot is not available")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _model_loader(training: bool = False):
    status = _local_model_status(DEFAULT_MODEL_DIR)
    if not status["available"]:
        raise RuntimeError("pinned Gemma 1B snapshot is not available")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_DIR,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = not training
    return model


def train_capacity(config: dict[str, Any], data_dir: Path = DEFAULT_DATA) -> dict[str, Any]:
    preflight = capacity_preflight()
    if not preflight["ready"]:
        raise RuntimeError(preflight["status"])
    spec = capacity_spec(config, data_dir)
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"gemma-1b-balanced-word-s{config['seed']}-{digest}"
    if (run_dir / "summary.json").exists() or (run_dir / "checkpoints").exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", spec["data"])
    set_seed(config["seed"])
    model = None
    try:
        model, accounting = train_sft(
            read_jsonl(data_dir / "train.jsonl"),
            run_dir,
            spec,
            tokenizer_loader=_tokenizer_loader,
            model_loader=_model_loader,
        )
        summary = {"status": "trained", "run_dir": str(run_dir), "accounting": accounting, "locked_test_access": False}
        write_json(run_dir / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def _load_capacity_adapter(checkpoint: Path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _model_loader(training=False)
    return PeftModel.from_pretrained(model, checkpoint).to("cuda"), tokenizer


def _capacity_spec_sha256(spec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def _capacity_evaluation_data_binding(spec: dict[str, Any]) -> dict[str, Any]:
    data = spec["data"]
    return {
        "rows_sha256": data["rows_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "state_manifest_sha256": data["state_manifest_sha256"],
        "canonical_sha256": data["canonical_sha256"],
        "universe_sha256": data["universe_sha256"],
        "train_secrets_sha256": data["train_secrets_sha256"],
        "dev_secrets_sha256": data["dev_secrets_sha256"],
        "allowed_words_sha256": data["allowed_words_sha256"],
        "retention_probes_sha256": data["retention_probes_sha256"],
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": spec["protocol_sha256"],
        "protocol_lock_file_sha256": spec["protocol_lock_file_sha256"],
    }


def _validated_capacity_run(run_dir: Path, data_dir: Path) -> dict[str, Any]:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    spec_path = run_dir / "spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = read_json(spec_path)
    if spec.get("experiment_id") != CAPACITY_ID or spec.get("locked_test_access") is not False:
        raise RuntimeError("not a locked-test-free Gemma 1B capacity run")
    expected_spec = capacity_spec(read_json(DEFAULT_CONFIG), data_dir)
    if spec != expected_spec:
        raise RuntimeError("Gemma 1B run spec differs from the fully reconstructed matched contract")
    if spec.get("evaluation") != capacity_evaluation_policy():
        raise RuntimeError("Gemma 1B evaluation policy differs from the frozen all-dose contract")
    return spec


def _unit_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing or invalid Gemma 1B metric: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid Gemma 1B metric: {label}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Gemma 1B metric {label} must be finite and in [0, 1]")
    return number


def _capacity_dose_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("experiment_id") != CAPACITY_ID:
        raise ValueError("Gemma 1B dose summary has the wrong experiment id")
    if summary.get("split") != CAPACITY_SPLIT:
        raise ValueError("Gemma 1B dose selection requires the frozen development split")
    if summary.get("locked_test_access") is not False or summary.get("locked_test_authorized") is not False:
        raise ValueError("Gemma 1B dose summary must explicitly keep the locked test closed")
    gameplay = summary.get("gameplay")
    diagnostics = summary.get("diagnostics")
    retention = summary.get("retention")
    if not isinstance(gameplay, dict) or not isinstance(diagnostics, dict) or not isinstance(retention, dict):
        raise ValueError("Gemma 1B dose summary is missing gameplay, diagnostics, or retention")
    by_turn = diagnostics.get("by_turn")
    if not isinstance(by_turn, dict):
        raise ValueError("Gemma 1B dose summary is missing diagnostic turn metrics")
    turn_two = by_turn.get("2", by_turn.get(2))
    if not isinstance(turn_two, dict):
        raise ValueError("Gemma 1B dose summary is missing turn-2 diagnostics")
    wins = gameplay.get("wins")
    games = gameplay.get("n_games")
    if isinstance(wins, bool) or not isinstance(wins, int) or wins < 0:
        raise ValueError("Gemma 1B gameplay wins must be a nonnegative integer")
    if (
        isinstance(games, bool)
        or not isinstance(games, int)
        or games != capacity_evaluation_policy()["dev_games"]
        or wins > games
    ):
        raise ValueError("Gemma 1B gameplay n_games must equal the frozen 32-game development dose")
    metrics = {
        "terminal_marker_compliance": _unit_metric(
            gameplay.get("terminal_marker_compliance"), "gameplay.terminal_marker_compliance"
        ),
        "invalid_guess_rate": _unit_metric(gameplay.get("invalid_guess_rate"), "gameplay.invalid_guess_rate"),
        "repeat_guess_rate": _unit_metric(gameplay.get("repeat_guess_rate"), "gameplay.repeat_guess_rate"),
        "gameplay_constraint_violation_rate": _unit_metric(
            gameplay.get("constraint_violation_rate"), "gameplay.constraint_violation_rate"
        ),
        "posterior_constraint_violation_rate": _unit_metric(
            diagnostics.get("posterior_constraint_violation_rate"),
            "diagnostics.posterior_constraint_violation_rate",
        ),
        "turn_2_posterior_constraint_violation_rate": _unit_metric(
            turn_two.get("posterior_constraint_violation_rate"),
            "diagnostics.by_turn.2.posterior_constraint_violation_rate",
        ),
        "singleton_answer_accuracy": _unit_metric(
            diagnostics.get("singleton_answer_accuracy"), "diagnostics.singleton_answer_accuracy"
        ),
        "retention_overall_score": _unit_metric(retention.get("overall_score"), "retention.overall_score"),
        "win_rate": _unit_metric(gameplay.get("win_rate"), "gameplay.win_rate"),
        "wins": wins,
        "n_games": games,
    }
    for optional in ("action_target_accuracy", "posterior_consistency"):
        if diagnostics.get(optional) is not None:
            metrics[optional] = _unit_metric(diagnostics[optional], f"diagnostics.{optional}")
    return metrics


def capacity_gate_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the complete matched 270M development contract to one 1B dose."""
    metrics = _capacity_dose_metrics(summary)
    checks: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[bool]] = defaultdict(list)
    failures: list[str] = []
    for metric, rule in CAPACITY_GATE_THRESHOLDS.items():
        observed = metrics[metric]
        if rule["op"] == ">=":
            passed = observed >= rule["value"]
        elif rule["op"] == "<=":
            passed = observed <= rule["value"]
        elif rule["op"] == "<":
            passed = observed < rule["value"]
        else:  # pragma: no cover - constants above freeze the supported operators
            raise ValueError(f"unsupported Gemma 1B gate operator: {rule['op']}")
        checks[metric] = {"observed": observed, **rule, "passed": passed}
        grouped[rule["group"]].append(passed)
        if not passed:
            failures.append(f"threshold_failed:{metric}")
    return {
        "schema_version": "gemma-1b-capacity-development-gates-v1",
        "registration_status": "declared_before_training",
        "preregistered": True,
        "passed": not failures,
        "checks": checks,
        "groups": {group: all(group_checks) for group, group_checks in sorted(grouped.items())},
        "failures": failures,
        "metrics": metrics,
        "singleton_correctness_mandatory": True,
        "locked_test_access": False,
        "locked_test_authorized": False,
    }


def validate_capacity_evaluation_summary(
    summary: dict[str, Any],
    spec: dict[str, Any],
    checkpoint: str,
) -> dict[str, Any]:
    """Fail closed before an existing development dose is reused."""
    if checkpoint not in CAPACITY_CHECKPOINTS:
        raise ValueError(f"Gemma 1B checkpoint must be one of {list(CAPACITY_CHECKPOINTS)}")
    exact = {
        "status": "dev_evaluated",
        "experiment_id": CAPACITY_ID,
        "model": spec["model"],
        "matched_270m_comparator": spec["matched_270m_comparator"],
        "checkpoint": checkpoint,
        "split": CAPACITY_SPLIT,
        "decoder": "greedy",
        "prompt_variant": "explicit_feedback",
        "run_spec_sha256": _capacity_spec_sha256(spec),
        "evaluation_data": _capacity_evaluation_data_binding(spec),
        "evaluation_policy": capacity_evaluation_policy(),
        "locked_test_access": False,
        "locked_test_authorized": False,
    }
    drift = {key: {"expected": value, "actual": summary.get(key)} for key, value in exact.items() if summary.get(key) != value}
    if drift:
        raise ValueError(f"Gemma 1B development summary binding drift: {drift}")
    gates = capacity_gate_status(summary)
    if summary.get("development_gates") != gates:
        raise ValueError(f"stored Gemma 1B development gates drift for {checkpoint}")
    return gates


def _validate_reuse_artifacts(summary: dict[str, Any], run_dir: Path, checkpoint: str) -> None:
    expected = {
        "games": f"eval-{checkpoint}-games.jsonl",
        "retention": f"eval-{checkpoint}-retention.jsonl",
    }
    artifacts = summary.get("evaluation_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected):
        raise ValueError(f"Gemma 1B reused evidence declaration drift for {checkpoint}")
    for label, relative in expected.items():
        record = artifacts[label]
        if not isinstance(record, dict) or record.get("path") != relative:
            raise ValueError(f"Gemma 1B reused {label} path drift for {checkpoint}")
        path = run_dir / relative
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Gemma 1B reused {label} hash drift for {checkpoint}")


def aggregate_capacity_doses(
    summaries: Sequence[dict[str, Any]],
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate all four doses and select one deterministically, or select none."""
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        checkpoint = summary.get("checkpoint")
        if checkpoint not in CAPACITY_CHECKPOINTS:
            raise ValueError(f"unexpected Gemma 1B checkpoint summary: {checkpoint}")
        if checkpoint in by_checkpoint:
            raise ValueError(f"duplicate Gemma 1B checkpoint summary: {checkpoint}")
        by_checkpoint[str(checkpoint)] = summary
    missing = [checkpoint for checkpoint in CAPACITY_CHECKPOINTS if checkpoint not in by_checkpoint]
    if missing:
        raise ValueError(f"missing Gemma 1B checkpoint summaries: {missing}")

    doses: list[dict[str, Any]] = []
    for checkpoint in CAPACITY_CHECKPOINTS:
        summary = by_checkpoint[checkpoint]
        gates = validate_capacity_evaluation_summary(summary, spec, checkpoint)
        doses.append(
            {
                "checkpoint": checkpoint,
                "development_gates": gates,
                "evaluation_summary_sha256": hashlib.sha256(
                    canonical_json(summary).encode("utf-8")
                ).hexdigest(),
            }
        )
    passing = [dose for dose in doses if dose["development_gates"]["passed"]]

    def selection_key(dose: dict[str, Any]) -> tuple[Any, ...]:
        metrics = dose["development_gates"]["metrics"]
        return (
            metrics["posterior_constraint_violation_rate"],
            metrics["turn_2_posterior_constraint_violation_rate"],
            -metrics["singleton_answer_accuracy"],
            -metrics["terminal_marker_compliance"],
            -metrics["retention_overall_score"],
            -metrics["win_rate"],
            metrics["gameplay_constraint_violation_rate"],
            metrics["repeat_guess_rate"],
            CAPACITY_CHECKPOINTS.index(dose["checkpoint"]),
        )

    ranked = sorted(passing, key=selection_key)
    selected = ranked[0]["checkpoint"] if ranked else None
    promotable = [dose["checkpoint"] for dose in ranked]
    return {
        "schema_version": "gemma-1b-capacity-dose-evaluation-v1",
        "status": "evaluation_complete",
        "experiment_id": CAPACITY_ID,
        "model": spec["model"],
        "matched_270m_comparator": spec["matched_270m_comparator"],
        "run_spec_sha256": _capacity_spec_sha256(spec),
        "evaluation_data": _capacity_evaluation_data_binding(spec),
        "evaluation_policy": capacity_evaluation_policy(),
        "split": CAPACITY_SPLIT,
        "single_seed": True,
        "checkpoints": list(CAPACITY_CHECKPOINTS),
        "thresholds": deepcopy(CAPACITY_GATE_THRESHOLDS),
        "selection_contract": {
            "registration_status": "declared_before_training",
            "preregistered": True,
            "singleton_correctness_mandatory": True,
            "rank_order": [
                "posterior_constraint_violation_rate ascending",
                "turn_2_posterior_constraint_violation_rate ascending",
                "singleton_answer_accuracy descending",
                "terminal_marker_compliance descending",
                "retention_overall_score descending",
                "win_rate descending",
                "gameplay_constraint_violation_rate ascending",
                "repeat_guess_rate ascending",
                "earlier checkpoint dose",
            ],
            "scope": "single-seed development selection for replication only",
        },
        "doses": doses,
        "promotable_checkpoints": promotable,
        "selected_checkpoint": selected,
        "replication_allowed": selected is not None,
        "decision": (
            "development_gates_passed_replication_candidate_locked_test_closed"
            if selected is not None
            else "development_gates_failed_no_promotable_checkpoint_locked_test_closed"
        ),
        "locked_test_access": False,
        "locked_test_authorized": False,
    }


def evaluate_capacity_checkpoint(
    run_dir: Path,
    checkpoint: str,
    data_dir: Path = DEFAULT_DATA,
    *,
    dev_games: int = 32,
) -> dict[str, Any]:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    spec = _validated_capacity_run(run_dir, data_dir)
    if checkpoint not in CAPACITY_CHECKPOINTS:
        raise ValueError(f"Gemma 1B checkpoint must be one of {list(CAPACITY_CHECKPOINTS)}")
    if dev_games != spec["evaluation"]["dev_games"]:
        raise ValueError("Gemma 1B capacity evaluation requires exactly 32 development games")
    checkpoint_dir = run_dir / "checkpoints" / checkpoint
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    summary_path = run_dir / f"eval-{checkpoint}-summary.json"
    games_path = run_dir / f"eval-{checkpoint}-games.jsonl"
    retention_path = run_dir / f"eval-{checkpoint}-retention.jsonl"
    diagnostics_parent = run_dir / f"eval-{checkpoint}"
    conflicts = [path for path in (summary_path, games_path, retention_path, diagnostics_parent) if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite existing Gemma 1B evaluation artifacts: {conflicts}")
    universe = read_json(data_dir / "universe.json")
    dev_answers = read_json(data_dir / "dev_secrets.json")[:dev_games]
    if len(dev_answers) != dev_games:
        raise ValueError("Gemma 1B capacity development split does not contain exactly 32 answers")
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = None
    previous_messages = generation.inference_messages
    previous_config = dict(generation.GENERATION_CONFIG)
    try:
        model, tokenizer = _load_capacity_adapter(checkpoint_dir)
        set_seed(int(spec["seed"]))
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(spec["evaluation"]["generation"])
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        probes = generate_canonical_states(
            dev_answers,
            "gemma_1b_dev",
            spec["evaluation"]["diagnostic_items"],
            seed=spec["seed"],
            answer_vocabulary=universe,
        )
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            probes,
            read_jsonl(data_dir / "canonical.jsonl"),
            allowed,
            universe,
            diagnostics_parent,
        )
        retention_rows, retention = evaluate_retention(
            model,
            tokenizer,
            read_jsonl(DATA / "retention_probes_v1.jsonl"),
        )
        write_jsonl(games_path, games)
        write_jsonl(retention_path, retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": CAPACITY_ID,
            "model": spec["model"],
            "matched_270m_comparator": spec["matched_270m_comparator"],
            "run_spec_sha256": _capacity_spec_sha256(spec),
            "checkpoint": checkpoint,
            "split": CAPACITY_SPLIT,
            "decoder": spec["evaluation"]["decoder"],
            "prompt_variant": spec["evaluation"]["prompt_variant"],
            "evaluation_data": _capacity_evaluation_data_binding(spec),
            "evaluation_policy": capacity_evaluation_policy(),
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
            "evaluation_artifacts": {
                "games": {"path": games_path.name, "sha256": sha256_file(games_path)},
                "retention": {"path": retention_path.name, "sha256": sha256_file(retention_path)},
            },
            "locked_test_access": False,
            "locked_test_authorized": False,
        }
        summary["development_gates"] = capacity_gate_status(summary)
        validate_capacity_evaluation_summary(summary, spec, checkpoint)
        _validate_reuse_artifacts(summary, run_dir, checkpoint)
        write_json(summary_path, summary)
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_config)
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_capacity(
    run_dir: Path,
    checkpoint: str = "step-000600",
    data_dir: Path = DEFAULT_DATA,
) -> dict[str, Any]:
    """Backward-compatible single-dose entry point with an exact dose default."""
    return evaluate_capacity_checkpoint(run_dir, checkpoint, data_dir)


def evaluate_capacity_doses(
    run_dir: Path,
    *,
    data_dir: Path = DEFAULT_DATA,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Evaluate all exact doses and write one deterministic development decision."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    spec = _validated_capacity_run(run_dir, data_dir)
    for checkpoint in CAPACITY_CHECKPOINTS:
        checkpoint_dir = run_dir / "checkpoints" / checkpoint
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(checkpoint_dir)
    aggregate_path = run_dir / "evaluation_summary.json"
    summary_paths = {checkpoint: run_dir / f"eval-{checkpoint}-summary.json" for checkpoint in CAPACITY_CHECKPOINTS}
    if not reuse_existing:
        conflicts = [path for path in [aggregate_path, *summary_paths.values()] if path.exists()]
        if conflicts:
            raise FileExistsError(f"refusing to overwrite existing Gemma 1B evaluation artifacts: {conflicts}")
    elif aggregate_path.is_file() and any(not path.is_file() for path in summary_paths.values()):
        raise ValueError("existing Gemma 1B aggregate cannot be reused with missing dose summaries")

    summaries: dict[str, dict[str, Any]] = {}
    for checkpoint, summary_path in summary_paths.items():
        if summary_path.is_file():
            summary = read_json(summary_path)
            validate_capacity_evaluation_summary(summary, spec, checkpoint)
            _validate_reuse_artifacts(summary, run_dir, checkpoint)
            summaries[checkpoint] = summary

    for checkpoint in CAPACITY_CHECKPOINTS:
        if checkpoint not in summaries:
            summary = evaluate_capacity_checkpoint(run_dir, checkpoint, data_dir)
            validate_capacity_evaluation_summary(summary, spec, checkpoint)
            _validate_reuse_artifacts(summary, run_dir, checkpoint)
            summaries[checkpoint] = summary

    aggregate = aggregate_capacity_doses(
        [summaries[checkpoint] for checkpoint in CAPACITY_CHECKPOINTS],
        spec=spec,
    )
    aggregate["run_dir"] = str(run_dir.resolve())
    if aggregate_path.is_file():
        existing = read_json(aggregate_path)
        if existing != aggregate:
            raise ValueError("existing Gemma 1B aggregate evaluation summary drift")
        return existing
    write_json(aggregate_path, aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma 3 1B same-family Wordle capacity experiment")
    parser.add_argument(
        "command",
        choices=("preflight", "provision", "dry-run", "train", "evaluate", "evaluate-all"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    validate_capacity_config(config)
    if args.command == "preflight":
        result = capacity_preflight()
    elif args.command == "provision":
        result = provision_model()
    elif args.command == "dry-run":
        result = {"status": "dry_run_passed", "preflight": capacity_preflight(), "spec": capacity_spec(config)}
    elif args.command == "train":
        result = train_capacity(config)
    elif args.command == "evaluate":
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir")
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint at one exact dose")
        result = evaluate_capacity_checkpoint(args.run_dir, args.checkpoint)
    else:
        if args.run_dir is None:
            parser.error("evaluate-all requires --run-dir")
        if args.checkpoint is not None:
            parser.error("evaluate-all evaluates the frozen four-dose set and does not accept --checkpoint")
        result = evaluate_capacity_doses(
            args.run_dir,
            reuse_existing=not args.no_reuse_existing,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
