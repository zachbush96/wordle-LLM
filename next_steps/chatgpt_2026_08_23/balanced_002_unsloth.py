from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
DEFAULT_CONFIG = MODULE_DIR / "balanced_002_unsloth_config.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Unsloth must patch Transformers before project model classes are imported.
# Unit tests and read-only commands intentionally do not require the isolated
# Unsloth environment.
if __name__ == "__main__" and "train" in sys.argv[1:]:
    import unsloth  # noqa: F401

from wordle_lab.common import (  # noqa: E402
    ARTIFACTS,
    DATA,
    canonical_json,
    read_json,
    read_jsonl,
    set_seed,
    sha256_file,
    write_json,
    write_jsonl,
)
from wordle_lab.experiments.intervention_sweep import (  # noqa: E402
    DECODING_VARIANTS,
    _explicit_feedback_messages,
)
from wordle_lab.methods.unsloth_sft import (  # noqa: E402
    UNSLOTH_WEIGHTED_BACKEND_ID,
    train_unsloth_sft,
    unsloth_environment,
    validate_unsloth_objective,
)
from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION  # noqa: E402
from wordle_lab.protocol.env import posterior_candidates, score_wordle  # noqa: E402


EXPERIMENT_ID = "UNSLOTH-BALANCED-002-WORD8-001"
CURRICULUM_ID = "COMMON-WORD-CURRICULUM-002"
EXPECTED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _resolve(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed if the historical matched recipe drifts."""
    exact = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": "WORDLE-PROTOCOL-002",
        "backend": UNSLOTH_WEIGHTED_BACKEND_ID,
        "seed": 2026,
        "max_steps": 600,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": 320,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "word_token_weight": 8.0,
        "checkpoint_steps": [150, 300, 450, 600],
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in exact.items()
        if config.get(key) != expected
    }
    model = config.get("model", {})
    if model != {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION}:
        mismatches["model"] = {
            "expected": {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION},
            "actual": model,
        }
    data = config.get("data", {})
    data_exact = {
        "curriculum_id": CURRICULUM_ID,
        "training_rows": 512,
        "state_copy_cap": 4,
        "target_word_cap": 8,
        "universe_size": 128,
        "train_secret_count": 96,
        "dev_secret_count": 32,
        "prompt_renderer": "explicit-constraints-v2-compact",
    }
    for key, expected in data_exact.items():
        if data.get(key) != expected:
            mismatches[f"data.{key}"] = {"expected": expected, "actual": data.get(key)}
    lora = config.get("lora", {})
    expected_lora = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": EXPECTED_TARGET_MODULES}
    if lora != expected_lora:
        mismatches["lora"] = {"expected": expected_lora, "actual": lora}
    evaluation = config.get("evaluation", {})
    expected_checkpoints = [f"step-{step:06d}" for step in exact["checkpoint_steps"]]
    if evaluation.get("checkpoints") != expected_checkpoints:
        mismatches["evaluation.checkpoints"] = {
            "expected": expected_checkpoints,
            "actual": evaluation.get("checkpoints"),
        }
    if evaluation.get("primary_decoder") != "greedy" or evaluation.get("sensitivity_decoder") != "greedy_rep105":
        mismatches["evaluation.decoders"] = {
            "expected": ["greedy", "greedy_rep105"],
            "actual": [evaluation.get("primary_decoder"), evaluation.get("sensitivity_decoder")],
        }
    if evaluation.get("prompt_variant") != "explicit_feedback":
        mismatches["evaluation.prompt_variant"] = {
            "expected": "explicit_feedback",
            "actual": evaluation.get("prompt_variant"),
        }
    validate_unsloth_objective(config)
    if mismatches:
        raise ValueError(f"matched balanced-002 recipe drifted: {json.dumps(mismatches, sort_keys=True)}")


def audit_protocol(config: dict[str, Any]) -> dict[str, Any]:
    expected_components = config["protocol_component_hashes"]
    actual_components = {
        name: sha256_file(ROOT / "wordle_lab" / "protocol" / name)
        for name in expected_components
    }
    if actual_components != expected_components:
        raise AssertionError("WORDLE-PROTOCOL-002 component hash drift")
    lock_path = _resolve(config["protocol_lock"])
    lock = read_json(lock_path)
    if lock.get("protocol_id") != config["protocol_id"]:
        raise AssertionError("protocol id mismatch")
    if lock.get("protocol_sha256") != config["protocol_sha256"]:
        raise AssertionError("protocol lock hash mismatch")
    if lock.get("component_files") != expected_components:
        raise AssertionError("protocol lock component manifest mismatch")
    return {
        "status": "passed",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "component_hashes": actual_components,
        "locked_test_file_read": False,
    }


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Audit only training and development inputs; never open the locked test."""
    data_config = config["data"]
    directory = _resolve(data_config["directory"])
    expected_hashes = data_config["hashes"]
    actual_hashes = {name: sha256_file(directory / name) for name in expected_hashes}
    if actual_hashes != expected_hashes:
        mismatched = {
            name: {"expected": expected_hashes[name], "actual": actual_hashes.get(name)}
            for name in expected_hashes
            if actual_hashes.get(name) != expected_hashes[name]
        }
        raise AssertionError(f"balanced-002 content hash mismatch: {mismatched}")

    manifest = read_json(directory / "manifest.json")
    rows = read_jsonl(directory / data_config["training_file"])
    state_manifest = read_jsonl(directory / "state_manifest.jsonl")
    canonical = read_jsonl(directory / "canonical.jsonl")
    recovery = read_jsonl(directory / "recovery_states.jsonl")
    universe = read_json(directory / "universe.json")
    train_secrets = set(read_json(directory / "train_secrets.json"))
    dev_secrets = set(read_json(directory / "dev_secrets.json"))

    if manifest.get("curriculum_id") != CURRICULUM_ID:
        raise AssertionError("unexpected curriculum id")
    if manifest.get("rendered_sha256") != expected_hashes["train.jsonl"]:
        raise AssertionError("manifest does not pin the training rows")
    if manifest.get("state_manifest_sha256") != expected_hashes["state_manifest.jsonl"]:
        raise AssertionError("manifest does not pin the state manifest")
    if len(rows) != data_config["training_rows"] or len(state_manifest) != len(rows):
        raise AssertionError("balanced-002 row count mismatch")
    if len(universe) != data_config["universe_size"] or len(set(universe)) != len(universe):
        raise AssertionError("balanced-002 universe shape mismatch")
    if len(train_secrets) != data_config["train_secret_count"] or len(dev_secrets) != data_config["dev_secret_count"]:
        raise AssertionError("balanced-002 split size mismatch")
    if train_secrets & dev_secrets:
        raise AssertionError("balanced-002 train/dev secret leakage")

    comparison_dir = _resolve(data_config["comparison_split_directory"])
    for name in ("universe.json", "train_secrets.json", "dev_secrets.json"):
        if sha256_file(comparison_dir / name) != actual_hashes[name]:
            raise AssertionError(f"balanced-002 {name} differs from the tracked matched split")

    source_variants: dict[str, list[dict]] = defaultdict(list)
    for source in canonical + recovery:
        source_variants[source["state_id"]].append(source)
        if source["secret_answer"] not in train_secrets or source["secret_answer"] in dev_secrets:
            raise AssertionError("source provenance is not training-only")

    state_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    composition: Counter[str] = Counter()
    projected_manifest = []
    for row in rows:
        source = row["source_state"]
        secret = source["secret_answer"]
        history = [(item["guess"], item["feedback"]) for item in source["history"]]
        target = row["target_word"]
        if source not in source_variants.get(source["state_id"], []):
            raise AssertionError(f"unproven source state: {row['example_id']}")
        if secret not in train_secrets or secret in dev_secrets:
            raise AssertionError(f"held-out label provenance: {row['example_id']}")
        if any(score_wordle(secret, guess) != feedback for guess, feedback in history):
            raise AssertionError(f"incorrect source feedback: {row['example_id']}")
        if secret not in posterior_candidates(history, universe):
            raise AssertionError(f"source history removes its secret: {row['example_id']}")
        if target != source["facts"]["oracle_action"] or target not in universe:
            raise AssertionError(f"target/source mismatch: {row['example_id']}")
        if row["completion"] != [{"role": "assistant", "content": f"Final answer: {target}"}]:
            raise AssertionError(f"completion envelope mismatch: {row['example_id']}")
        if row["prompt"] != _explicit_feedback_messages(history):
            raise AssertionError(f"prompt renderer mismatch: {row['example_id']}")
        state_counts[row["state_id"]] += 1
        target_counts[target] += 1
        composition[row["state_type"]] += 1
        projected_manifest.append({
            key: row[key]
            for key in (
                "example_id",
                "state_id",
                "state_type",
                "turn",
                "posterior_size",
                "target_word",
                "target_frequency",
            )
        })
    if projected_manifest != state_manifest:
        raise AssertionError("state manifest does not exactly project the training rows")
    if max(state_counts.values()) > data_config["state_copy_cap"]:
        raise AssertionError("state copy cap exceeded")
    if max(target_counts.values()) > data_config["target_word_cap"]:
        raise AssertionError("target word cap exceeded")
    if dict(sorted(composition.items())) != manifest.get("achieved_composition"):
        raise AssertionError("curriculum composition mismatch")
    if dict(sorted(target_counts.items())) != manifest.get("target_frequency_distribution"):
        raise AssertionError("target distribution mismatch")

    allowed_path = _resolve(config["evaluation"]["allowed_words"])
    if sha256_file(allowed_path) != config["evaluation"]["allowed_words_sha256"]:
        raise AssertionError("allowed-word list hash mismatch")

    return {
        "status": "passed",
        "curriculum_id": CURRICULUM_ID,
        "directory": str(directory),
        "training_rows": len(rows),
        "unique_source_states": len(state_counts),
        "state_copy_max": max(state_counts.values()),
        "target_word_max": max(target_counts.values()),
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "train_dev_overlap": len(train_secrets & dev_secrets),
        "split_identity_with_comparison_bundle": True,
        "prompt_renderer": data_config["prompt_renderer"],
        "hashes": actual_hashes,
        "checks": [
            "exact_content_hashes",
            "matched_u128_train96_dev32_split",
            "training_only_secret_provenance",
            "feedback_recomputed",
            "secret_retained_in_posterior",
            "oracle_target_and_completion_match",
            "explicit_prompt_exact_match",
            "state_manifest_exact_projection",
            "state_and_target_caps",
            "allowed_word_hash",
        ],
        "locked_test_file_read": False,
    }


def audit_all(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "protocol": audit_protocol(config),
        "dataset": audit_dataset(config),
        "locked_test_access": False,
    }


def build_spec(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "experiment_id": config["experiment_id"],
        "method": "unsloth-sft-balanced-002",
        "backend": config["backend"],
        "representation": "common_balanced_curriculum",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "seed": config["seed"],
        "max_steps": config["max_steps"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "max_length": config["max_length"],
        "precision": config["precision"],
        "quantization": config["quantization"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "word_token_weight": config["word_token_weight"],
        "checkpoint_steps": config["checkpoint_steps"],
        "lora": config["lora"],
        "model": config["model"],
        "curriculum": {
            "curriculum_id": config["data"]["curriculum_id"],
            "prompt_version": config["data"]["prompt_renderer"],
            "universe_size": config["data"]["universe_size"],
            "train_secret_count": config["data"]["train_secret_count"],
            "dev_secret_count": config["data"]["dev_secret_count"],
            "rendered_examples": config["data"]["training_rows"],
            "rendered_sha256": config["data"]["hashes"]["train.jsonl"],
            "state_manifest_sha256": config["data"]["hashes"]["state_manifest.jsonl"],
        },
        "data": {
            "directory": config["data"]["directory"],
            "training_file": config["data"]["training_file"],
            "training_rows": config["data"]["training_rows"],
            "hashes": config["data"]["hashes"],
            "audit_status": audit["dataset"]["status"],
            "dev_role": "evaluation_only_never_training",
        },
        "evaluation": config["evaluation"],
        "promotion_gates": config["promotion_gates"],
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    validate_unsloth_objective(spec)
    return spec


def run_id_for_spec(spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    return f"unsloth-balanced-002-word8-s{spec['seed']}-{digest}"


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    audit = audit_all(config)
    spec = build_spec(config, audit)
    run_id = run_id_for_spec(spec)
    return {
        "status": "dry_run_passed",
        "run_id": run_id,
        "planned_run_dir": str(_resolve(config["output_root"]) / run_id),
        "spec": spec,
        "audit": audit,
        "environment": unsloth_environment(),
        "training_started": False,
        "locked_test_access": False,
    }


def prepare_run_directory(
    config: dict[str, Any], run_dir: str | Path | None = None
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    audit = audit_all(config)
    spec = build_spec(config, audit)
    destination = Path(run_dir).resolve() if run_dir else _resolve(config["output_root"]) / run_id_for_spec(spec)
    spec_path = destination / "spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise RuntimeError(f"refusing to overwrite a different run spec at {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    write_json(destination / "dataset_manifest.json", {
        "experiment_id": config["experiment_id"],
        "curriculum_id": config["data"]["curriculum_id"],
        "source_directory": config["data"]["directory"],
        "hashes": config["data"]["hashes"],
        "audit": audit["dataset"],
        "dev_probe_role": "evaluation_only_never_training",
        "locked_test_access": False,
    })
    write_json(destination / "preflight_audit.json", audit)
    return destination, spec, audit


def train(config: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
    destination, spec, _ = prepare_run_directory(config, run_dir)
    if (destination / "summary.json").exists() or (destination / "checkpoints").exists():
        raise RuntimeError(f"refusing to overwrite an existing training run at {destination}")
    rows = read_jsonl(_resolve(config["data"]["directory"]) / config["data"]["training_file"])
    set_seed(config["seed"])
    _, _, accounting = train_unsloth_sft(rows, destination, spec)
    if accounting.get("checkpoint_steps") != config["checkpoint_steps"]:
        raise RuntimeError("trainer did not save the preregistered dose checkpoints")
    summary = {
        "status": "trained",
        "experiment_id": config["experiment_id"],
        "run_id": run_id_for_spec(spec),
        "run_dir": str(destination),
        "accounting": accounting,
        "locked_test_access": False,
    }
    write_json(destination / "summary.json", summary)
    return summary


def evaluation_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = config["evaluation"]["primary_decoder"]
    plan = [
        {"checkpoint": checkpoint, "decoder": primary, "include_retention": True}
        for checkpoint in config["evaluation"]["checkpoints"]
    ]
    plan.append({
        "checkpoint": "final",
        "decoder": config["evaluation"]["sensitivity_decoder"],
        "include_retention": False,
    })
    return plan


def promotion_gate_status(summary: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    terminal = summary["gameplay"].get("terminal_marker_compliance")
    turn_2 = summary["diagnostics"].get("by_turn", {}).get("2", {}).get(
        "posterior_constraint_violation_rate"
    )
    singleton = summary["diagnostics"].get("singleton_answer_accuracy")
    checks = {
        "terminal_compliance": terminal is not None and terminal >= gates["terminal_compliance_min"],
        "turn_2_constraint_violation": turn_2 is not None and turn_2 < gates["turn_2_constraint_violation_max"],
        "singleton_answer_accuracy": singleton is not None and singleton >= gates["singleton_answer_accuracy_min"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "terminal_compliance": terminal,
            "turn_2_constraint_violation": turn_2,
            "singleton_answer_accuracy": singleton,
        },
        "thresholds": gates,
    }


def _evaluate_one(
    config: dict[str, Any],
    run_dir: Path,
    checkpoint_name: str,
    decoder_name: str,
    include_retention: bool,
) -> dict[str, Any]:
    import torch

    from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
    from wordle_lab.data.canonical import generate_canonical_states
    from wordle_lab.models import load_adapter, load_tokenizer
    from wordle_lab.protocol import generation
    from wordle_lab.protocol.evaluator import evaluate
    from wordle_lab.protocol.retention import evaluate_retention

    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    tag = f"{checkpoint_name}-{decoder_name}"
    output_dir = run_dir / "evaluations" / tag
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)

    directory = _resolve(config["data"]["directory"])
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[: config["evaluation"]["dev_games"]]
    allowed = [
        line.strip().upper()
        for line in _resolve(config["evaluation"]["allowed_words"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_records = generate_canonical_states(
        dev_answers,
        "common_dev_diagnostic",
        config["evaluation"]["diagnostic_items"],
        seed=config["seed"],
        answer_vocabulary=universe,
    )
    training_records = read_jsonl(directory / "canonical.jsonl")
    tokenizer = load_tokenizer(checkpoint)
    model = load_adapter(checkpoint)
    original_messages = generation.inference_messages
    original_generation = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(config["seed"])
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(DECODING_VARIANTS[decoder_name])
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        write_jsonl(output_dir / "games.jsonl", games)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            training_records,
            allowed,
            universe,
            output_dir,
        )
        retention = None
        if include_retention:
            probes = read_jsonl(_resolve(config["evaluation"]["retention_probes"]))
            if len(probes) != config["evaluation"]["retention_probe_count"]:
                raise AssertionError("retention probe count mismatch")
            retention_rows, retention = evaluate_retention(model, tokenizer, probes)
            write_jsonl(output_dir / "retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": config["experiment_id"],
            "split": "balanced_002_dev_32",
            "checkpoint": checkpoint_name,
            "decoder": decoder_name,
            "prompt_variant": config["evaluation"]["prompt_variant"],
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
            "locked_test_access": False,
        }
        summary["promotion_gates"] = promotion_gate_status(summary, config["promotion_gates"])
        write_json(summary_path, summary)
        return summary
    finally:
        generation.inference_messages = original_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(original_generation)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_all(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    audit = audit_all(config)
    destination = Path(run_dir).resolve()
    spec = read_json(destination / "spec.json")
    expected_spec = build_spec(config, audit)
    if spec != expected_spec:
        raise RuntimeError("run spec does not match the frozen balanced-002 reproduction")
    results = [
        _evaluate_one(config, destination, item["checkpoint"], item["decoder"], item["include_retention"])
        for item in evaluation_plan(config)
    ]
    primary = results[: len(config["evaluation"]["checkpoints"])]
    promotable = [row["checkpoint"] for row in primary if row["promotion_gates"]["passed"]]
    aggregate = {
        "status": "evaluation_complete",
        "experiment_id": config["experiment_id"],
        "run_dir": str(destination),
        "conditions": results,
        "promotable_primary_checkpoints": promotable,
        "replication_allowed": bool(promotable),
        "locked_test_access": False,
        "decision": (
            "development_gate_passed_replication_allowed_locked_test_still_closed"
            if promotable
            else "development_gate_failed_stop_locked_test_closed"
        ),
    }
    write_json(destination / "evaluation_summary.json", aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matched Unsloth reproduction of balanced-002 with 8x action loss")
    parser.add_argument("command", choices=("audit", "dry-run", "prepare", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "audit":
        result = audit_all(config)
    elif args.command == "dry-run":
        result = dry_run(config)
    elif args.command == "prepare":
        destination, spec, audit = prepare_run_directory(config, args.run_dir)
        result = {"status": "prepared", "run_dir": str(destination), "spec": spec, "audit": audit}
    elif args.command == "train":
        result = train(config, args.run_dir)
    else:
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir")
        result = evaluate_all(config, args.run_dir)
    if args.compact:
        result = {
            key: result.get(key)
            for key in ("status", "experiment_id", "run_id", "run_dir", "decision", "locked_test_access")
            if key in result
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
