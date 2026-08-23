from __future__ import annotations

"""CLI for the matched Gemma 270M LoRA-versus-full-capacity experiment."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import ARTIFACTS, DATA, ROOT, canonical_json, read_json, read_jsonl, set_seed, sha256_file, sha256_text, write_json, write_jsonl
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.models import model_metadata
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention

from .full_finetune import (
    EXPECTED_CHECKPOINT_FRACTIONS,
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_COMPARATOR_EVIDENCE,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_LORA_WRAPPED_PARAMETER_COUNT,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_PROTOCOL_SHA256,
    FULL_FINETUNE_BACKEND_ID,
    FULL_FINETUNE_EXPERIMENT_ID,
    FULL_FINETUNE_SMOKE_ID,
    PRIMARY_MODE,
    SMOKE_MODE,
    full_finetune_vram_preflight,
    load_full_checkpoint,
    train_full_finetune,
    validate_full_finetune_spec,
)


DEFAULT_DATA = ROOT / "data" / "common-curriculum-002" / "u128-train96"
DEFAULT_COMPARATOR_RUN = ARTIFACTS / "runs" / EXPECTED_COMPARATOR_EVIDENCE["run_id"]
EXPECTED_DATA_HASHES = {
    "train.jsonl": "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f",
    "manifest.json": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
    "state_manifest.jsonl": "4ab23b5cd883d8ad9b542befadc23c2aec3a3d631b78f239bb551ca998fd6a3c",
    "canonical.jsonl": "6bfefc11ce390048ca3cfcb8d44db4f583501d5174dc89d3140caccc49cf958f",
    "universe.json": "1256cd1c1075246251cafb4d01612dae26a73808a4915c3d88006478f3f736ac",
    "train_secrets.json": "e8ace1e06a6f35a1b600702099c029e232b6124fe76265a5cf4da2d981386a4e",
    "dev_secrets.json": "e94dea81d06f464a55ea7463b36837c998d1e405ef3f1e6e0500c78ea627c8a2",
}
EXPECTED_PROTOCOL_LOCK_FILE_SHA256 = "afb074f6aafa5d30b16595890c1087556c0b8078c92bceacc283a296c3a462e7"
EXPECTED_CHECKPOINT_TREE_HASHES = {
    "step-000150": "6b40afa18ff3d737adafe03d5f8c282d5dbc5ee230aed4b80e8048a6ca2b59cc",
    "step-000300": "c4e6c516ca3d5fa80955da591379121a372e5ea1905c6e90fb5f433b3c04ba08",
    "step-000450": "38d1bb27cf0474d0e38b969338e595fd32ea9b51d249debb6fbe25dcba02450e",
    "step-000600": "7deadbb541d4059d5e90958cf4d83670da490b97b63512509fe45498da6c665a",
    "final": EXPECTED_COMPARATOR_EVIDENCE["final_adapter_tree_sha256"],
}
EXPECTED_ALLOWED_WORDS_SHA256 = "9df5dad1b44cf1b9e0fa7c3ebff94d69ff7efb59f8691ac88e74e1c7b3da121e"
EXPECTED_RETENTION_SHA256 = "f510af87dfb67f3200ec39982e3470a5b08c40ac3e444f35712c603da07975c6"
EXPECTED_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _project_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _tree_digest(directory: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise AssertionError(f"comparator adapter tree contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not files:
        raise AssertionError(f"empty comparator adapter tree: {directory}")
    return sha256_text(canonical_json(files)), files


def audit_protocol() -> dict[str, Any]:
    lock_path = DATA / "protocol_lock.json"
    lock = read_json(lock_path)
    if sha256_file(lock_path) != EXPECTED_PROTOCOL_LOCK_FILE_SHA256:
        raise AssertionError("WORDLE-PROTOCOL-002 lock file hash drift")
    if lock.get("protocol_id") != EXPECTED_PROTOCOL_ID or lock.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError("WORDLE-PROTOCOL-002 identity drift")
    actual_components = {
        name: sha256_file(ROOT / "wordle_lab" / "protocol" / name)
        for name in lock["component_files"]
    }
    if actual_components != lock["component_files"]:
        raise AssertionError("WORDLE-PROTOCOL-002 component drift")
    allowed_path = ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt"
    retention_path = DATA / "retention_probes_v1.jsonl"
    if sha256_file(allowed_path) != EXPECTED_ALLOWED_WORDS_SHA256:
        raise AssertionError("frozen allowed-word list drift")
    if sha256_file(retention_path) != EXPECTED_RETENTION_SHA256:
        raise AssertionError("frozen retention probes drift")
    return {
        "status": "passed",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "lock_file_sha256": EXPECTED_PROTOCOL_LOCK_FILE_SHA256,
        "component_files": actual_components,
        "allowed_words_sha256": EXPECTED_ALLOWED_WORDS_SHA256,
        "retention_probes_sha256": EXPECTED_RETENTION_SHA256,
        "locked_test_access": False,
    }


def audit_lora_comparator(directory: Path = DEFAULT_COMPARATOR_RUN) -> dict[str, Any]:
    """Pin the completed native-Transformers PEFT LoRA arm byte-for-byte."""
    directory = Path(directory)
    required = ("spec.json", "dataset_manifest.json", "accounting.json", "summary.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"native LoRA comparator artifacts missing: {missing}")
    artifact_hashes = {name: sha256_file(directory / name) for name in required}
    expected_hashes = {
        "spec.json": EXPECTED_COMPARATOR_EVIDENCE["spec_sha256"],
        "dataset_manifest.json": EXPECTED_COMPARATOR_EVIDENCE["dataset_manifest_sha256"],
        "accounting.json": EXPECTED_COMPARATOR_EVIDENCE["accounting_sha256"],
        "summary.json": EXPECTED_COMPARATOR_EVIDENCE["summary_sha256"],
    }
    if artifact_hashes != expected_hashes:
        raise AssertionError(f"native LoRA comparator artifact hash drift: {artifact_hashes}")

    spec = read_json(directory / "spec.json")
    dataset_manifest = read_json(directory / "dataset_manifest.json")
    accounting = read_json(directory / "accounting.json")
    expected_spec = {
        "method": "sft",
        "representation": "common_balanced_curriculum",
        "prompt_version": "explicit-constraints-v2-compact",
        "seed": 2026,
        "max_steps": 600,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": 320,
        "word_token_weight": 8.0,
        "lora": EXPECTED_LORA,
    }
    drift = {key: (value, spec.get(key)) for key, value in expected_spec.items() if spec.get(key) != value}
    if spec.get("model", {}).get("model_id") != EXPECTED_MODEL_ID or spec.get("model", {}).get("revision") != EXPECTED_MODEL_REVISION:
        drift["model"] = ({"model_id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION}, spec.get("model"))
    if spec.get("curriculum") != dataset_manifest:
        drift["dataset_manifest"] = ("identical to spec.curriculum", "different")
    expected_accounting = {
        "train_examples": 512,
        "optimizer_steps": 600,
        "effective_batch_size": 4,
        "total_parameters": EXPECTED_LORA_WRAPPED_PARAMETER_COUNT,
        "trainable_parameters": 3_796_992,
        "checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
        "loss_mode": "word_focused",
        "word_token_weight": 8.0,
    }
    for key, value in expected_accounting.items():
        if accounting.get(key) != value:
            drift[f"accounting.{key}"] = (value, accounting.get(key))
    if drift:
        raise AssertionError(f"native LoRA comparator recipe drift: {drift}")

    checkpoint_trees: dict[str, str] = {}
    final_files: dict[str, dict[str, Any]] = {}
    for checkpoint, expected_digest in EXPECTED_CHECKPOINT_TREE_HASHES.items():
        digest, files = _tree_digest(directory / "checkpoints" / checkpoint)
        if digest != expected_digest:
            raise AssertionError(f"native LoRA comparator {checkpoint} adapter tree drift")
        checkpoint_trees[checkpoint] = digest
        if checkpoint == "final":
            final_files = files
    return {
        "status": "passed",
        "run_id": EXPECTED_COMPARATOR_EVIDENCE["run_id"],
        "run_directory": _project_path(directory),
        "backend": "native_transformers_peft_lora",
        "spec_sha256": artifact_hashes["spec.json"],
        "dataset_manifest_sha256": artifact_hashes["dataset_manifest.json"],
        "accounting_sha256": artifact_hashes["accounting.json"],
        "summary_sha256": artifact_hashes["summary.json"],
        "final_adapter_tree_sha256": checkpoint_trees["final"],
        "checkpoint_adapter_tree_sha256": checkpoint_trees,
        "final_adapter_files": final_files,
        "recipe": expected_spec,
        "accounting": expected_accounting,
        "tree_hash_algorithm": "sha256(canonical_json({relative_path: {bytes, sha256}}))",
        "provenance_limitations": [
            "artifact directory is git-ignored and has no source-commit binding",
            "optimizer, scheduler, RNG, and trainer state were not saved for deterministic resume",
            "historical summary curriculum label was corrected after evaluation; spec and dataset hashes are primary evidence",
            "historical comparator summary has no retention result or embedded protocol hash",
        ],
        "locked_test_access": False,
    }


def audit_balanced_source(directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    directory = Path(directory)
    required = list(EXPECTED_DATA_HASHES)
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"balanced-002 files missing: {missing}")
    actual_hashes = {name: sha256_file(directory / name) for name in required}
    if actual_hashes != EXPECTED_DATA_HASHES:
        mismatched = {
            name: {"expected": EXPECTED_DATA_HASHES[name], "actual": actual_hashes.get(name)}
            for name in EXPECTED_DATA_HASHES
            if actual_hashes.get(name) != EXPECTED_DATA_HASHES[name]
        }
        raise AssertionError(f"balanced-002 pinned content hash drift: {mismatched}")
    manifest = read_json(directory / "manifest.json")
    rows = read_jsonl(directory / "train.jsonl")
    train = set(read_json(directory / "train_secrets.json"))
    dev = set(read_json(directory / "dev_secrets.json"))
    if train & dev:
        raise AssertionError("balanced-002 train/dev secrets overlap")
    if sha256_file(directory / "train.jsonl") != manifest["rendered_sha256"]:
        raise AssertionError("balanced-002 rendered hash differs from its manifest")
    if sha256_file(directory / "state_manifest.jsonl") != manifest["state_manifest_sha256"]:
        raise AssertionError("balanced-002 state-manifest hash differs from its manifest")
    if len(rows) != 512 or manifest["curriculum_id"] != "COMMON-WORD-CURRICULUM-002":
        raise AssertionError("full fine-tune requires the exact 512-row balanced-002 curriculum")
    labelled = {row["source_state"]["secret_answer"] for row in rows}
    if not labelled <= train or labelled & dev:
        raise AssertionError("balanced-002 labels are not training-only")
    return {
        "status": "passed",
        "directory": _project_path(directory),
        "rows": len(rows),
        "rendered_sha256": manifest["rendered_sha256"],
        "state_manifest_sha256": manifest["state_manifest_sha256"],
        "manifest_sha256": actual_hashes["manifest.json"],
        "canonical_sha256": actual_hashes["canonical.jsonl"],
        "universe_sha256": actual_hashes["universe.json"],
        "train_secrets_sha256": actual_hashes["train_secrets.json"],
        "dev_secrets_sha256": actual_hashes["dev_secrets.json"],
        "files": actual_hashes,
        "train_secret_count": len(train),
        "dev_secret_count": len(dev),
        "locked_test_access": False,
    }


def matched_full_spec(
    directory: Path = DEFAULT_DATA,
    *,
    steps: int = 600,
    seed: int = 2026,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    accumulation: int = 1,
    gradient_checkpointing: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    audit = audit_balanced_source(directory)
    protocol = audit_protocol()
    comparator = audit_lora_comparator()
    mode = SMOKE_MODE if smoke else PRIMARY_MODE
    checkpoint_steps = [] if smoke else list(EXPECTED_CHECKPOINT_STEPS)
    checkpoint_fractions = [] if smoke else list(EXPECTED_CHECKPOINT_FRACTIONS)
    spec = {
        "experiment_id": FULL_FINETUNE_SMOKE_ID if smoke else FULL_FINETUNE_EXPERIMENT_ID,
        "experiment_mode": mode,
        "backend": FULL_FINETUNE_BACKEND_ID,
        "method": "full_parameter_sft",
        "representation": "common_balanced_curriculum",
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "seed": seed,
        "max_steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_fractions": checkpoint_fractions,
        "word_token_weight": 8.0,
        "gradient_checkpointing": gradient_checkpointing,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "optimizer": "torch.optim.AdamW",
        "scheduler": "linear_warmup_5pct_cosine",
        "model": model_metadata(),
        "data": audit,
        "protocol": protocol,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "evaluation": {
            "split": "balanced_002_dev_32",
            "dev_games": 32,
            "diagnostic_items": 128,
            "prompt_variant": "explicit_feedback",
            "decoder": "greedy",
            "generation": {"do_sample": False, "max_new_tokens": 128, "use_cache": True},
            "allowed_words_sha256": EXPECTED_ALLOWED_WORDS_SHA256,
            "retention_probes_sha256": EXPECTED_RETENTION_SHA256,
        },
        "comparator": comparator,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
        "smoke": smoke,
        "matched_comparison": not smoke,
        "comparison_contract": {
            "matched": not smoke,
            "comparator_backend": "native_transformers_peft_lora",
            "comparator_run_id": comparator["run_id"],
            "matched_to_lora": [
                "base_model_revision",
                "training_rows_and_order",
                "seed",
                "learning_rate",
                "batch_size_and_accumulation",
                "effective_batch_size",
                "max_length",
                "word_token_weighted_objective",
                "optimizer_and_scheduler",
                "prompt_representation",
                "development_evaluation",
                "checkpoint_doses",
            ],
            "intended_difference": "all model parameters trainable instead of rank-16 LoRA parameters",
            "interpretation": (
                "two-step allocation benchmark only; not a matched comparison"
                if smoke
                else "single-seed matched trainable-scope ablation"
            ),
            "declared_unavoidable_difference": (
                "historical PEFT LoRA trainables were FP32 while full-model trainables use BF16; "
                "the forward base revision and BF16 model path remain fixed"
            ),
        },
    }
    validate_full_finetune_spec(spec)
    return spec


def smoke_full_spec(directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    """Build the fixed two-step allocation benchmark, explicitly outside the matched cell."""
    return matched_full_spec(
        directory,
        steps=2,
        seed=2026,
        learning_rate=5e-5,
        batch_size=1,
        accumulation=1,
        gradient_checkpointing=False,
        smoke=True,
    )


def prepare_run(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> Path:
    validate_full_finetune_spec(spec)
    assert_evaluation_data_binding(spec, directory)
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    label = "smoke" if spec.get("smoke") else "primary"
    run_dir = ARTIFACTS / "runs" / f"full-finetune-balanced-word-{label}-s{spec['seed']}-{digest}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", spec["data"])
    write_json(run_dir / "comparison_manifest.json", comparison_manifest_for_spec(spec))
    return run_dir


def _binding_view(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": audit["rows"],
        "rendered_sha256": audit["rendered_sha256"],
        "state_manifest_sha256": audit["state_manifest_sha256"],
        "manifest_sha256": audit["manifest_sha256"],
        "canonical_sha256": audit["canonical_sha256"],
        "universe_sha256": audit["universe_sha256"],
        "train_secrets_sha256": audit["train_secrets_sha256"],
        "dev_secrets_sha256": audit["dev_secrets_sha256"],
        "train_secret_count": audit["train_secret_count"],
        "dev_secret_count": audit["dev_secret_count"],
        "locked_test_access": audit["locked_test_access"],
    }


def assert_evaluation_data_binding(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    """Require evaluation inputs to be byte-identical to the training source in the spec."""
    current = audit_balanced_source(directory)
    expected = _binding_view(spec["data"])
    observed = _binding_view(current)
    if observed != expected:
        raise AssertionError(f"evaluation data does not match run spec: expected={expected}, actual={observed}")
    protocol = audit_protocol()
    if protocol != spec["protocol"]:
        raise AssertionError("evaluation protocol does not match run spec")
    return current


def comparison_manifest_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "gemma-270m-lora-vs-full-v1",
        "experiment_id": spec["experiment_id"],
        "experiment_mode": spec["experiment_mode"],
        "matched": spec["matched_comparison"],
        "full_spec_sha256": sha256_text(canonical_json(spec)),
        "native_transformers_lora": spec["comparator"],
        "full_parameter_condition": {
            "backend": spec["backend"],
            "model": spec["model"],
            "data": _binding_view(spec["data"]),
            "protocol": spec["protocol"],
            "seed": spec["seed"],
            "max_steps": spec["max_steps"],
            "learning_rate": spec["learning_rate"],
            "batch_size": spec["batch_size"],
            "gradient_accumulation_steps": spec["gradient_accumulation_steps"],
            "effective_batch_size": spec["effective_batch_size"],
            "max_length": spec["max_length"],
            "word_token_weight": spec["word_token_weight"],
            "optimizer": spec["optimizer"],
            "scheduler": spec["scheduler"],
            "checkpoint_steps": spec["checkpoint_steps"],
            "trainable_scope": "all_model_parameters",
        },
        "comparison_contract": spec["comparison_contract"],
        "locked_test_access": False,
    }


def experiment_preflight() -> dict[str, Any]:
    """Run only read-only audits and CUDA memory queries; never load the model."""
    vram = full_finetune_vram_preflight()
    return {
        "status": "ready" if vram["ready"] else "blocked",
        "vram": vram,
        "data": audit_balanced_source(),
        "protocol": audit_protocol(),
        "comparator": audit_lora_comparator(),
        "locked_test_access": False,
    }


def run_training(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    validate_full_finetune_spec(spec)
    assert_evaluation_data_binding(spec, directory)
    comparator = audit_lora_comparator()
    if comparator["spec_sha256"] != spec["comparator"]["spec_sha256"]:
        raise AssertionError("native LoRA comparator no longer matches the run spec")
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError(f"full fine-tune preflight blocked: {preflight['status']}")
    rows = read_jsonl(Path(directory) / "train.jsonl")
    run_dir = prepare_run(spec, directory)
    write_json(run_dir / "preflight.json", preflight)
    set_seed(int(spec["seed"]))
    model = tokenizer = None
    try:
        model, tokenizer, accounting = train_full_finetune(rows, run_dir, spec)
        metrics = read_jsonl(run_dir / "train_metrics.jsonl")
        summary = {
            "status": "smoke_memory_benchmark_completed" if spec["smoke"] else "matched_full_training_completed",
            "experiment_id": spec["experiment_id"],
            "experiment_mode": spec["experiment_mode"],
            "matched_comparison": spec["matched_comparison"],
            "run_dir": str(run_dir),
            "initial_loss": metrics[0]["train_loss"],
            "final_loss": metrics[-1]["train_loss"],
            "accounting": accounting,
            "preflight": preflight,
            "comparator": spec["comparator"],
            "locked_test_access": False,
        }
        write_json(run_dir / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_full_checkpoint(
    run_dir: Path,
    checkpoint: str,
    *,
    directory: Path = DEFAULT_DATA,
    dev_games: int = 32,
) -> dict[str, Any]:
    run_dir, directory = Path(run_dir), Path(directory)
    spec = read_json(run_dir / "spec.json")
    validate_full_finetune_spec(spec)
    if spec.get("experiment_mode") != PRIMARY_MODE or spec.get("smoke"):
        raise ValueError("smoke runs are memory checks and are not policy-evaluated")
    if dev_games != spec["evaluation"]["dev_games"]:
        raise ValueError(f"matched evaluation requires exactly {spec['evaluation']['dev_games']} development games")
    data_audit = assert_evaluation_data_binding(spec, directory)
    comparator = audit_lora_comparator()
    if comparator["final_adapter_tree_sha256"] != spec["comparator"]["final_adapter_tree_sha256"]:
        raise AssertionError("native LoRA comparator adapter drift before paired evaluation")
    expected_checkpoints = {f"step-{step:06d}" for step in spec["checkpoint_steps"]}
    if checkpoint not in expected_checkpoints:
        raise ValueError(f"checkpoint must be one of the matched doses: {sorted(expected_checkpoints)}")
    checkpoint_dir = run_dir / "checkpoints" / checkpoint
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    summary_path = run_dir / f"eval-{checkpoint}-summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {summary_path}")
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[:dev_games]
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, tokenizer = load_full_checkpoint(checkpoint_dir)
    previous_messages = generation.inference_messages
    previous_generation = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(int(spec["seed"]))
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(spec["evaluation"]["generation"])
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        dev_records = generate_canonical_states(
            dev_answers,
            "common_dev_diagnostic",
            spec["evaluation"]["diagnostic_items"],
            seed=int(spec["seed"]),
            answer_vocabulary=universe,
        )
        diagnostic_parent = run_dir / f"eval-{checkpoint}"
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            read_jsonl(directory / "canonical.jsonl"),
            allowed,
            universe,
            diagnostic_parent,
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(run_dir / f"eval-{checkpoint}-games.jsonl", games)
        write_jsonl(run_dir / f"eval-{checkpoint}-retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": FULL_FINETUNE_EXPERIMENT_ID,
            "experiment_mode": PRIMARY_MODE,
            "matched_comparison": True,
            "spec_sha256": sha256_text(canonical_json(spec)),
            "checkpoint": checkpoint,
            "split": spec["evaluation"]["split"],
            "decoder": spec["evaluation"]["decoder"],
            "prompt_variant": spec["evaluation"]["prompt_variant"],
            "evaluation_data": _binding_view(data_audit),
            "protocol": spec["protocol"],
            "comparator": spec["comparator"],
            "locked_test_access": False,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(summary_path, summary)
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_generation)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _comparison_metrics(summary: dict[str, Any], *, comparator: bool) -> dict[str, Any]:
    gameplay = summary if comparator else summary["gameplay"]
    diagnostics = summary["state_diagnostics"] if comparator else summary["diagnostics"]
    turn_2 = diagnostics.get("by_turn", {}).get("2", {})
    return {
        "wins": gameplay.get("wins"),
        "win_rate": gameplay.get("win_rate"),
        "terminal_marker_compliance": gameplay.get("terminal_marker_compliance"),
        "invalid_guess_rate": gameplay.get("invalid_guess_rate"),
        "repeat_guess_rate": gameplay.get("repeat_guess_rate"),
        "posterior_constraint_violation_rate": diagnostics.get("posterior_constraint_violation_rate"),
        "turn_2_posterior_constraint_violation_rate": turn_2.get("posterior_constraint_violation_rate"),
        "singleton_answer_accuracy": diagnostics.get("singleton_answer_accuracy"),
        "action_target_accuracy": diagnostics.get("action_target_accuracy"),
    }


def build_paired_comparison_summary(
    run_dir: Path,
    checkpoint: str,
    *,
    comparator_dir: Path = DEFAULT_COMPARATOR_RUN,
) -> dict[str, Any]:
    """Join the pinned historical LoRA metrics with an already-evaluated full dose."""
    run_dir = Path(run_dir)
    spec = read_json(run_dir / "spec.json")
    validate_full_finetune_spec(spec)
    if spec["experiment_mode"] != PRIMARY_MODE:
        raise ValueError("only a matched primary run can produce a paired comparison")
    if checkpoint != "step-000600":
        raise ValueError("the pinned historical LoRA run has comparable metrics only for step-000600/final")
    comparator_audit = audit_lora_comparator(comparator_dir)
    if comparator_audit["spec_sha256"] != spec["comparator"]["spec_sha256"]:
        raise AssertionError("paired comparator differs from the run spec")
    full_path = run_dir / f"eval-{checkpoint}-summary.json"
    if not full_path.is_file():
        raise FileNotFoundError(f"full checkpoint must be evaluated before comparison: {full_path}")
    full_summary = read_json(full_path)
    expected_spec_sha256 = sha256_text(canonical_json(spec))
    full_binding = {
        "spec_sha256": full_summary.get("spec_sha256"),
        "split": full_summary.get("split"),
        "checkpoint": full_summary.get("checkpoint"),
        "matched_comparison": full_summary.get("matched_comparison"),
        "evaluation_data": full_summary.get("evaluation_data"),
        "protocol": full_summary.get("protocol"),
        "locked_test_access": full_summary.get("locked_test_access"),
    }
    expected_binding = {
        "spec_sha256": expected_spec_sha256,
        "split": spec["evaluation"]["split"],
        "checkpoint": checkpoint,
        "matched_comparison": True,
        "evaluation_data": _binding_view(spec["data"]),
        "protocol": spec["protocol"],
        "locked_test_access": False,
    }
    if full_binding != expected_binding:
        raise AssertionError("full evaluation summary is not bound to the matched run spec")
    comparator_summary = read_json(Path(comparator_dir) / "summary.json")
    lora_metrics = _comparison_metrics(comparator_summary, comparator=True)
    full_metrics = _comparison_metrics(full_summary, comparator=False)
    deltas = {
        key: (full_metrics[key] - lora_metrics[key])
        if isinstance(full_metrics[key], (int, float)) and isinstance(lora_metrics[key], (int, float))
        else None
        for key in lora_metrics
    }
    output_path = run_dir / f"paired-{checkpoint}-summary.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing paired comparison: {output_path}")
    result = {
        "status": "paired_development_comparison_ready",
        "experiment_id": FULL_FINETUNE_EXPERIMENT_ID,
        "checkpoint": checkpoint,
        "split": spec["evaluation"]["split"],
        "single_seed": True,
        "causal_claim_boundary": "matched single-seed diagnostic; not replicated superiority",
        "intended_difference": spec["comparison_contract"]["intended_difference"],
        "native_transformers_lora": {
            "run_id": comparator_audit["run_id"],
            "summary_sha256": comparator_audit["summary_sha256"],
            "metrics": lora_metrics,
            "provenance_limitations": comparator_audit["provenance_limitations"],
        },
        "full_parameter": {
            "run_directory": _project_path(run_dir),
            "evaluation_summary_sha256": sha256_file(full_path),
            "metrics": full_metrics,
        },
        "delta_full_minus_lora": deltas,
        "retention_comparison": "unavailable: historical comparator summary has no retention result",
        "locked_test_access": False,
    }
    write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matched Gemma 270M full-fine-tuning capacity experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="read-only artifact, protocol, data, and free-VRAM audit")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--steps", type=int, default=600)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--accumulation", type=int, default=1)
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    eval_parser.add_argument("--dev-games", type=int, default=32)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--run-dir", type=Path, required=True)
    compare_parser.add_argument("--checkpoint", required=True)
    compare_parser.add_argument("--comparator-dir", type=Path, default=DEFAULT_COMPARATOR_RUN)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = experiment_preflight()
    elif args.command == "train":
        if args.smoke:
            nondefaults = {
                "steps": args.steps != 600,
                "seed": args.seed != 2026,
                "learning_rate": args.learning_rate != 5e-5,
                "batch_size": args.batch_size != 4,
                "accumulation": args.accumulation != 1,
                "gradient_checkpointing": args.gradient_checkpointing,
            }
            changed = sorted(key for key, value in nondefaults.items() if value)
            if changed:
                parser.error(f"--smoke is a fixed two-step non-matched benchmark; remove overrides: {changed}")
            spec = smoke_full_spec(args.data_dir)
        else:
            spec = matched_full_spec(
                args.data_dir,
                steps=args.steps,
                seed=args.seed,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                accumulation=args.accumulation,
                gradient_checkpointing=args.gradient_checkpointing,
                smoke=False,
            )
        if args.dry_run:
            preflight = full_finetune_vram_preflight()
            result = {
                "status": "dry_run_ready" if preflight["ready"] else "dry_run_blocked_preflight",
                "spec": spec,
                "preflight": preflight,
                "model_loaded": False,
            }
        else:
            result = run_training(spec, args.data_dir)
    elif args.command == "evaluate":
        result = evaluate_full_checkpoint(args.run_dir, args.checkpoint, directory=args.data_dir, dev_games=args.dev_games)
    else:
        result = build_paired_comparison_summary(args.run_dir, args.checkpoint, comparator_dir=args.comparator_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
