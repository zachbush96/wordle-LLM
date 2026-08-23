from __future__ import annotations

"""Isolated Unsloth launcher for the two tiny memorization cells."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

# Actual training keeps Unsloth's required import ordering. Read-only imports,
# dry runs, and unit tests do not patch Transformers or initialize a model.
if __name__ == "__main__" and "--dataset" in sys.argv[1:] and "--dry-run" not in sys.argv[1:]:
    import unsloth  # noqa: F401

from wordle_lab.common import ARTIFACTS, ROOT, canonical_json, read_json, read_jsonl, set_seed, sha256_file, write_json
from wordle_lab.methods.unsloth_sft import (
    UNSLOTH_WEIGHTED_BACKEND_ID,
    train_unsloth_sft,
    unsloth_environment,
    validate_unsloth_objective,
)
from wordle_lab.models import model_metadata

from .tiny_overfit import (
    DEFAULT_OUTPUT,
    TINY_OVERFIT_ID,
    build_tiny_overfit_bundle,
    load_audited_cell,
)


def tiny_overfit_spec(
    dataset: str,
    rows_path: Path,
    *,
    steps: int = 400,
    seed: int = 2026,
    learning_rate: float = 5e-5,
) -> dict:
    if dataset not in {"general", "singleton"}:
        raise ValueError("dataset must be general or singleton")
    if steps < 1:
        raise ValueError("steps must be positive")
    rows_path = Path(rows_path).resolve()
    rows, _, data_audit = load_audited_cell(rows_path)
    expected_cell = "general_32" if dataset == "general" else "singleton_32"
    if data_audit["cell"] != expected_cell:
        raise ValueError(f"dataset {dataset} does not match audited cell {data_audit['cell']}")
    protocol_lock = read_json(ROOT / "data" / "protocol-002" / "protocol_lock.json")
    spec = {
        "experiment_id": TINY_OVERFIT_ID,
        "method": "unsloth_tiny_overfit",
        "backend": UNSLOTH_WEIGHTED_BACKEND_ID,
        "representation": f"tiny_overfit_{dataset}_32",
        "dataset": dataset,
        "seed": seed,
        "max_steps": steps,
        "learning_rate": learning_rate,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 16,
        "max_length": 320,
        "word_token_weight": 8.0,
        "warmup_fraction": 0.05,
        "quantization": "none_16bit",
        "gradient_checkpointing": "unsloth",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "model": model_metadata(),
        "data": {
            "path": str(rows_path),
            "rows": len(rows),
            "sha256": sha256_file(rows_path),
            "cell": data_audit["cell"],
            "bundle_manifest_sha256": data_audit["bundle_manifest_sha256"],
            "source_manifest_sha256": data_audit["source"]["hashes"]["manifest.json"],
            "source_rows_sha256": data_audit["source"]["hashes"]["train.jsonl"],
            "universe_sha256": data_audit["universe_sha256"],
            "role": "training-set memorization only",
        },
        "protocol_id": protocol_lock["protocol_id"],
        "protocol_sha256": protocol_lock["protocol_sha256"],
        "locked_test_access": False,
        "candidate_injection": False,
        "reranking": False,
        "output_repair": False,
        "acceptance": {
            "loss_goal": 0.05,
            "natural_exact_accuracy_goal": 0.99,
            "singleton_exact_accuracy_goal": 0.99 if dataset == "singleton" else None,
            "contrast_pair_exact_accuracy_goal": 0.99 if dataset == "general" else None,
        },
    }
    validate_unsloth_objective(spec)
    return spec


def prepare_run(
    spec: dict,
    rows_path: Path,
    *,
    output_root: Path = ARTIFACTS / "runs",
) -> Path:
    rows_path = Path(rows_path).resolve()
    _, _, data_audit = load_audited_cell(rows_path)
    if spec.get("data", {}).get("sha256") != sha256_file(rows_path):
        raise RuntimeError("tiny-overfit spec rows hash does not match the audited cell")
    if spec.get("data", {}).get("bundle_manifest_sha256") != data_audit["bundle_manifest_sha256"]:
        raise RuntimeError("tiny-overfit spec bundle provenance does not match the audited cell")
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = Path(output_root).resolve() / f"tiny-overfit-{spec['dataset']}-s{spec['seed']}-{digest}"
    if run_dir.exists():
        raise RuntimeError(f"deterministic tiny-overfit run already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "spec.json", spec)
    write_json(
        run_dir / "dataset_manifest.json",
        {
            "experiment_id": TINY_OVERFIT_ID,
            "dataset": spec["dataset"],
            "rows": spec["data"]["rows"],
            "rows_sha256": sha256_file(rows_path),
            "cell": data_audit["cell"],
            "suite_manifest_sha256": data_audit["bundle_manifest_sha256"],
            "source_manifest_sha256": data_audit["source"]["hashes"]["manifest.json"],
            "source_rows_sha256": data_audit["source"]["hashes"]["train.jsonl"],
            "universe_sha256": data_audit["universe_sha256"],
            "locked_test_access": False,
        },
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Gemma on exactly 32 memorization states")
    parser.add_argument("--dataset", choices=("general", "singleton"), required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    build_tiny_overfit_bundle()
    rows_path = DEFAULT_OUTPUT / ("general_32.jsonl" if args.dataset == "general" else "singleton_32.jsonl")
    rows = read_jsonl(rows_path)
    spec = tiny_overfit_spec(
        args.dataset,
        rows_path,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "spec": spec, "environment": unsloth_environment()}, indent=2, sort_keys=True))
        return 0
    run_dir = prepare_run(spec, rows_path)
    set_seed(args.seed)
    model = tokenizer = None
    try:
        model, tokenizer, accounting = train_unsloth_sft(rows, run_dir, spec)
        metrics = read_jsonl(run_dir / "train_metrics.jsonl")
        final_loss = metrics[-1]["train_loss"]
        summary = {
            "status": "memorization_training_completed",
            "run_dir": str(run_dir),
            "dataset": args.dataset,
            "initial_loss": metrics[0]["train_loss"],
            "final_loss": final_loss,
            "loss_goal": spec["acceptance"]["loss_goal"],
            "loss_goal_met": final_loss <= spec["acceptance"]["loss_goal"],
            "accounting": accounting,
            "next_required_action": "ordinary-PEFT natural generation and rank/probability evaluation",
        }
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
