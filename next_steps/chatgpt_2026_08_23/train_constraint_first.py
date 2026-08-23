from __future__ import annotations

# Unsloth must patch Transformers before project model imports.
import unsloth  # noqa: F401

import argparse
import gc
import json
from pathlib import Path
from typing import Sequence

import torch

from wordle_lab.common import read_jsonl, set_seed, write_json
from wordle_lab.methods.unsloth_sft import train_unsloth_sft

from .constraint_first_policy import (
    DEFAULT_OUTPUT,
    build_constraint_first_bundle,
    constraint_policy_spec,
    prepare_run,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the legality-first Wordle policy with Unsloth")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    build_constraint_first_bundle()
    rows_path = DEFAULT_OUTPUT / "train.jsonl"
    spec = constraint_policy_spec(rows_path, steps=args.steps, seed=args.seed, learning_rate=args.learning_rate)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "spec": spec}, indent=2, sort_keys=True))
        return 0
    run_dir = prepare_run(spec)
    if (run_dir / "summary.json").exists() or (run_dir / "checkpoints").exists():
        raise RuntimeError(f"refusing to overwrite existing run at {run_dir}")
    set_seed(args.seed)
    model = tokenizer = None
    try:
        model, tokenizer, accounting = train_unsloth_sft(read_jsonl(rows_path), run_dir, spec)
        summary = {"status": "trained", "run_dir": str(run_dir), "accounting": accounting, "locked_test_access": False}
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
