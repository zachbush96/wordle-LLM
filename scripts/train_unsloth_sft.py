from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Unsloth must patch Transformers before the project imports model classes.
import unsloth  # noqa: F401

import _bootstrap  # noqa: F401
from wordle_lab.common import canonical_json, set_seed, sha256_file, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.unsloth_sft import UNSLOTH_BACKEND_ID, select_nested_rows, train_unsloth_sft, unsloth_environment
from wordle_lab.standalone import base_spec, comparison_context, prepare_run


def build_spec(partition: str, seed: int, steps: int, learning_rate: float, train_states: int | None) -> dict:
    spec = base_spec("unsloth-sft", partition, seed, steps, learning_rate)
    spec.update(
        {
            "backend": UNSLOTH_BACKEND_ID,
            # The 16 GB 4060 Ti fits the declared effective batch directly;
            # one launch is materially faster than 4 x 4-step accumulation.
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "train_state_limit": train_states,
            "word_token_weight": 1.0,
            "quantization": "none_16bit",
            "gradient_checkpointing": "unsloth",
        }
    )
    spec["lora"]["dropout"] = 0.0
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Audited Unsloth LoRA SFT for pinned Gemma 3 270M")
    parser.add_argument("--partition", choices=PARTITIONS, default="non_reasoning_single_step")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-states", type=int)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir, all_rows, _, audit = comparison_context(args.data_dir, args.partition)
    rows = select_nested_rows(all_rows, args.train_states)
    spec = build_spec(args.partition, args.seed, args.steps, args.learning_rate, args.train_states)
    selected_hash = hashlib.sha256("\n".join(canonical_json(row) for row in rows).encode("utf-8")).hexdigest()
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_passed",
            "spec": spec,
            "training_rows": len(rows),
            "selected_rows_sha256": selected_hash,
            "source_partition_sha256": sha256_file(data_dir / f"{args.partition}.jsonl"),
            "data_audit": audit,
            "environment": unsloth_environment(),
        }, indent=2, sort_keys=True))
        return 0

    run_dir = prepare_run("unsloth-sft", args.partition, spec, data_dir)
    write_json(run_dir / "dataset_manifest.json", {
        "partition": args.partition,
        "training_rows": len(rows),
        "selected_rows_sha256": selected_hash,
        "source_partition_sha256": sha256_file(data_dir / f"{args.partition}.jsonl"),
        "audit": audit,
        "dev_probe_role": "evaluation_only_never_training",
    })
    set_seed(args.seed)
    _, _, accounting = train_unsloth_sft(rows, run_dir, spec)
    summary = {"status": "trained", "run_dir": str(run_dir), "accounting": accounting}
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
