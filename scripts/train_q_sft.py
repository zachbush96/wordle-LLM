from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import set_seed, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.q_sft import train_q_sft, validate_q_sft_rows
from wordle_lab.standalone import assert_gemma_parent_adapter, base_spec, comparison_context, dry_run_summary, load_jsonl_required, prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone offline Q-SFT with frozen Bellman snapshots")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--offline-transitions", type=Path, required=True,
                        help="Train-only JSONL containing prompts/completions and frozen Bellman targets")
    parser.add_argument("--parent-adapter", type=Path, required=True); parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--discount", type=float, default=0.99); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, _, _, audit = comparison_context(args.data_dir, args.partition)
    assert_gemma_parent_adapter(args.parent_adapter)
    rows = load_jsonl_required(args.offline_transitions); targets = validate_q_sft_rows(rows, args.discount)
    spec = {**base_spec("q_sft", args.partition, args.seed, args.steps, args.learning_rate), "discount": args.discount,
            "parent_checkpoint": str(args.parent_adapter), "bellman_targets": "frozen_offline_snapshots"}
    extra = {"audit": audit, "mean_bellman_target": sum(targets) / len(targets), "offline_transitions": str(args.offline_transitions)}
    if args.dry_run: print(json.dumps(dry_run_summary("q_sft", args.partition, data_dir, rows, spec, **extra), indent=2, sort_keys=True)); return 0
    run_dir = prepare_run("q_sft", args.partition, spec, data_dir); set_seed(args.seed)
    _, accounting = train_q_sft(rows, args.parent_adapter, run_dir, spec)
    summary = {"status": "trained", "run_dir": str(run_dir), "accounting": accounting, **extra}; write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
