from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import set_seed, write_json, write_jsonl
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.dpo import train_dpo
from wordle_lab.standalone import assert_gemma_parent_adapter, base_spec, comparison_context, dry_run_summary, preference_rows, prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone matched-data DPO")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--parent-adapter", type=Path, required=True); parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, rendered, sources, audit = comparison_context(args.data_dir, args.partition)
    rows = preference_rows(rendered, sources, args.partition)
    spec = {**base_spec("dpo", args.partition, args.seed, args.steps, args.learning_rate), "beta": args.beta, "parent_checkpoint": str(args.parent_adapter)}
    assert_gemma_parent_adapter(args.parent_adapter)
    if args.dry_run: print(json.dumps(dry_run_summary("dpo", args.partition, data_dir, rows, spec, audit=audit), indent=2, sort_keys=True)); return 0
    run_dir = prepare_run("dpo", args.partition, spec, data_dir); write_jsonl(run_dir / "preferences.jsonl", rows); set_seed(args.seed)
    train_dpo(rows, args.parent_adapter, run_dir, spec); summary = {"status": "trained", "run_dir": str(run_dir), "preference_pairs": len(rows)}; write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
