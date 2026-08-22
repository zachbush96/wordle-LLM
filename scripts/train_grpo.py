from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import read_json, set_seed, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.grpo import train_grpo
from wordle_lab.methods.reward_rubrics import NOTEBOOKLM_REWARD_VERSION
from wordle_lab.standalone import assert_gemma_parent_adapter, base_spec, comparison_context, dry_run_summary, prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone warm-started GRPO")
    parser.add_argument("--partition", choices=PARTITIONS, required=True, help="Matched SFT parent/data lineage")
    parser.add_argument("--data-dir", type=Path); parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-6); parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, _, sources, audit = comparison_context(args.data_dir, args.partition)
    assert_gemma_parent_adapter(args.parent_adapter)
    spec = {**base_spec("grpo", args.partition, args.seed, args.steps, args.learning_rate), "batch_size": 2,
            "gradient_accumulation_steps": 8, "group_size": args.group_size, "temperature": 1.0,
            "parent_checkpoint": str(args.parent_adapter), "reward_rubric": {"version": NOTEBOOKLM_REWARD_VERSION,
            "weights": {"format": 0.05, "validity": 0.20, "completion": 1.0, "repetition": -0.30,
                        "green_violation": -0.40, "missing_yellow": -0.25, "gray_reuse": -0.20}}}
    if args.dry_run: print(json.dumps(dry_run_summary("grpo", args.partition, data_dir, sources, spec, audit=audit), indent=2, sort_keys=True)); return 0
    run_dir = prepare_run("grpo", args.partition, spec, data_dir); set_seed(args.seed)
    universe = read_json(data_dir / "universe.json"); train_secrets = read_json(data_dir / "train_secrets.json")
    train_grpo(sources, args.parent_adapter, run_dir, spec, universe, train_secrets)
    summary = {"status": "trained", "run_dir": str(run_dir), "source_states": len(sources)}; write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
