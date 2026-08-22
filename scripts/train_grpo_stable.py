from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import read_json, set_seed, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.grpo import train_grpo
from wordle_lab.methods.grpo_stability import validate_virtual_support_spec
from wordle_lab.standalone import assert_gemma_parent_adapter, base_spec, comparison_context, dry_run_summary, prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone AVSPO-stabilized GRPO ablation")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--parent-adapter", type=Path, required=True); parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--group-size", type=int, default=8); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, _, sources, audit = comparison_context(args.data_dir, args.partition)
    assert_gemma_parent_adapter(args.parent_adapter)
    virtual = validate_virtual_support_spec({"enabled": True, "collapse_std_threshold": 1e-6, "alpha": 0.5,
        "zero_reward_anchor": 0.1, "adaptive_threshold_initial": 0.5, "adaptive_eta": 0.01,
        "normalization_epsilon": 1e-6, "usage": "advantage_estimation_only"})
    spec = {**base_spec("grpo", args.partition, args.seed, args.steps, args.learning_rate), "batch_size": 2,
            "gradient_accumulation_steps": 8, "group_size": args.group_size, "temperature": 1.0,
            "parent_checkpoint": str(args.parent_adapter), "virtual_support": virtual,
            "reward_rubric": {"version": "wordle-shaped-v1", "weights": {"solve": 5.0,
            "information_gain": 1.0, "oracle_regret": -1.0, "repeat": -2.0, "format": -3.0}}}
    if args.dry_run: print(json.dumps(dry_run_summary("grpo_stable", args.partition, data_dir, sources, spec, audit=audit), indent=2, sort_keys=True)); return 0
    run_dir = prepare_run("grpo-stable", args.partition, spec, data_dir); set_seed(args.seed)
    train_grpo(sources, args.parent_adapter, run_dir, spec, read_json(data_dir / "universe.json"), read_json(data_dir / "train_secrets.json"))
    summary = {"status": "trained", "run_dir": str(run_dir), "source_states": len(sources), "virtual_support": virtual}; write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
