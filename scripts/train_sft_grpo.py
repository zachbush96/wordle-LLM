from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from wordle_lab.common import read_json, read_jsonl, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.experiments.hybrid_sft_grpo import build_hybrid_plan, promotion_decision, validate_hybrid_spec
from wordle_lab.methods.grpo import train_grpo
from wordle_lab.standalone import assert_gemma_parent_adapter, comparison_context, prepare_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone gated SFT-to-GRPO continuation")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sft-checkpoint", type=Path, required=True, help="Checkpoint produced by scripts/train_sft.py")
    parser.add_argument("--dev-metrics", type=Path, required=True, help="Frozen dev-only diagnostic metrics JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/studies/sft_grpo_hybrid.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, _, sources, audit = comparison_context(args.data_dir, args.partition)
    assert_gemma_parent_adapter(args.sft_checkpoint)
    if not args.dev_metrics.exists(): raise FileNotFoundError(args.dev_metrics)
    spec = validate_hybrid_spec(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    metrics = read_json(args.dev_metrics); decision = promotion_decision(metrics, spec["promotion_gate"], args.sft_checkpoint)
    report = {"status": "dry_run_passed" if args.dry_run else "validated", "partition": args.partition,
              "model_constraint": "google/gemma-3-270m-it only", "data_audit": audit,
              "plan": build_hybrid_plan(spec), "promotion": decision, "dev_metrics": metrics}
    if args.dry_run or not decision["promote"]:
        if not args.dry_run: report["status"] = "stopped_at_sft_dev_gate"
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    grpo_spec = {**spec["stages"]["grpo"], "seed": spec["seed"], "representation": args.partition,
                 "max_length": 320, "parent_checkpoint": str(args.sft_checkpoint)}
    run_dir = prepare_run("sft-grpo", args.partition, grpo_spec, data_dir)
    train_grpo(sources, args.sft_checkpoint, run_dir, grpo_spec, read_json(data_dir / "universe.json"), read_json(data_dir / "train_secrets.json"))
    report.update({"status": "grpo_completed", "run_dir": str(run_dir)}); write_json(run_dir / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
