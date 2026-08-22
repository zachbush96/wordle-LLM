from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import read_jsonl, write_jsonl
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.q_sft import validate_q_sft_rows
from wordle_lab.standalone import comparison_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Join matched prompts to independently frozen Q-SFT Bellman snapshots")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--snapshots", type=Path, required=True,
                        help="JSONL keyed by comparison_id, containing bellman_target or Bellman inputs; no secret/candidate fields")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--discount", type=float, default=0.99)
    args = parser.parse_args(); _, rendered, _, audit = comparison_context(args.data_dir, args.partition)
    snapshots = read_jsonl(args.snapshots); by_id = {row.get("comparison_id"): row for row in snapshots}
    if None in by_id or len(by_id) != len(snapshots): raise ValueError("snapshots require unique comparison_id values")
    allowed_snapshot_fields = {"comparison_id", "bellman_target", "reward", "terminal", "next_q_probabilities", "next_behavior_probabilities"}
    for index, snapshot in enumerate(snapshots):
        unknown = sorted(set(snapshot) - allowed_snapshot_fields)
        if unknown: raise ValueError(f"snapshot {index} has forbidden/unrecognized fields: {unknown}")
    rows = []
    for example in rendered:
        snapshot = by_id.get(example["comparison_id"])
        if snapshot is None: raise ValueError(f"missing snapshot for {example['comparison_id']}")
        target_fields = {key: value for key, value in snapshot.items() if key != "comparison_id"}
        rows.append({"example_id": example["example_id"], "comparison_id": example["comparison_id"],
                     "prompt": example["prompt"], "completion": example["completion"], **target_fields})
    validate_q_sft_rows(rows, args.discount); write_jsonl(args.output, rows)
    print(json.dumps({"status": "passed", "rows": len(rows), "output": str(args.output), "data_audit": audit}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
