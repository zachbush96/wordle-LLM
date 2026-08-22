from __future__ import annotations

import argparse
import json
from pathlib import Path

from wordle_lab.common import set_seed, write_json
from wordle_lab.data.comparison import PARTITIONS
from wordle_lab.methods.sft import train_sft
from wordle_lab.standalone import base_spec, comparison_context, dry_run_summary, prepare_run


ADAPTERS = {
    "lora": {"use_rslora": False, "use_dora": False},
    "rslora": {"use_rslora": True, "use_dora": False},
    "dora": {"use_rslora": False, "use_dora": True},
}


def adapter_main(adapter_name: str) -> int:
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown adapter parameterization: {adapter_name}")
    parser = argparse.ArgumentParser(description=f"Standalone {adapter_name} SFT adapter ablation")
    parser.add_argument("--partition", choices=PARTITIONS, required=True); parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--steps", type=int, default=600); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); data_dir, rows, _, audit = comparison_context(args.data_dir, args.partition)
    spec = base_spec("sft", args.partition, args.seed, args.steps, args.learning_rate)
    spec["adapter_parameterization"] = adapter_name
    spec["lora"].update({"r": args.rank, "alpha": args.alpha, **ADAPTERS[adapter_name]})
    if args.dry_run:
        print(json.dumps(dry_run_summary(adapter_name, args.partition, data_dir, rows, spec, audit=audit), indent=2, sort_keys=True)); return 0
    run_dir = prepare_run(adapter_name, args.partition, spec, data_dir); set_seed(args.seed)
    _, accounting = train_sft(rows, run_dir, spec)
    summary = {"status": "trained", "adapter_parameterization": adapter_name, "run_dir": str(run_dir), "accounting": accounting}
    write_json(run_dir / "summary.json", summary); print(json.dumps(summary, indent=2, sort_keys=True)); return 0

