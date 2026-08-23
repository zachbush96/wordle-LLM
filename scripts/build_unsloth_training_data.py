from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import ROOT
from wordle_lab.data.comparison import build_comparison_bundle


DEFAULT_OUTPUT = ROOT / "data" / "gemma-270m-unsloth-alpaca-v2" / "u160-train120-n2000"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build 2,000 audited Unsloth/Gemma Wordle examples for each of three representations"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-variety", type=int, default=2000)
    parser.add_argument("--dev-states", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    directory, manifest = build_comparison_bundle(
        args.output_dir,
        universe_size=160,
        train_secret_count=120,
        states=args.examples_per_variety,
        dev_states=args.dev_states,
        seed=args.seed,
        word_profile="stratified",
        force=args.force,
    )
    print(json.dumps({"directory": str(directory), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
