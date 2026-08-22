from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.data.comparison import build_comparison_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build matched Gemma 3 270M Wordle training partitions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--dev-states", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path, manifest = build_comparison_bundle(args.output_dir, states=args.states, dev_states=args.dev_states, seed=args.seed, force=args.force)
    print(json.dumps({"directory": str(path), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
