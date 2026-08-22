from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.data.comparison import audit_comparison_bundle, default_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute all correctness checks for comparison training data")
    parser.add_argument("--data-dir", type=Path, default=default_directory())
    parser.add_argument("--token-lengths", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_comparison_bundle(args.data_dir, include_token_lengths=args.token_lengths), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
