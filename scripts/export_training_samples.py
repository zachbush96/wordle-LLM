from __future__ import annotations

"""Export a few deterministic examples per training representation.

Full generated corpora remain local and ignored. Run the corpus builders first,
then use this script to refresh the small, reviewable Git examples.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "training_data"

PLAIN_SOURCES = {
    "comparison_direct_single_step": "data/gemma-270m-comparison-v1/u128-train96-n4096/non_reasoning_single_step.jsonl",
    "comparison_multi_turn": "data/gemma-270m-comparison-v1/u128-train96-n4096/non_reasoning_multi_step.jsonl",
    "comparison_visible_reasoning": "data/gemma-270m-comparison-v1/u128-train96-n4096/reasoning_single_step.jsonl",
    "unsloth_direct_single_step": "data/gemma-270m-unsloth-alpaca-v2/u160-train120-n2000/non_reasoning_single_step.jsonl",
    "unsloth_multi_turn": "data/gemma-270m-unsloth-alpaca-v2/u160-train120-n2000/non_reasoning_multi_step.jsonl",
    "unsloth_visible_reasoning": "data/gemma-270m-unsloth-alpaca-v2/u160-train120-n2000/reasoning_single_step.jsonl",
    "structured_feedback_decode": "next_steps/chatgpt_2026_08_23/generated/structured_microtasks_v1/train/feedback_decode.jsonl",
    "structured_constraint_merge": "next_steps/chatgpt_2026_08_23/generated/structured_microtasks_v1/train/constraint_merge.jsonl",
    "structured_candidate_validity": "next_steps/chatgpt_2026_08_23/generated/structured_microtasks_v1/train/candidate_validity.jsonl",
    "structured_singleton_solve": "next_steps/chatgpt_2026_08_23/generated/structured_microtasks_v1/train/singleton_solve.jsonl",
    "structured_full_policy": "next_steps/chatgpt_2026_08_23/generated/structured_microtasks_v1/train/full_policy.jsonl",
    "constraint_first_full_policy": "next_steps/chatgpt_2026_08_23/generated/constraint_first/train.jsonl",
    "q_sft_frozen_target": "next_steps/chatgpt_2026_08_23/generated/q_sft_frozen/training_rows.jsonl",
    "tiny_overfit_general": "next_steps/chatgpt_2026_08_23/generated/tiny_overfit/general_32.jsonl",
    "tiny_overfit_singleton": "next_steps/chatgpt_2026_08_23/generated/tiny_overfit/singleton_32.jsonl",
}

GROUPED_SOURCES = {
    "balanced_002": (
        "data/common-curriculum-002/u128-train96/train.jsonl",
        "state_type",
        ("root", "turn_2", "later_on_policy", "recovery_singleton"),
    ),
    "coverage_growth": (
        "data/common-curriculum-008/u128-train96-growth-07168-to-20480/train.jsonl",
        "state_type",
        ("turn_2", "low_posterior", "true_singleton", "later_broad"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def export_samples(output: Path = DEFAULT_OUTPUT, count: int = 3) -> dict[str, Any]:
    if count < 1 or count > 10:
        raise ValueError("count must be between 1 and 10")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for name, relative in PLAIN_SOURCES.items():
        source = ROOT / relative
        rows = _read_rows(source)
        if len(rows) < count:
            raise RuntimeError(f"{relative} has only {len(rows)} rows")
        destination = output / f"{name}.jsonl"
        _write_rows(destination, rows[:count])
        entries.append(
            {
                "training_type": name,
                "source": relative,
                "source_rows": len(rows),
                "source_sha256": _sha256(source),
                "sample": destination.relative_to(ROOT).as_posix(),
                "sample_rows": count,
                "sample_sha256": _sha256(destination),
            }
        )

    for prefix, (relative, field, values) in GROUPED_SOURCES.items():
        source = ROOT / relative
        rows = _read_rows(source)
        for value in values:
            matches = [row for row in rows if row.get(field) == value]
            if len(matches) < count:
                raise RuntimeError(f"{relative} has only {len(matches)} rows for {field}={value}")
            name = f"{prefix}_{value}"
            destination = output / f"{name}.jsonl"
            _write_rows(destination, matches[:count])
            entries.append(
                {
                    "training_type": name,
                    "source": relative,
                    "source_filter": {field: value},
                    "source_rows": len(rows),
                    "matching_source_rows": len(matches),
                    "source_sha256": _sha256(source),
                    "sample": destination.relative_to(ROOT).as_posix(),
                    "sample_rows": count,
                    "sample_sha256": _sha256(destination),
                }
            )

    manifest = {
        "schema_version": "wordle-training-samples-v1",
        "policy": "examples_only_full_corpora_local_and_gitignored",
        "selection": "first N rows in deterministic generated order; grouped sources select first N matching rows",
        "sample_rows_per_type": count,
        "training_types": len(entries),
        "entries": sorted(entries, key=lambda entry: entry["training_type"]),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export small Git-safe samples from local generated training corpora")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    result = export_samples(args.output, args.count)
    print(json.dumps({"output": str(args.output), "training_types": result["training_types"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
