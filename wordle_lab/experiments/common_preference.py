from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import torch

from wordle_lab.common import ARTIFACTS, ROOT, canonical_json, read_jsonl, set_seed, sha256_file, write_json, write_jsonl
from wordle_lab.experiments.common_curriculum import BALANCED_CURRICULUM_ID, evaluate_saved_checkpoint
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.orpo import train_orpo
from wordle_lab.protocol.env import posterior_candidates


def build_repeat_preferences(common_dir: Path) -> list[dict]:
    rows = []
    for record in read_jsonl(common_dir / "canonical.jsonl"):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        chosen_word = record["facts"]["oracle_action"]
        rejected_word = history[-1][0] if history else record["facts"]["hard_negative"]
        if chosen_word == rejected_word:
            continue
        rows.append(
            {
                "pair_id": f"{record['state_id']}-repeat-preference",
                "state_id": record["state_id"],
                "negative_type": "prior_repeat" if history else "root_suboptimal",
                "prompt": _explicit_feedback_messages(history),
                "chosen": [{"role": "assistant", "content": f"Final answer: {chosen_word}"}],
                "rejected": [{"role": "assistant", "content": f"Final answer: {rejected_word}"}],
            }
        )
    return rows


def build_mixed_preferences(
    common_dir: Path,
    recovery_rows: Sequence[dict],
    total: int | None = None,
    seed: int = 2026,
) -> list[dict]:
    """Build 50/25/25 action-quality pairs with an identical output envelope."""
    canonical = read_jsonl(common_dir / "canonical.jsonl")
    universe = json.loads((common_dir / "universe.json").read_text(encoding="utf-8"))
    constraint_sources = [
        row for row in recovery_rows
        if row.get("error_type") == "constraint_violation" and row.get("model_guess")
    ]
    if not constraint_sources:
        raise ValueError("model-generated constraint-violation recovery rows are required")
    repeats = [row for row in canonical if row["history"]]
    strategic = [
        row for row in canonical
        if row["facts"]["hard_negative"] != row["facts"]["oracle_action"]
        and row["facts"]["hard_negative"] in posterior_candidates(
            [(item["guess"], item["feedback"]) for item in row["history"]], universe
        )
    ]
    if not repeats or not strategic:
        raise ValueError("canonical states do not support the requested negative mix")
    total = total or min(len(constraint_sources) * 2, max(4, len(canonical)))
    total -= total % 4
    if total < 4:
        raise ValueError("total must permit a 50/25/25 split")
    rng = random.Random(seed)
    rng.shuffle(constraint_sources)
    rng.shuffle(repeats)
    rng.shuffle(strategic)

    def pair(pair_id: str, state_id: str, negative_type: str, prompt: list[dict], chosen: str, rejected: str) -> dict:
        if chosen == rejected:
            raise ValueError(f"identical preference decisions for {pair_id}")
        return {
            "pair_id": pair_id,
            "state_id": state_id,
            "negative_type": negative_type,
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": f"Final answer: {chosen}"}],
            "rejected": [{"role": "assistant", "content": f"Final answer: {rejected}"}],
        }

    rows = []
    for index in range(total // 2):
        row = constraint_sources[index % len(constraint_sources)]
        rows.append(pair(
            f"{row['state_id']}-constraint-{index}", row["state_id"], "model_constraint_violation",
            row["prompt"], row["target_word"], row["model_guess"],
        ))
    for index in range(total // 4):
        row = repeats[index % len(repeats)]
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        rows.append(pair(
            f"{row['state_id']}-repeat-{index}", row["state_id"], "prior_repeat",
            _explicit_feedback_messages(history), row["facts"]["oracle_action"], history[-1][0],
        ))
    for index in range(total // 4):
        row = strategic[index % len(strategic)]
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        rows.append(pair(
            f"{row['state_id']}-strategic-{index}", row["state_id"], "strategically_inferior_consistent",
            _explicit_feedback_messages(history), row["facts"]["oracle_action"], row["facts"]["hard_negative"],
        ))
    rng.shuffle(rows)
    return rows


def run_orpo(
    parent_run_id: str,
    max_steps: int = 100,
    learning_rate: float = 5e-6,
    lambda_or: float = 0.1,
    seed: int = 2026,
    dev_games: int = 25,
    recovery_path: Path | None = None,
    preference_pairs: int = 512,
) -> tuple[str, dict]:
    parent_dir = ARTIFACTS / "runs" / parent_run_id
    parent_spec = json.loads((parent_dir / "spec.json").read_text(encoding="utf-8"))
    curriculum = parent_spec["curriculum"]
    curriculum_folder = "common-curriculum-002" if curriculum.get("curriculum_id") == BALANCED_CURRICULUM_ID else "common-curriculum-001"
    common_dir = ROOT / "data" / curriculum_folder / f"u{curriculum['universe_size']}-train{curriculum['train_secret_count']}"
    if recovery_path:
        recovery_rows = read_jsonl(recovery_path)
        rows = build_mixed_preferences(common_dir, recovery_rows, total=preference_pairs, seed=seed)
        representation = "common_inference_shaped_mixed_preferences"
    else:
        rows = build_repeat_preferences(common_dir)
        representation = "common_explicit_prior_repeat_preferences"
    spec = {
        "method": "orpo",
        "representation": representation,
        "parent_run_id": parent_run_id,
        "parent_checkpoint": str(parent_dir / "checkpoints" / "final"),
        "seed": seed,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "lambda_or": lambda_or,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 320,
        "curriculum": curriculum,
        "preference_pairs": len(rows),
        "negative_distribution": dict(sorted({
            kind: sum(row["negative_type"] == kind for row in rows) for kind in {row["negative_type"] for row in rows}
        }.items())),
        "recovery_source_sha256": sha256_file(recovery_path) if recovery_path else None,
    }
    run_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_id = f"orpo-common-repeat-s{seed}-{run_hash}"
    run_dir = ARTIFACTS / "runs" / run_id
    if (run_dir / "summary.json").exists():
        return run_id, json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_jsonl(run_dir / "preferences.jsonl", rows)
    set_seed(seed)
    model = train_orpo(rows, Path(spec["parent_checkpoint"]), run_dir, spec)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = evaluate_saved_checkpoint(run_id, "final", dev_games, "greedy_rep105")
    write_json(run_dir / "summary.json", summary)
    return run_id, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ORPO repeat-preference continuation for common curriculum")
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lambda-or", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dev-games", type=int, default=25)
    parser.add_argument("--recovery-jsonl", type=Path)
    parser.add_argument("--preference-pairs", type=int, default=512)
    args = parser.parse_args(argv)
    run_id, summary = run_orpo(
        args.parent_run_id,
        args.steps,
        args.learning_rate,
        args.lambda_or,
        args.seed,
        args.dev_games,
        args.recovery_jsonl,
        args.preference_pairs,
    )
    print(json.dumps({"run_id": run_id, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
