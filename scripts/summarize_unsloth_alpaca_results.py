from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from wordle_lab.common import ARTIFACTS, ROOT, read_json, sha256_file, source_tree_sha256, write_json
from wordle_lab.data.comparison import PINNED_SOLUTION_URL


DATA_DIR = ROOT / "data" / "gemma-270m-unsloth-alpaca-v2" / "u160-train120-n2000"
RUNS = {
    "single_step": ARTIFACTS / "runs" / "unsloth-sft-non_reasoning_single_step-s2026-b60cc18cd3",
    "multi_step": ARTIFACTS / "runs" / "unsloth-sft-non_reasoning_multi_step-s2026-46e287240c",
    "reasoning": ARTIFACTS / "runs" / "unsloth-sft-reasoning_single_step-s2026-c2e8d1797d",
}
BASE_SUMMARY = ARTIFACTS / "runs" / "base-gemma-270m-u160-train120-n2000-dev-72e3384000" / "summary.json"
PRIOR_RUN = ARTIFACTS / "runs" / "unsloth-sft-non_reasoning_single_step-s2026-b04e5f76a5"
PRIOR_SUMMARY = PRIOR_RUN / "eval-step-000300-on-u160-v2-summary.json"
OUTPUT = ROOT / "docs" / "research" / "unsloth_gemma_wordle_alpaca_v2_results.json"


def compact(summary: dict) -> dict:
    gameplay = summary["gameplay"]
    diagnostics = summary["diagnostics"]
    return {
        "wins": gameplay["wins"],
        "games": gameplay["n_games"],
        "terminal_compliance": gameplay["terminal_marker_compliance"],
        "invalid_guess_rate": gameplay["invalid_guess_rate"],
        "gameplay_repeat_rate": gameplay["repeat_guess_rate"],
        "gameplay_constraint_violation_rate": gameplay["constraint_violation_rate"],
        "mean_generated_tokens": gameplay["mean_generated_tokens"],
        "mean_latency_s": gameplay["mean_latency_s"],
        "fixed_state_terminal_compliance": diagnostics["terminal_compliance"],
        "fixed_state_constraint_violation_rate": diagnostics["posterior_constraint_violation_rate"],
        "singleton_accuracy": diagnostics["singleton_answer_accuracy"],
        "action_target_accuracy": diagnostics["action_target_accuracy"],
        "retention": summary["retention"]["overall_score"],
    }


def artifact_hashes(run_dir: Path) -> dict:
    return {
        "adapter_step_300_sha256": sha256_file(run_dir / "checkpoints" / "step-000300" / "adapter_model.safetensors"),
        "train_metrics_sha256": sha256_file(run_dir / "train_metrics.jsonl"),
        "final_eval_summary_sha256": sha256_file(run_dir / "eval-step-000300-summary.json"),
    }


def main() -> int:
    manifest = read_json(DATA_DIR / "manifest.json")
    variants = {}
    for name, run_dir in RUNS.items():
        accounting = read_json(run_dir / "accounting.json")
        variants[name] = {
            "run_directory": str(run_dir.relative_to(ROOT)),
            "training": accounting,
            "checkpoints": {
                str(step): compact(read_json(run_dir / f"eval-step-{step:06d}-summary.json"))
                for step in (75, 150, 225, 300)
            },
            "hashes": artifact_hashes(run_dir),
        }
    result = {
        "experiment_id": "UNSLOTH-GEMMA-ALPACA-002",
        "protocol": "WORDLE-PROTOCOL-002",
        "locked_test_access": False,
        "official_format_sources": [
            "https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide",
            "https://docs.unsloth.ai/basics/chat-templates",
            "https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3",
        ],
        "dataset": {
            "directory": str(DATA_DIR.relative_to(ROOT)),
            "lexical_source_url": PINNED_SOLUTION_URL,
            "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
            "manifest": manifest,
        },
        "matched_base": compact(read_json(BASE_SUMMARY)),
        "prior_unsloth_step_300_on_same_holdout": compact(read_json(PRIOR_SUMMARY)),
        "variants": variants,
        "source_tree_sha256": source_tree_sha256(),
        "decision": "no_strategic_improvement_do_not_promote_locked_test_closed",
    }
    write_json(OUTPUT, result)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
