from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wordle_lab.common import canonical_json, read_json, sha256_file, sha256_text, write_json

from .qwen3_experiment import LOCAL_RUNS, MODEL_ID, MODEL_REVISION, PROTOCOL_ID, RESULTS, TRACK


GAMEPLAY_SUMMARIES = {
    "base_nonthinking": RESULTS / "base-nonthinking" / "base-nonthinking-summary.json",
    "base_thinking": RESULTS / "base-thinking" / "base-thinking-summary.json",
    "balanced_lora_150": RESULTS / "balanced-lora-s2026-n600" / "step-000150" / "balanced-lora-s2026-n600-step-000150-summary.json",
    "balanced_lora_300": RESULTS / "balanced-lora-s2026-n600" / "step-000300" / "balanced-lora-s2026-n600-step-000300-summary.json",
    "balanced_lora_450": RESULTS / "balanced-lora-s2026-n600" / "step-000450" / "balanced-lora-s2026-n600-step-000450-summary.json",
    "balanced_lora_600": RESULTS / "balanced-lora-s2026-n600" / "step-000600" / "balanced-lora-s2026-n600-step-000600-summary.json",
    "balanced_rslora_150": RESULTS / "balanced-rslora-s2026-n600" / "step-000150" / "balanced-rslora-s2026-n600-step-000150-summary.json",
    "balanced_rslora_300": RESULTS / "balanced-rslora-s2026-n600" / "step-000300" / "balanced-rslora-s2026-n600-step-000300-summary.json",
    "balanced_rslora_450": RESULTS / "balanced-rslora-s2026-n600" / "step-000450" / "balanced-rslora-s2026-n600-step-000450-summary.json",
    "balanced_rslora_600": RESULTS / "balanced-rslora-s2026-n600" / "step-000600" / "balanced-rslora-s2026-n600-step-000600-summary.json",
    "balanced_dora_150": RESULTS / "balanced-dora-s2026-n600" / "step-000150" / "balanced-dora-s2026-n600-step-000150-summary.json",
    "balanced_dora_300": RESULTS / "balanced-dora-s2026-n600" / "step-000300" / "balanced-dora-s2026-n600-step-000300-summary.json",
    "balanced_dora_450": RESULTS / "balanced-dora-s2026-n600" / "step-000450" / "balanced-dora-s2026-n600-step-000450-summary.json",
    "balanced_dora_600": RESULTS / "balanced-dora-s2026-n600" / "step-000600" / "balanced-dora-s2026-n600-step-000600-summary.json",
    "coverage4096_1024": RESULTS / "coverage4096-lora-s2026-n256" / "step-000064" / "coverage4096-lora-s2026-n256-step-000064-summary.json",
    "coverage4096_2048": RESULTS / "coverage4096-lora-s2026-n256" / "step-000128" / "coverage4096-lora-s2026-n256-step-000128-summary.json",
    "coverage4096_3072": RESULTS / "coverage4096-lora-s2026-n256" / "step-000192" / "coverage4096-lora-s2026-n256-step-000192-summary.json",
    "coverage4096_4096": RESULTS / "coverage4096-lora-s2026-n256" / "step-000256" / "coverage4096-lora-s2026-n256-step-000256-summary.json",
    "constraint_150": RESULTS / "constraint-lora-s2026-n600" / "step-000150" / "constraint-lora-s2026-n600-step-000150-summary.json",
    "constraint_300": RESULTS / "constraint-lora-s2026-n600" / "step-000300" / "constraint-lora-s2026-n600-step-000300-summary.json",
    "constraint_450": RESULTS / "constraint-lora-s2026-n600" / "step-000450" / "constraint-lora-s2026-n600-step-000450-summary.json",
    "constraint_600": RESULTS / "constraint-lora-s2026-n600" / "step-000600" / "constraint-lora-s2026-n600-step-000600-summary.json",
    "balanced_full_150": RESULTS / "balanced-full-s2026-n600" / "step-000150" / "balanced-full-s2026-n600-step-000150-summary.json",
    "balanced_full_300": RESULTS / "balanced-full-s2026-n600" / "step-000300" / "balanced-full-s2026-n600-step-000300-summary.json",
    "balanced_full_450": RESULTS / "balanced-full-s2026-n600" / "step-000450" / "balanced-full-s2026-n600-step-000450-summary.json",
    "balanced_full_600": RESULTS / "balanced-full-s2026-n600" / "step-000600" / "balanced-full-s2026-n600-step-000600-summary.json",
    "balanced_full_450_rep105": RESULTS / "balanced-full-s2026-n600" / "step-000450-rep1.05" / "balanced-full-s2026-n600-step-000450-rep1.05-summary.json",
}


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    gameplay, diagnostics = summary["gameplay"], summary["diagnostics"]
    return {
        "wins": gameplay["wins"],
        "win_rate": gameplay["win_rate"],
        "terminal_compliance": gameplay["terminal_marker_compliance"],
        "invalid_guess_rate": gameplay["invalid_guess_rate"],
        "repeat_guess_rate": gameplay["repeat_guess_rate"],
        "posterior_violation_rate": diagnostics["posterior_constraint_violation_rate"],
        "turn_2_violation_rate": diagnostics["by_turn"]["2"]["posterior_constraint_violation_rate"],
        "singleton_accuracy": diagnostics["singleton_answer_accuracy"],
        "action_target_accuracy": diagnostics["action_target_accuracy"],
        "retention": summary["retention"]["overall_score"],
    }


def parent_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "terminal_compliance_gte_0_99": metrics["terminal_compliance"] is not None and metrics["terminal_compliance"] >= 0.99,
        "turn_2_violation_lt_0_30": metrics["turn_2_violation_rate"] is not None and metrics["turn_2_violation_rate"] < 0.30,
        "singleton_accuracy_gte_0_80": metrics["singleton_accuracy"] is not None and metrics["singleton_accuracy"] >= 0.80,
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": {"terminal_compliance": 0.99, "turn_2_violation_rate_strictly_below": 0.30, "singleton_accuracy": 0.80}}


def collect() -> dict[str, Any]:
    missing = [str(path) for path in GAMEPLAY_SUMMARIES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"declared result files missing: {missing}")
    conditions = {name: compact(read_json(path)) for name, path in GAMEPLAY_SUMMARIES.items()}
    best_parent_name = "balanced_full_450"
    gate = parent_gate(conditions[best_parent_name])
    technique_status = {
        "q_sft": {"status": "blocked_prerequisite_legality_gate_failed", "training_started": False, "parent": best_parent_name, "gate": gate},
        "sft_to_grpo": {"status": "blocked_ineligible_warm_start", "training_started": False, "parent": best_parent_name, "gate": gate},
        "grpo_avspo": {"status": "blocked_ineligible_warm_start_and_no_baseline_grpo", "training_started": False, "parent": best_parent_name, "gate": gate},
        "dpo": {"status": "not_run_parent_strategy_unreliable", "training_started": False, "parent": best_parent_name, "gate": gate},
        "orpo": {"status": "not_run_parent_strategy_unreliable", "training_started": False, "parent": best_parent_name, "gate": gate},
        "locked_test": {"status": "closed", "accessed": False, "three_seed_replication": False},
    }
    tiny = {
        "general": read_json(RESULTS / "tiny-general-lora-s2026-n400" / "memorization-summary.json"),
        "singleton": read_json(RESULTS / "tiny-singleton-lora-s2026-n400" / "memorization-summary.json"),
    }
    structured = read_json(RESULTS / "structured-lora-s2026-n608" / "final" / "structured-summary.json")
    accounting = {}
    for run in sorted(LOCAL_RUNS.iterdir()):
        if (run / "accounting.json").is_file() and (run / "train-summary.json").is_file():
            accounting[run.name] = read_json(run / "accounting.json")
    matrix = {
        "experiment_id": "QWEN3-0.6B-WORDLE-001",
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "protocol_id": PROTOCOL_ID,
        "seed": 2026,
        "locked_test_access": False,
        "conditions": conditions,
        "tiny_memorization": tiny,
        "structured_microtasks": structured,
        "training_accounting": accounting,
        "post_training_techniques": technique_status,
    }
    write_json(TRACK / "experiment_matrix.json", matrix)
    files = {}
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            relative = path.relative_to(TRACK).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "experiment_id": matrix["experiment_id"],
        "model": matrix["model"],
        "protocol_id": PROTOCOL_ID,
        "locked_test_access": False,
        "declared_files": len(files),
        "files": files,
        "matrix_sha256": sha256_text(canonical_json(matrix)),
    }
    write_json(TRACK / "results_manifest.json", manifest)
    return {"conditions": len(conditions), "declared_files": len(files), "best_parent": best_parent_name, "gate": gate}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
