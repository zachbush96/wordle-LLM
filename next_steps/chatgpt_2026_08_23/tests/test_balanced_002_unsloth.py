from __future__ import annotations

import copy
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.balanced_002_unsloth import (
    CURRICULUM_ID,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    audit_all,
    build_spec,
    dry_run,
    evaluation_plan,
    load_config,
    prepare_run_directory,
    promotion_gate_status,
    run_id_for_spec,
    validate_config,
)
from wordle_lab.methods.unsloth_sft import UNSLOTH_WEIGHTED_BACKEND_ID


def test_config_is_the_exact_historical_matched_recipe():
    config = load_config(DEFAULT_CONFIG)
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["backend"] == UNSLOTH_WEIGHTED_BACKEND_ID
    assert config["data"]["curriculum_id"] == CURRICULUM_ID
    assert config["data"]["training_rows"] == 512
    assert config["seed"] == 2026
    assert config["max_steps"] == 600
    assert config["learning_rate"] == 5e-5
    assert config["batch_size"] * config["gradient_accumulation_steps"] == 4
    assert config["max_length"] == 320
    assert config["word_token_weight"] == 8.0
    assert config["lora"]["dropout"] == 0.05
    assert config["checkpoint_steps"] == [150, 300, 450, 600]
    assert config["locked_test_access"] is False


def test_recipe_drift_fails_closed():
    config = load_config(DEFAULT_CONFIG)
    drifted = copy.deepcopy(config)
    drifted["batch_size"] = 16
    with pytest.raises(ValueError, match="recipe drifted"):
        validate_config(drifted)


def test_balanced_bundle_and_protocol_audit_pass_without_locked_test_access():
    result = audit_all(load_config(DEFAULT_CONFIG))
    assert result["status"] == "passed"
    assert result["locked_test_access"] is False
    assert result["protocol"]["locked_test_file_read"] is False
    assert result["dataset"]["locked_test_file_read"] is False
    assert result["dataset"]["training_rows"] == 512
    assert result["dataset"]["unique_source_states"] == 378
    assert result["dataset"]["state_copy_max"] <= 4
    assert result["dataset"]["target_word_max"] <= 8
    assert result["dataset"]["train_dev_overlap"] == 0
    assert result["dataset"]["split_identity_with_comparison_bundle"] is True


def test_dry_run_is_deterministic_and_does_not_start_training():
    config = load_config(DEFAULT_CONFIG)
    first = dry_run(config)
    second = dry_run(config)
    assert first["run_id"] == second["run_id"]
    assert first["training_started"] is False
    assert first["locked_test_access"] is False
    assert first["spec"]["word_token_weight"] == 8.0


def test_prepare_run_directory_writes_audited_spec_and_manifest(tmp_path: Path):
    config = load_config(DEFAULT_CONFIG)
    destination, spec, audit = prepare_run_directory(config, tmp_path / "run")
    assert destination == (tmp_path / "run").resolve()
    assert (destination / "spec.json").is_file()
    assert (destination / "dataset_manifest.json").is_file()
    assert (destination / "preflight_audit.json").is_file()
    assert spec["backend"] == UNSLOTH_WEIGHTED_BACKEND_ID
    assert spec["data"]["hashes"]["train.jsonl"] == config["data"]["hashes"]["train.jsonl"]
    assert audit["locked_test_access"] is False
    assert run_id_for_spec(spec).startswith("unsloth-balanced-002-word8-s2026-")


def test_prepare_refuses_to_overwrite_a_different_spec(tmp_path: Path):
    config = load_config(DEFAULT_CONFIG)
    destination, _, _ = prepare_run_directory(config, tmp_path / "run")
    (destination / "spec.json").write_text('{"different": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prepare_run_directory(config, destination)


def test_evaluation_plan_covers_every_dose_and_only_final_sensitivity():
    config = load_config(DEFAULT_CONFIG)
    plan = evaluation_plan(config)
    assert [item["checkpoint"] for item in plan[:4]] == [
        "step-000150",
        "step-000300",
        "step-000450",
        "step-000600",
    ]
    assert all(item["decoder"] == "greedy" and item["include_retention"] for item in plan[:4])
    assert plan[4] == {"checkpoint": "final", "decoder": "greedy_rep105", "include_retention": False}


def test_promotion_gate_requires_all_three_metrics():
    config = load_config(DEFAULT_CONFIG)
    summary = {
        "gameplay": {"terminal_marker_compliance": 0.99},
        "diagnostics": {
            "singleton_answer_accuracy": 0.80,
            "by_turn": {"2": {"posterior_constraint_violation_rate": 0.29}},
        },
    }
    assert promotion_gate_status(summary, config["promotion_gates"])["passed"] is True
    summary["diagnostics"]["by_turn"]["2"]["posterior_constraint_violation_rate"] = 0.30
    assert promotion_gate_status(summary, config["promotion_gates"])["passed"] is False


def test_build_spec_keeps_all_non_cheating_controls_disabled():
    config = load_config(DEFAULT_CONFIG)
    spec = build_spec(config, audit_all(config))
    for key in (
        "locked_test_access",
        "candidate_injection",
        "vocabulary_masking",
        "reranking",
        "repeat_ban",
        "output_repair",
        "harness_selected_guess",
    ):
        assert spec[key] is False
