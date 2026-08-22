from pathlib import Path

import pytest
import yaml

from wordle_lab.experiments.hybrid_sft_grpo import (
    build_hybrid_plan,
    run_hybrid_pipeline,
    validate_hybrid_spec,
)


CONFIG = Path(__file__).parents[1] / "configs" / "studies" / "sft_grpo_hybrid.yaml"


def load_spec():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_hybrid_config_is_frozen_dev_only_and_warm_started():
    spec = validate_hybrid_spec(load_spec())
    plan = build_hybrid_plan(spec)
    assert spec["test_access"] == "forbidden"
    assert [step["stage"] for step in plan] == ["sft", "sft_dev_gate", "grpo"]
    assert plan[-1]["parent_from"] == "sft.final_checkpoint"


def test_pipeline_promotes_only_after_dev_gate():
    calls = []

    def train_sft(stage):
        calls.append("sft")
        return {"final_checkpoint": "runs/sft/checkpoints/final"}

    def evaluate_dev(*, checkpoint, split):
        calls.append(("evaluate", split, checkpoint))
        return {
            "split": split,
            "terminal_compliance": 1.0,
            "invalid_guess_rate": 0.0,
            "repeat_guess_rate": 0.0,
            "turn_2_posterior_violation_rate": 0.0,
            "singleton_answer_accuracy": 1.0,
        }

    def train_grpo(stage, *, parent_checkpoint, protocol_id):
        calls.append(("grpo", parent_checkpoint, protocol_id))
        return {"final_checkpoint": "runs/grpo/checkpoints/final"}

    result = run_hybrid_pipeline(load_spec(), train_sft_stage=train_sft, evaluate_dev=evaluate_dev, train_grpo_stage=train_grpo)
    assert result["status"] == "grpo_completed"
    assert calls[1][1] == "dev"
    assert calls[2][1] == "runs/sft/checkpoints/final"


def test_pipeline_stops_when_sft_dev_gate_fails():
    grpo_called = False

    def grpo(*args, **kwargs):
        nonlocal grpo_called
        grpo_called = True
        return {}

    result = run_hybrid_pipeline(
        load_spec(),
        train_sft_stage=lambda stage: {"final_checkpoint": "checkpoint"},
        evaluate_dev=lambda **kwargs: {"split": "dev", "terminal_compliance": 0.5},
        train_grpo_stage=grpo,
    )
    assert result["status"] == "stopped_at_sft_dev_gate"
    assert grpo_called is False


def test_hybrid_spec_rejects_test_access_and_behavior_hacks():
    spec = load_spec()
    spec["test_access"] = "allowed"
    with pytest.raises(ValueError, match="locked-test"):
        validate_hybrid_spec(spec)
    spec = load_spec()
    spec["candidate_forcing"] = True
    with pytest.raises(ValueError, match="forbidden behavior"):
        validate_hybrid_spec(spec)
