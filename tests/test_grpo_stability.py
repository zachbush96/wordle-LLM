import pytest

from wordle_lab.methods.grpo_stability import (
    VIRTUAL_SAMPLE_USAGE,
    advantage_collapse_diagnostics,
    entropy_collapse_diagnostics,
    supported_advantage_estimate,
    update_adaptive_acr_threshold,
    validate_reward_rubric,
    virtual_sample_count,
)


def test_advantage_collapse_rate_is_group_level():
    result = advantage_collapse_diagnostics([[1.0, 1.0], [0.0, 1.0], [3.0, 3.0]])
    assert result["advantage_collapse_rate"] == pytest.approx(2 / 3)
    assert result["collapsed_groups"] == 2


def test_virtual_support_is_symmetric_auditable_and_not_an_outcome():
    result = supported_advantage_estimate(
        [2.0, 2.0],
        {"enabled": True, "alpha": 0.5, "zero_reward_anchor": 0.1},
        batch_acr=1.0,
        adaptive_threshold=0.5,
    )
    assert len(result["real_advantages"]) == 2
    assert len(result["real_advantages"]) == 2
    assert result["normalization_std"] > 0
    assert all(sample["synthetic"] for sample in result["virtual_samples"])
    assert all(sample["environment_outcome"] is False for sample in result["virtual_samples"])
    assert all(sample["usage"] == VIRTUAL_SAMPLE_USAGE for sample in result["virtual_samples"])
    assert [sample["reward"] for sample in result["virtual_samples"]] == pytest.approx([4 / 3, 2 / 3])


def test_avspo_count_and_adaptive_threshold_formulas():
    assert virtual_sample_count(8, 0.25, alpha=0.5) == 4
    assert update_adaptive_acr_threshold(0.5, 0.8, 0.1, eta=0.01) == pytest.approx(0.503)
    assert update_adaptive_acr_threshold(0.5, 0.8, -0.1, eta=0.01) == pytest.approx(0.497)


def test_virtual_support_requires_batch_trigger_and_collapsed_group():
    result = supported_advantage_estimate(
        [0.0, 0.0, 0.0, 0.0],
        {"enabled": True, "alpha": 0.5, "zero_reward_anchor": 0.1},
        batch_acr=0.5,
        adaptive_threshold=0.5,
    )
    assert result["virtual_sample_count"] == 0


def test_binary_reward_virtual_rule_is_not_invented_for_all_negative_shaped_group():
    result = supported_advantage_estimate(
        [-2.0, -2.0],
        {"enabled": True, "alpha": 0.5, "zero_reward_anchor": 0.1},
        batch_acr=1.0,
        adaptive_threshold=0.5,
    )
    assert result["virtual_sample_count"] == 0


def test_reward_rubric_matches_existing_shaped_reward_components():
    rubric = validate_reward_rubric(
        {
            "version": "wordle-shaped-v1",
            "weights": {"solve": 5, "information_gain": 1, "oracle_regret": -1, "repeat": -2, "format": -3},
        }
    )
    assert set(rubric["weights"]) == {"solve", "information_gain", "oracle_regret", "repeat", "format"}
    with pytest.raises(ValueError, match="exactly"):
        validate_reward_rubric({"weights": {"solve": 1}})


def test_entropy_guard_requires_sustained_collapse():
    spec = {"baseline_window": 3, "minimum_observations": 6, "absolute_floor": 0.2, "relative_floor": 0.5, "patience": 3}
    safe = entropy_collapse_diagnostics([2.0, 2.1, 1.9, 0.5, 1.0, 0.4], spec)
    stopped = entropy_collapse_diagnostics([2.0, 2.1, 1.9, 0.4, 0.3, 0.2], spec)
    assert safe["stop"] is False
    assert stopped["stop"] is True
    assert stopped["reason"] == "entropy_collapse"
