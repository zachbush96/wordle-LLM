import pytest

from wordle_lab.experiments.technique_catalog import (
    ProtocolCompatibilityError,
    TECHNIQUES,
    technique_manifest,
    validate_canonical_techniques,
)
from wordle_lab.methods.reward_rubrics import RewardSignals, multigranularity_reward, wordle_constraint_violations


def test_catalog_covers_every_notebooklm_technique_and_adapter():
    expected = {
        "lora", "sft", "dpo", "orpo", "grpo", "q_sft", "sft_grpo", "acr", "avspo",
        "dynamic_state_curriculum", "multigranularity_reward", "structured_letter_prompt", "hardcoded_opener",
    }
    assert expected <= set(TECHNIQUES)
    assert {row["name"] for row in technique_manifest()} == set(TECHNIQUES)


def test_canonical_catalog_refuses_harness_selected_opening_guess():
    with pytest.raises(ProtocolCompatibilityError, match="hardcoded_opener"):
        validate_canonical_techniques(["sft", "hardcoded_opener"])


def test_canonical_catalog_requires_new_protocol_for_prompt_reformatting():
    with pytest.raises(ProtocolCompatibilityError, match="structured_letter_prompt"):
        validate_canonical_techniques(["structured_letter_prompt"])


def test_canonical_training_stack_is_allowed():
    selected = validate_canonical_techniques(["lora", "q_sft", "acr", "avspo", "sft_grpo"])
    assert all(item.canonical_compatible for item in selected)


def test_multigranularity_reward_has_auditable_components():
    reward = multigranularity_reward(RewardSignals(
        format_valid=True,
        valid_word=True,
        solved=False,
        repeated=True,
        green_violations=1,
        missing_yellow_violations=2,
        gray_reuse_violations=1,
    ))
    assert reward["components"] == {
        "format": 0.05,
        "validity": 0.20,
        "completion": 0.0,
        "repetition": -0.30,
        "green_violation": -0.40,
        "missing_yellow": -0.50,
        "gray_reuse": -0.20,
    }
    assert reward["total"] == pytest.approx(-1.15)


def test_multigranularity_reward_rejects_negative_violation_counts():
    with pytest.raises(ValueError, match="non-negative"):
        multigranularity_reward(RewardSignals(True, True, False, green_violations=-1))


def test_constraint_ledger_handles_green_yellow_gray_and_duplicates():
    # The first P is present and the second P is excess/gray; A is fixed green.
    violations = wordle_constraint_violations([("PAPER", "YGBBB")], "APPLS")
    assert violations["green_violations"] == 1
    assert violations["missing_yellow_violations"] == 0
    assert violations["gray_reuse_violations"] == 1
