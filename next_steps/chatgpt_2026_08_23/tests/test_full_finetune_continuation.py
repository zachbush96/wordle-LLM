from __future__ import annotations

import copy

import pytest

from next_steps.chatgpt_2026_08_23 import full_finetune_continuation as continuation


def test_continuation_spec_is_exact_and_parent_bound():
    spec = continuation.continuation_spec()
    assert spec["experiment_id"] == continuation.EXPERIMENT_ID
    assert spec["parent_optimizer_steps"] == 600
    assert spec["continuation_optimizer_steps"] == 600
    assert spec["total_optimizer_steps"] == 1200
    assert spec["total_checkpoint_steps"] == [750, 900, 1050, 1200]
    assert spec["word_token_weight"] == 8.0
    assert spec["parent"]["checkpoint_tree_sha256"] == continuation.PARENT_TREE_SHA256
    assert spec["optimizer_restart_declared"] is True
    assert spec["locked_test_access"] is False


def test_continuation_spec_rejects_training_drift():
    spec = continuation.continuation_spec()
    changed = copy.deepcopy(spec)
    changed["word_token_weight"] = 1.0
    with pytest.raises(ValueError, match="continuation spec drift"):
        continuation.validate_spec(changed)


def test_continuation_parent_audit_has_expected_baseline_metrics():
    parent = continuation.audit_parent()
    assert parent["parent_metrics"]["wins"] == 14
    assert parent["parent_metrics"]["terminal_marker_compliance"] == 1.0
    assert parent["parent_metrics"]["singleton_answer_accuracy"] == pytest.approx(2 / 74)
    assert parent["optimizer_state_available"] is False


def test_continuation_dose_mapping_is_monotonic():
    assert continuation.RELATIVE_CHECKPOINT_STEPS == [150, 300, 450, 600]
    assert continuation.TOTAL_CHECKPOINT_STEPS == [750, 900, 1050, 1200]
    assert [continuation.PARENT_STEPS + step for step in continuation.RELATIVE_CHECKPOINT_STEPS] == continuation.TOTAL_CHECKPOINT_STEPS
