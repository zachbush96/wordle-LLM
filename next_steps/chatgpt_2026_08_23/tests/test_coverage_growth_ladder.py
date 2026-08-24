from __future__ import annotations

from next_steps.chatgpt_2026_08_23 import coverage_growth_ladder as growth


def _metrics(**updates):
    values = {
        "examples_seen": 7168,
        "wins": 17,
        "terminal_marker_compliance": 1.0,
        "invalid_guess_rate": 0.0,
        "repeat_guess_rate": 0.16,
        "posterior_constraint_violation_rate": 0.70,
        "turn_2_posterior_constraint_violation_rate": 0.60,
        "singleton_answer_accuracy": 10 / 74,
        "action_target_accuracy": 0.21,
        "retention": 0.0,
    }
    values.update(updates)
    return values


def test_growth_decision_requires_reliable_non_regressing_improvement():
    previous = _metrics()
    assert growth.growth_decision(previous, _metrics(examples_seen=10240, wins=18))["continue"] is True
    assert growth.growth_decision(previous, _metrics(examples_seen=10240, singleton_answer_accuracy=12 / 74))["continue"] is True
    assert growth.growth_decision(previous, _metrics(examples_seen=10240, terminal_marker_compliance=0.98, wins=18))["continue"] is False
    assert growth.growth_decision(previous, _metrics(examples_seen=10240, wins=16, singleton_answer_accuracy=15 / 74))["continue"] is False


def test_growth_bundle_is_disjoint_multi_turn_and_exact(tmp_path):
    directory, audit = growth.build_bundle(tmp_path / "bundle")
    assert directory == tmp_path / "bundle"
    assert audit["status"] == "passed"
    assert audit["rows"] == 13312
    assert audit["unique_states"] == 13312
    assert audit["multi_turn_only"] is True
    assert audit["target_cap"] <= 256
    assert set(audit["prefix_composition"]) == {"10240", "12288", "15360", "20480"}


def test_growth_spec_binds_parent_data_and_stop_policy():
    spec = growth.build_spec()
    assert spec["parent_coverage"] == 7168
    assert spec["milestones"] == [10240, 12288, 15360, 20480]
    assert spec["maximum_new_examples"] == 13312
    assert spec["parent"]["optimizer_state_available"] is False
    assert spec["data"]["multi_turn_only"] is True
    assert spec["locked_test_access"] is False


def test_forced_15k_spec_uses_only_unseen_ordered_slice():
    spec = growth.build_force_spec()
    assert spec["parent_coverage"] == 10240
    assert spec["target_coverage"] == 15360
    assert spec["milestones"] == [12288, 15360]
    assert spec["dataset_slice"] == {"start_inclusive": 3072, "end_exclusive": 8192, "rows": 5120}
    assert spec["steps"] == 1280
    assert spec["parent"]["optimizer_state_available"] is False
    assert spec["locked_test_access"] is False
