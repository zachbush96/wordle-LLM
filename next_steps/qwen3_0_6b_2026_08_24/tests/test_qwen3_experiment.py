from __future__ import annotations

from next_steps.qwen3_0_6b_2026_08_24.collect_results import parent_gate
from next_steps.qwen3_0_6b_2026_08_24.qwen3_experiment import MODEL_ID, MODEL_REVISION, model_manifest


def test_pinned_qwen_identity_and_closed_test():
    manifest = model_manifest()
    assert manifest["model_id"] == MODEL_ID == "Qwen/Qwen3-0.6B"
    assert manifest["revision"] == MODEL_REVISION == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert manifest["model_type"] == "qwen3"
    assert manifest["architecture"] == ["Qwen3ForCausalLM"]
    assert manifest["locked_test_access"] is False


def test_post_training_gate_requires_all_three_metrics():
    passing = {"terminal_compliance": 0.99, "turn_2_violation_rate": 0.299, "singleton_accuracy": 0.80}
    assert parent_gate(passing)["passed"] is True
    for key, bad in (
        ("terminal_compliance", 0.989),
        ("turn_2_violation_rate", 0.30),
        ("singleton_accuracy", 0.79),
    ):
        metrics = dict(passing)
        metrics[key] = bad
        assert parent_gate(metrics)["passed"] is False


def test_undefined_metrics_fail_closed():
    result = parent_gate({"terminal_compliance": 0.0, "turn_2_violation_rate": None, "singleton_accuracy": 0.0})
    assert result["passed"] is False
    assert result["checks"]["turn_2_violation_lt_0_30"] is False
