from __future__ import annotations

from next_steps.chatgpt_2026_08_23 import coverage_legality_extension as extension


def test_extension_is_disjoint_and_audited():
    audit = extension.audit_bundle()
    assert audit["status"] == "passed"
    assert audit["rows"] == audit["unique_states"] == 4096
    assert audit["target_cap"] <= 96
    assert audit["locked_test_access"] is False


def test_extension_spec_declares_conservative_restart():
    spec = extension.build_spec()
    assert spec["cumulative_unique_coverage"] == 8192
    assert spec["learning_rate"] == 1e-5
    assert spec["optimizer"] == "fresh_AdamW_declared"
    assert spec["parent"]["metrics"]["wins"] == 17
