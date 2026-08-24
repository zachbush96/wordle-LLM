from __future__ import annotations

import copy

import pytest

from next_steps.chatgpt_2026_08_23 import coverage_max_experiment as coverage


def test_coverage_max_constants_define_one_nonrepeating_epoch():
    assert sum(coverage.QUOTAS.values()) == 4096
    assert coverage.MAX_STEPS * coverage.BATCH_SIZE == coverage.TRAIN_ROWS
    assert coverage.CHECKPOINT_STEPS == [256, 512, 768, 1024]
    assert coverage.EXAMPLES_SEEN == [1024, 2048, 3072, 4096]


def test_generated_coverage_max_bundle_is_audited():
    audit = coverage.audit_bundle()
    assert audit["status"] == "passed"
    assert audit["rows"] == 4096
    assert audit["unique_non_root_states"] == 4064
    assert audit["singleton_target_coverage"] == 96
    assert audit["target_cap_non_root"] <= 48
    assert audit["locked_test_access"] is False


def test_coverage_max_spec_is_fail_closed():
    spec = coverage.build_spec()
    assert spec["training_epochs"] == 1.0
    assert spec["shuffle_without_replacement"] is True
    assert spec["word_token_weight"] == 8.0
    changed = copy.deepcopy(spec)
    changed["max_steps"] += 1
    with pytest.raises(ValueError, match="coverage-max spec drift"):
        coverage.validate_spec(changed)


def test_coverage_max_training_rows_embed_diagnostic_source_states():
    rows = coverage.read_jsonl(coverage.DEFAULT_OUTPUT / "train.jsonl")
    sources = [row["source_state"] for row in rows]
    assert len(sources) == 4096
    assert all("history" in source and "facts" in source for source in sources)
