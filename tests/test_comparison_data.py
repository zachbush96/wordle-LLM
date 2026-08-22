from __future__ import annotations

import json
from pathlib import Path

import pytest

from wordle_lab.data.comparison import PARTITIONS, audit_comparison_bundle, default_directory
from wordle_lab.methods.q_sft import validate_q_sft_rows
from wordle_lab.standalone import assert_gemma_parent_adapter, comparison_context, preference_rows


DATA = default_directory()


def test_generated_comparison_bundle_is_matched_and_correct():
    audit = audit_comparison_bundle(DATA)
    assert audit["status"] == "passed"
    assert audit["source_rows"] == 4096
    assert audit["rendered_rows"] == 12288
    for partition in PARTITIONS:
        _, rows, sources, _ = comparison_context(DATA, partition)
        assert len(rows) == len(sources) == 4096


def test_reasoning_preferences_do_not_disclose_preference_label():
    _, rendered, sources, _ = comparison_context(DATA, "reasoning_single_step")
    pairs = preference_rows(rendered, sources, "reasoning_single_step")
    assert len(pairs) == 4096
    for pair in pairs[:100]:
        chosen = pair["chosen"][0]["content"].lower()
        rejected = pair["rejected"][0]["content"].lower()
        assert "lowest expected" not in chosen
        assert "worse than" not in rejected
        assert "proposed next action for comparison" in chosen
        assert "proposed next action for comparison" in rejected


def test_q_sft_rejects_secret_bearing_rows():
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "completion": [{"role": "assistant", "content": "Final answer: ABOUT"}],
        "bellman_target": 1.0,
        "secret_answer": "ABOUT",
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_q_sft_rows([row], 0.99)


def test_manifest_declares_gemma_only_and_exact_scale():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["model_id"] == "google/gemma-3-270m-it"
    assert manifest["model"]["exclusive"] is True
    assert manifest["rows_per_partition"] == 4096
    assert manifest["turn_distribution"] == {"1": 128, "2": 1024, "3": 1024, "4": 819, "5": 614, "6": 487}


def test_parent_adapter_guard_rejects_non_adapter(tmp_path: Path):
    with pytest.raises(RuntimeError, match="trained PEFT adapter"):
        assert_gemma_parent_adapter(tmp_path)
