from __future__ import annotations

import pytest

from wordle_lab.common import DATA, ROOT, read_json, read_jsonl
from wordle_lab.data.comparison import audit_comparison_bundle
from wordle_lab.methods import unsloth_sft
from wordle_lab.methods.unsloth_sft import (
    UNSLOTH_BACKEND_ID,
    UNSLOTH_WEIGHTED_BACKEND_ID,
    select_nested_rows,
)
from wordle_lab.data.comparison import STRATIFIED_COMPLEXITY_COUNTS, stratified_complexity_words


def _rows(count: int) -> list[dict]:
    return [{"comparison_id": f"state-{index}", "target_word": "SHARE"} for index in range(count)]


def test_nested_state_doses_are_deterministic_and_nested():
    rows = _rows(20)
    small = select_nested_rows(rows, 5)
    repeated = select_nested_rows(list(reversed(rows)), 5)
    large = select_nested_rows(rows, 10)
    assert small == repeated
    assert {row["comparison_id"] for row in small} <= {row["comparison_id"] for row in large}


def test_state_dose_validation_and_full_selection():
    rows = _rows(4)
    assert select_nested_rows(rows, None) == rows
    assert select_nested_rows(rows, 10) == rows
    with pytest.raises(ValueError, match="positive"):
        select_nested_rows(rows, 0)


def test_backend_id_is_versioned():
    assert UNSLOTH_BACKEND_ID == "UNSLOTH-GEMMA-SFT-001"


@pytest.mark.parametrize("previous", [None, "legacy"])
def test_weighted_logit_environment_is_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path, previous):
    import os

    if previous is None:
        monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    else:
        monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", previous)

    observed = {}

    def fake_impl(rows, run_dir, spec):
        observed["during"] = os.environ.get("UNSLOTH_RETURN_LOGITS")
        return object(), object(), {}

    monkeypatch.setattr(unsloth_sft, "_train_unsloth_sft_impl", fake_impl)
    spec = {"backend": UNSLOTH_WEIGHTED_BACKEND_ID, "word_token_weight": 8.0}
    unsloth_sft.train_unsloth_sft([], tmp_path, spec)
    assert observed["during"] == "1"
    assert os.environ.get("UNSLOTH_RETURN_LOGITS") == previous


def test_stratified_words_are_unique_and_match_declared_bands():
    words, labels = stratified_complexity_words(160)
    assert len(words) == len(set(words)) == 160
    assert set(labels) == set(words)
    assert {label: list(labels.values()).count(label) for label in STRATIFIED_COMPLEXITY_COUNTS} == STRATIFIED_COMPLEXITY_COUNTS
    assert set(words) <= set(read_json(DATA / "splits" / "train_answers.json"))
    assert not set(words) & set(read_json(DATA / "splits" / "dev_answers.json"))


def test_stratified_profile_rejects_unmatched_universe_size():
    with pytest.raises(ValueError, match="exactly 160"):
        stratified_complexity_words(128)


def test_generated_unsloth_alpaca_bundle_is_complete_and_audited():
    directory = ROOT / "data" / "gemma-270m-unsloth-alpaca-v2" / "u160-train120-n2000"
    manifest = read_json(directory / "manifest.json")
    assert manifest["dataset_id"] == "GEMMA-270M-UNSLOTH-ALPACA-002"
    assert manifest["rows_per_partition"] == 2000
    assert manifest["rendered_training_rows"] == 6000
    assert manifest["complexity_bands"] == STRATIFIED_COMPLEXITY_COUNTS
    assert audit_comparison_bundle(directory)["status"] == "passed"
    for partition in ("non_reasoning_single_step", "non_reasoning_multi_step", "reasoning_single_step"):
        rows = read_jsonl(directory / f"{partition}.jsonl")
        assert len(rows) == 2000
        assert all(row["messages"] == row["prompt"] + row["completion"] for row in rows)
        assert all(row["output"] == row["completion"][0]["content"] for row in rows)
