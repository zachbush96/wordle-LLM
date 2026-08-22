from __future__ import annotations

import pytest

from wordle_lab.methods.unsloth_sft import UNSLOTH_BACKEND_ID, select_nested_rows


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
