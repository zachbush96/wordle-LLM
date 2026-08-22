from pathlib import Path

from wordle_lab.common import ROOT
from wordle_lab.experiments.common_preference import build_repeat_preferences


def test_repeat_preferences_reject_the_last_guess():
    directory = ROOT / "data" / "common-curriculum-001" / "u128-train96"
    if not (directory / "canonical.jsonl").exists():
        return
    rows = build_repeat_preferences(Path(directory))
    repeat_rows = [row for row in rows if row["negative_type"] == "prior_repeat"]
    assert repeat_rows
    assert all(row["chosen"] != row["rejected"] for row in repeat_rows)
