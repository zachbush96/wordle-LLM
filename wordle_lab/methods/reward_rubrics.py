from __future__ import annotations

from dataclasses import dataclass
from collections import Counter


NOTEBOOKLM_REWARD_VERSION = "notebooklm-multigranularity-v1"


@dataclass(frozen=True)
class RewardSignals:
    format_valid: bool
    valid_word: bool
    solved: bool
    repeated: bool = False
    green_violations: int = 0
    missing_yellow_violations: int = 0
    gray_reuse_violations: int = 0

    def validate(self) -> None:
        counts = (self.green_violations, self.missing_yellow_violations, self.gray_reuse_violations)
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("constraint violation counts must be non-negative integers")


DEFAULT_NOTEBOOKLM_WEIGHTS = {
    "format": 0.05,
    "validity": 0.20,
    "completion": 1.00,
    "repetition": -0.30,
    "green_violation": -0.40,
    "missing_yellow": -0.25,
    "gray_reuse": -0.20,
}


def multigranularity_reward(signals: RewardSignals, weights: dict | None = None) -> dict:
    """Compute the NotebookLM-proposed rubric without changing evaluation behavior.

    Constraint penalties are per violation. This function is training-only and
    returns a component ledger so scale and reward hacking can be audited.
    """
    signals.validate()
    applied = {**DEFAULT_NOTEBOOKLM_WEIGHTS, **(weights or {})}
    missing = sorted(set(DEFAULT_NOTEBOOKLM_WEIGHTS) - set(applied))
    if missing:
        raise ValueError(f"missing reward weights: {missing}")
    components = {
        "format": applied["format"] * float(signals.format_valid),
        "validity": applied["validity"] * float(signals.valid_word),
        "completion": applied["completion"] * float(signals.solved),
        "repetition": applied["repetition"] * float(signals.repeated),
        "green_violation": applied["green_violation"] * signals.green_violations,
        "missing_yellow": applied["missing_yellow"] * signals.missing_yellow_violations,
        "gray_reuse": applied["gray_reuse"] * signals.gray_reuse_violations,
    }
    return {
        "total": sum(components.values()),
        "components": components,
        "reward_version": NOTEBOOKLM_REWARD_VERSION,
        "weights": applied,
    }


def wordle_constraint_violations(history: list[tuple[str, str]], guess: str) -> dict[str, int]:
    """Count feedback violations with duplicate-letter accounting.

    This is a training-reward diagnostic. It never filters, repairs, or selects
    a model guess and therefore does not alter canonical generation.
    """
    guess = guess.upper()
    green_positions: dict[int, str] = {}
    minimum_counts: Counter[str] = Counter()
    maximum_counts: dict[str, int] = {}
    for old_guess, feedback in history:
        old_guess = old_guess.upper()
        row_positive = Counter(letter for letter, code in zip(old_guess, feedback) if code in "GY")
        for index, (letter, code) in enumerate(zip(old_guess, feedback)):
            if code == "G":
                green_positions[index] = letter
        for letter, count in row_positive.items():
            minimum_counts[letter] = max(minimum_counts[letter], count)
        for letter in set(old_guess):
            if any(code == "B" for tile, code in zip(old_guess, feedback) if tile == letter):
                bound = row_positive[letter]
                maximum_counts[letter] = min(maximum_counts.get(letter, 5), bound)

    counts = Counter(guess)
    green = sum(guess[index] != letter for index, letter in green_positions.items())
    missing_yellow = sum(max(0, required - counts[letter]) for letter, required in minimum_counts.items())
    gray_reuse = sum(max(0, counts[letter] - maximum) for letter, maximum in maximum_counts.items())
    return {
        "green_violations": green,
        "missing_yellow_violations": missing_yellow,
        "gray_reuse_violations": gray_reuse,
    }
