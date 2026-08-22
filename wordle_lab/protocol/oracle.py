from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .env import normalize_word, score_wordle

_MAP = {"B": 0, "Y": 1, "G": 2}


def encode_feedback(feedback: str) -> int:
    value = 0
    for char in feedback:
        value = value * 3 + _MAP[char]
    return value


class GreedyPartitionOracle:
    def __init__(self, answers: Sequence[str]):
        self.answers = sorted({normalize_word(word) for word in answers})
        self.index = {word: i for i, word in enumerate(self.answers)}
        n = len(self.answers)
        self.matrix = np.empty((n, n), dtype=np.uint8)
        for guess_i, guess in enumerate(self.answers):
            self.matrix[guess_i] = [encode_feedback(score_wordle(answer, guess)) for answer in self.answers]
        self.cache: dict[tuple[int, ...], list[dict]] = {}

    def remaining(self, history: Sequence[tuple[str, str]]) -> np.ndarray:
        indices = np.arange(len(self.answers), dtype=np.int32)
        for guess, feedback in history:
            if normalize_word(guess) in self.index:
                indices = indices[self.matrix[self.index[normalize_word(guess)], indices] == encode_feedback(feedback)]
            else:
                indices = np.array([i for i in indices if score_wordle(self.answers[i], guess) == feedback], dtype=np.int32)
        return indices

    def ranked(self, remaining: np.ndarray, limit: int | None = None) -> list[dict]:
        key = tuple(map(int, remaining))
        if key not in self.cache:
            rows = []
            denominator = float(len(remaining))
            for guess_i in remaining:
                counts = np.bincount(self.matrix[int(guess_i), remaining], minlength=243)
                counts = counts[counts > 0]
                expected = float(np.square(counts).sum() / denominator)
                probabilities = counts / denominator
                entropy = float(-(probabilities * np.log2(probabilities)).sum())
                rows.append({"guess": self.answers[int(guess_i)], "expected_remaining": expected, "entropy": entropy})
            rows.sort(key=lambda row: (row["expected_remaining"], -row["entropy"], row["guess"]))
            for rank, row in enumerate(rows, 1):
                row["rank"] = rank
                row["regret"] = row["expected_remaining"] - rows[0]["expected_remaining"]
            self.cache[key] = rows
        return self.cache[key][:limit] if limit else self.cache[key]

    def best(self, history: Sequence[tuple[str, str]]) -> dict:
        return self.ranked(self.remaining(history), limit=1)[0]
