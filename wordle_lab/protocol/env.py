from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

ALL_GREEN = "GGGGG"


def normalize_word(word: str) -> str:
    return word.strip().upper()


def is_five_ascii_letters(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{5}", word.strip()))


def score_wordle(answer: str, guess: str) -> str:
    answer, guess = normalize_word(answer), normalize_word(guess)
    if not is_five_ascii_letters(answer) or not is_five_ascii_letters(guess):
        raise ValueError("answer and guess must be five ASCII letters")
    feedback = ["B"] * 5
    unmatched: Counter[str] = Counter()
    for i, (a, g) in enumerate(zip(answer, guess)):
        if a == g:
            feedback[i] = "G"
        else:
            unmatched[a] += 1
    for i, g in enumerate(guess):
        if feedback[i] != "G" and unmatched[g]:
            feedback[i] = "Y"
            unmatched[g] -= 1
    return "".join(feedback)


def posterior_candidates(history: Sequence[tuple[str, str]], answers: Sequence[str]) -> list[str]:
    remaining = [normalize_word(word) for word in answers]
    for guess, feedback in history:
        remaining = [word for word in remaining if score_wordle(word, guess) == feedback]
    return remaining


class WordleEnv:
    """Six valid guesses; invalid calls are observed but do not consume a turn."""

    def __init__(self, answer: str, allowed_words: Sequence[str], max_guesses: int = 6):
        self.allowed = {normalize_word(word) for word in allowed_words}
        self.max_guesses = max_guesses
        self.answer = normalize_word(answer)
        if self.answer not in self.allowed:
            raise ValueError("answer must be in allowed_words")
        self.history: list[tuple[str, str]] = []
        self.invalid_guesses = 0
        self.done = False
        self.won = False

    def step(self, guess: str | None) -> dict:
        if self.done:
            raise RuntimeError("game is finished")
        guess = normalize_word(guess or "")
        if guess not in self.allowed or not is_five_ascii_letters(guess):
            self.invalid_guesses += 1
            return {"valid": False, "guess": guess or None, "feedback": None, "repeat": False, "done": False, "won": False}
        repeat = any(old == guess for old, _ in self.history)
        feedback = score_wordle(self.answer, guess)
        self.history.append((guess, feedback))
        self.won = feedback == ALL_GREEN
        self.done = self.won or len(self.history) >= self.max_guesses
        return {"valid": True, "guess": guess, "feedback": feedback, "repeat": repeat, "done": self.done, "won": self.won}
