from __future__ import annotations

import re
from collections.abc import Sequence

from .env import normalize_word

TERMINAL_PATTERN = r"^\s*Final answer:\s*([A-Za-z]{5})\s*$"
_TERMINAL_RE = re.compile(TERMINAL_PATTERN, flags=re.IGNORECASE)


def parse_terminal_answer(raw: str, allowed_words: Sequence[str]) -> dict:
    raw = raw or ""
    nonempty = [line for line in raw.splitlines() if line.strip()]
    if not nonempty:
        return {"parsed_guess": None, "format_valid": False, "status": "empty", "reasoning_text": "", "reasoning_tokens": 0}
    match = _TERMINAL_RE.fullmatch(nonempty[-1])
    if not match:
        status = "prose_after_terminal" if any(_TERMINAL_RE.fullmatch(line) for line in nonempty[:-1]) else "missing_terminal_line"
        return {"parsed_guess": None, "format_valid": False, "status": status, "reasoning_text": "\n".join(nonempty), "reasoning_tokens": 0}
    word = normalize_word(match.group(1))
    reasoning = "\n".join(nonempty[:-1]).strip()
    if word not in {normalize_word(item) for item in allowed_words}:
        return {"parsed_guess": word, "format_valid": True, "status": "invalid_word", "reasoning_text": reasoning, "reasoning_tokens": 0}
    return {"parsed_guess": word, "format_valid": True, "status": "ok", "reasoning_text": reasoning, "reasoning_tokens": 0}
