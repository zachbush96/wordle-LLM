from __future__ import annotations

from collections.abc import Sequence

PROMPT_VERSION = "reasoning-envelope-v1"
SYSTEM_PROMPT = (
    "You are a careful Wordle player. Use the complete feedback history. "
    "G means correct letter and position, Y means present in another position, and B means absent except for duplicate-letter accounting. "
    "Never repeat a previous guess. A guess must be a valid five-letter English word. "
    "You may reason visibly. End with exactly `Final answer: WORD` on the final non-empty line, with no prose after it."
)


def render_user_prompt(history: Sequence[tuple[str, str]]) -> str:
    rendered = "\n".join(f"{guess} -> {feedback}" for guess, feedback in history) or "(none)"
    return f"WORDLE\nPrevious guesses:\n{rendered}\n\nChoose the next guess."


def inference_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(history)},
    ]
