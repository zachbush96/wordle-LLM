from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from wordle_lab.common import canonical_json, sha256_file, sha256_text

_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])")


def rendered_text(rows: list[dict]) -> str:
    return "\n".join(canonical_json({"prompt": row.get("prompt"), "completion": row.get("completion"), "chosen": row.get("chosen"), "rejected": row.get("rejected")}) for row in rows)


def assert_no_test_leakage(rows: list[dict], test_answers: list[str]) -> None:
    tokens = {token.upper() for token in _WORD.findall(rendered_text(rows))}
    leaked = sorted(tokens & {word.upper() for word in test_answers})
    if leaked:
        raise RuntimeError(f"test-answer leakage in rendered training data: {leaked[:20]}")


def file_manifest(path: Path, rows: list[dict], tokenizer=None) -> dict:
    text = rendered_text(rows)
    token_counts = {}
    if tokenizer is not None:
        prompt_tokens = completion_tokens = 0
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            completion_obj = row.get("completion") or row.get("chosen")
            completion = completion_obj[0]["content"]
            prompt_tokens += len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            completion_tokens += len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        token_counts = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return {
        "path": str(path), "sha256": sha256_file(path), "records": len(rows),
        "unique_states": len({row["state_id"] for row in rows}),
        "turn_distribution": dict(sorted(Counter(str(row.get("turn", "preference")) for row in rows).items())),
        "rendered_content_sha256": sha256_text(text), **token_counts,
    }
