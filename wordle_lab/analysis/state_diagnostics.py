from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from wordle_lab.common import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl
from wordle_lab.protocol.env import normalize_word, posterior_candidates
from wordle_lab.protocol.parsing import parse_terminal_answer


DIAGNOSTIC_ID = "COMMON-STATE-DIAGNOSTICS-001"


def history_from_record(record: dict) -> list[tuple[str, str]]:
    return [(item["guess"], item["feedback"]) for item in record["history"]]


def history_key(history: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(canonical_json(list(history)).encode("utf-8")).hexdigest()


def build_probe_items(records: Iterable[dict], training_records: Iterable[dict] = ()) -> list[dict]:
    """Build development probes without exposing a secret to the model prompt."""
    train_keys = {history_key(history_from_record(row)) for row in training_records}
    items = []
    for record in records:
        history = history_from_record(record)
        facts = record["facts"]
        items.append(
            {
                "item_id": record["state_id"],
                "history": [{"guess": guess, "feedback": feedback} for guess, feedback in history],
                "turn": len(history) + 1,
                "posterior_size": int(facts["posterior_count"]),
                "oracle_action": normalize_word(facts["oracle_action"]),
                "train_state_seen": history_key(history) in train_keys,
                # Kept only for offline scoring/auditing; renderers receive history alone.
                "secret_answer": normalize_word(record["secret_answer"]),
            }
        )
    return items


def _rate(rows: Sequence[dict], field: str, denominator_field: str | None = None) -> float | None:
    eligible = [row for row in rows if denominator_field is None or row[denominator_field]]
    return sum(bool(row[field]) for row in eligible) / len(eligible) if eligible else None


def _metrics(rows: Sequence[dict]) -> dict:
    return {
        "items": len(rows),
        "terminal_compliance": _rate(rows, "format_valid"),
        "valid_word_accuracy": _rate(rows, "valid_word"),
        "posterior_consistency": _rate(rows, "posterior_consistent", "valid_word"),
        "posterior_constraint_violation_rate": (
            None
            if not [row for row in rows if row["valid_word"]]
            else 1.0 - float(_rate(rows, "posterior_consistent", "valid_word"))
        ),
        "repeat_rate": _rate(rows, "repeat", "valid_word"),
        "action_target_accuracy": _rate(rows, "oracle_match", "valid_word"),
        "singleton_answer_accuracy": _rate(rows, "singleton_correct", "singleton"),
        "train_state_coverage": _rate(rows, "train_state_seen"),
    }


def score_probe_outputs(
    items: Sequence[dict], outputs: Sequence[str | dict], allowed_words: Sequence[str], answer_vocabulary: Sequence[str]
) -> tuple[list[dict], dict]:
    if len(items) != len(outputs):
        raise ValueError("one model output is required for every diagnostic item")
    allowed = {normalize_word(word) for word in allowed_words}
    rows = []
    for item, output in zip(items, outputs):
        raw = output.get("raw_output", "") if isinstance(output, dict) else output
        history = history_from_record(item)
        posterior = posterior_candidates(history, answer_vocabulary)
        parsed = parse_terminal_answer(raw, allowed)
        guess = parsed["parsed_guess"]
        valid_word = parsed["status"] == "ok" and guess in allowed
        repeated = bool(valid_word and any(old == guess for old, _ in history))
        consistent = bool(valid_word and guess in posterior)
        singleton = len(posterior) == 1
        rows.append(
            {
                **item,
                "raw_output": raw,
                "parsed_guess": guess,
                "parse_status": parsed["status"],
                "format_valid": parsed["format_valid"],
                "valid_word": valid_word,
                "repeat": repeated,
                "posterior_consistent": consistent,
                "constraint_violation": bool(valid_word and not consistent),
                "oracle_match": bool(valid_word and guess == item["oracle_action"]),
                "singleton": singleton,
                "singleton_correct": bool(singleton and valid_word and guess == posterior[0]),
            }
        )
    by_turn = {str(key): _metrics(value) for key, value in sorted(_group(rows, "turn").items())}
    by_posterior = {
        str(key): _metrics(value) for key, value in sorted(_group(rows, "posterior_size").items())
    }
    summary = {
        "diagnostic_id": DIAGNOSTIC_ID,
        **_metrics(rows),
        "by_turn": by_turn,
        "by_posterior_size": by_posterior,
        "error_counts": dict(
            sorted(
                Counter(
                    "format" if not row["format_valid"] else
                    "invalid_word" if not row["valid_word"] else
                    "repeat" if row["repeat"] else
                    "constraint_violation" if row["constraint_violation"] else
                    "singleton_miss" if row["singleton"] and not row["singleton_correct"] else
                    "none"
                    for row in rows
                ).items()
            )
        ),
    }
    return rows, summary


def _group(rows: Sequence[dict], field: str) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row[field])].append(row)
    return groups


def rollout_state_coverage(games: Iterable[dict], training_records: Iterable[dict]) -> dict:
    train_keys = {history_key(history_from_record(row)) for row in training_records}
    counts = Counter()
    for game in games:
        history: list[tuple[str, str]] = []
        for turn in game["turns"]:
            counts["states"] += 1
            counts["seen"] += history_key(history) in train_keys
            if turn.get("valid"):
                history.append((turn["guess"], turn["feedback"]))
    return {
        "rollout_states": counts["states"],
        "train_states_seen": counts["seen"],
        "train_state_coverage": counts["seen"] / counts["states"] if counts["states"] else None,
    }


def run_state_diagnostics(
    model,
    tokenizer,
    probe_records: Sequence[dict],
    training_records: Sequence[dict],
    allowed_words: Sequence[str],
    answer_vocabulary: Sequence[str],
    output_parent: Path,
    generate_fn: Callable | None = None,
    batch_size: int = 16,
) -> tuple[Path, dict]:
    """Run one inference call per fixed state and persist content-addressed results."""
    if generate_fn is None:
        from wordle_lab.protocol.generation import generate as generate_fn
    items = build_probe_items(probe_records, training_records)
    histories = [history_from_record(item) for item in items]
    generated = generate_fn(model, tokenizer, histories, batch_size=batch_size)
    rows, summary = score_probe_outputs(items, generated, allowed_words, answer_vocabulary)
    spec = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_artifact": Path(output_parent).name,
        "item_ids": [item["item_id"] for item in items],
        "answer_vocabulary": sorted(map(normalize_word, answer_vocabulary)),
    }
    artifact_id = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:12]
    output_dir = Path(output_parent) / "diagnostics" / artifact_id
    items_path = write_jsonl(output_dir / "items.jsonl", rows)
    summary.update({"artifact_id": artifact_id, "items_sha256": sha256_file(items_path)})
    write_json(output_dir / "summary.json", summary)
    return output_dir, summary
