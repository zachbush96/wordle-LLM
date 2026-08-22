from __future__ import annotations

import random
from collections.abc import Sequence

from wordle_lab.common import canonical_json, sha256_text
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.oracle import GreedyPartitionOracle


def _facts(oracle: GreedyPartitionOracle, history: list[tuple[str, str]], secret: str) -> dict:
    remaining = oracle.remaining(history)
    words = [oracle.answers[int(i)] for i in remaining]
    ranked = oracle.ranked(remaining)
    chosen = ranked[0]
    hard = next((row for row in ranked[1:] if row["guess"] != secret and row["regret"] > 0), None)
    if hard is None:
        hard = next((row for row in ranked[1:] if row["guess"] != secret), chosen)
    fixed = {str(i + 1): letters[0] for i in range(5) if len(letters := sorted({word[i] for word in words})) == 1}
    present = sorted(set.intersection(*(set(word) for word in words))) if words else []
    seen = set("".join(guess for guess, _ in history))
    possible = set("".join(words))
    excluded = sorted(seen - possible)
    return {
        "posterior_count": len(words),
        "fixed_positions": fixed,
        "letters_in_every_candidate": present,
        "excluded_seen_letters": excluded,
        "oracle_action": chosen["guess"],
        "oracle_expected_remaining": chosen["expected_remaining"],
        "oracle_entropy": chosen["entropy"],
        "hard_negative": hard["guess"],
        "hard_negative_expected_remaining": hard["expected_remaining"],
        "hard_negative_regret": hard["regret"],
    }


def generate_canonical_states(
    answers: Sequence[str],
    split: str,
    target_count: int,
    seed: int = 1337,
    opener_count: int = 32,
    answer_vocabulary: Sequence[str] | None = None,
) -> list[dict]:
    """Generate deterministic states.

    ``answers`` are episode secrets. ``answer_vocabulary`` is the public answer
    universe used by the policy and defaults to the historical split-local
    behavior. Keeping these concepts separate prevents train/eval policy drift
    in new studies without changing protocol-002 artifacts.
    """
    oracle = GreedyPartitionOracle(answer_vocabulary or answers)
    secrets = sorted({word.strip().upper() for word in answers})
    missing = sorted(set(secrets) - set(oracle.answers))
    if missing:
        raise ValueError(f"episode secrets absent from answer vocabulary: {missing[:5]}")
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    openers = rng.sample(oracle.answers, k=min(opener_count, len(oracle.answers)))
    records: dict[str, dict] = {}
    episode = 0
    max_episodes = max(target_count * 20, len(answers) * 4)
    while len(records) < target_count and episode < max_episodes:
        secret = secrets[episode % len(secrets)]
        variant = episode // len(secrets)
        history: list[tuple[str, str]] = []
        if variant:
            opener = openers[(variant - 1) % len(openers)]
            if opener != secret:
                history.append((opener, score_wordle(secret, opener)))
        for _turn in range(6 - len(history)):
            key = sha256_text(canonical_json(history))
            if key not in records:
                facts = _facts(oracle, history, secret)
                record = {
                    "schema_version": "wordle-canonical-state-v2",
                    "split": split,
                    "state_id": f"{split}-{key[:16]}",
                    "episode_id": f"{split}-episode-{episode:06d}",
                    "secret_answer": secret,
                    "history": [{"guess": guess, "feedback": feedback} for guess, feedback in history],
                    "turn": len(history) + 1,
                    "facts": facts,
                }
                records[key] = record
                if len(records) >= target_count:
                    break
            action = records[key]["facts"]["oracle_action"] if key in records else oracle.best(history)["guess"]
            feedback = score_wordle(secret, action)
            history.append((action, feedback))
            if feedback == "GGGGG":
                break
        episode += 1
    if len(records) < target_count:
        raise RuntimeError(f"generated only {len(records)}/{target_count} unique {split} states")
    return sorted(records.values(), key=lambda row: (row["turn"], row["state_id"]))[:target_count]
