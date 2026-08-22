from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import torch
from wordfreq import zipf_frequency

from wordle_lab.common import ARTIFACTS, ROOT, canonical_json, read_jsonl, set_seed, sha256_file, write_json, write_jsonl
from wordle_lab.data.canonical import _facts, generate_canonical_states
from wordle_lab.experiments.intervention_sweep import (
    DECODING_VARIANTS,
    _explicit_feedback_messages,
    _strict_explicit_feedback_messages,
)
from wordle_lab.methods.sft import train_sft
from wordle_lab.models import load_adapter, load_tokenizer, model_metadata
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.oracle import GreedyPartitionOracle
from wordle_lab.analysis.state_diagnostics import run_state_diagnostics


CURRICULUM_ID = "COMMON-WORD-CURRICULUM-001"
BALANCED_CURRICULUM_ID = "COMMON-WORD-CURRICULUM-002"
TARGETED_CURRICULUM_ID = "COMMON-WORD-CURRICULUM-003"
BALANCED_STRICT_CURRICULUM_ID = "COMMON-WORD-CURRICULUM-004"
BALANCED_STRICT_ANCHORED_CURRICULUM_ID = "COMMON-WORD-CURRICULUM-005"
BALANCED_MIXTURE = {
    "root": 0.10,
    "turn_2": 0.40,
    "later_on_policy": 0.30,
    "recovery_singleton": 0.20,
}
TARGETED_MIXTURE = {
    "format_root": 0.05,
    "turn_2": 0.35,
    "low_posterior": 0.25,
    "true_singleton": 0.30,
    "later_broad": 0.05,
}


def ranked_common_words(universe_size: int) -> list[str]:
    source = ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt"
    allowed = {line.strip().upper() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()}
    ranked = sorted(allowed, key=lambda word: (-zipf_frequency(word.lower(), "en"), word))
    if universe_size > len(ranked):
        raise ValueError(f"requested {universe_size} words from {len(ranked)} allowed words")
    return ranked[:universe_size]


def _render_rows(records: list[dict], repeat_fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for record in records:
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        prompt = _explicit_feedback_messages(history)
        completion = [{"role": "assistant", "content": f"Final answer: {record['facts']['oracle_action']}"}]
        rows.append(
            {
                "example_id": f"{record['state_id']}-direct",
                "state_id": record["state_id"],
                "turn": record["turn"],
                "kind": "explicit_direct",
                "prompt": prompt,
                "completion": completion,
            }
        )
        if history and rng.random() < repeat_fraction:
            repeated = history[-1][0]
            correction_prompt = prompt + [
                {"role": "assistant", "content": f"Final answer: {repeated}"},
                {
                    "role": "user",
                    "content": (
                        f"Rejected: {repeated} was already guessed, so it cannot be used again. Apply the feedback "
                        "constraints and provide a different next guess."
                    ),
                },
            ]
            rows.append(
                {
                    "example_id": f"{record['state_id']}-repeat-correction",
                    "state_id": record["state_id"],
                    "turn": record["turn"],
                    "kind": "repeat_correction",
                    "prompt": correction_prompt,
                    "completion": completion,
                }
            )
    # The root state is necessarily unique and successful oracle trajectories
    # make turns 5-6 rare. Give every turn enough optimization exposure without
    # changing the unique-state accounting or fabricating any target actions.
    exposure_floor = {1: 128, 2: 256, 3: 256, 4: 256, 5: 256, 6: 256}
    augmented = list(rows)
    for turn, floor in exposure_floor.items():
        bucket = [row for row in rows if row["turn"] == turn]
        if bucket:
            augmented.extend(bucket[index % len(bucket)] for index in range(max(0, floor - len(bucket))))
    random.Random(seed).shuffle(augmented)
    return augmented


def prepare(
    universe_size: int = 512,
    train_secret_count: int = 384,
    states: int = 2048,
    seed: int = 2026,
    repeat_fraction: float = 1.0,
    force: bool = False,
) -> tuple[Path, dict]:
    if not 0 < train_secret_count < universe_size:
        raise ValueError("train_secret_count must be between zero and universe_size")
    directory = ROOT / "data" / "common-curriculum-001" / f"u{universe_size}-train{train_secret_count}"
    rows_path = directory / "train.jsonl"
    manifest_path = directory / "manifest.json"
    if rows_path.exists() and manifest_path.exists() and not force:
        return rows_path, json.loads(manifest_path.read_text(encoding="utf-8"))

    universe = ranked_common_words(universe_size)
    shuffled = list(universe)
    random.Random(seed).shuffle(shuffled)
    train_secrets = sorted(shuffled[:train_secret_count])
    dev_secrets = sorted(shuffled[train_secret_count:])
    canonical = generate_canonical_states(
        train_secrets,
        "common_train",
        states,
        seed=seed,
        answer_vocabulary=universe,
    )
    rows = _render_rows(canonical, repeat_fraction, seed)
    write_jsonl(directory / "canonical.jsonl", canonical)
    write_jsonl(rows_path, rows)
    write_json(directory / "universe.json", universe)
    write_json(directory / "train_secrets.json", train_secrets)
    write_json(directory / "dev_secrets.json", dev_secrets)
    manifest = {
        "curriculum_id": CURRICULUM_ID,
        "universe_size": universe_size,
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "canonical_states": len(canonical),
        "rendered_examples": len(rows),
        "repeat_correction_examples": sum(row["kind"] == "repeat_correction" for row in rows),
        "unique_rendered_examples": len({row["example_id"] for row in rows}),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "target_unique_words": len({row["completion"][0]["content"] for row in rows}),
        "frequency_cutoff_zipf": zipf_frequency(universe[-1].lower(), "en"),
        "prompt_version": "explicit-constraints-v2-compact",
        "rendered_sha256": sha256_file(rows_path),
        "seed": seed,
        "state_policy_universe": "full_common_universe",
        "secret_split_role": "episode sampling only; the public answer vocabulary is fixed across train and dev",
    }
    write_json(manifest_path, manifest)
    return rows_path, manifest


def _state_key(history: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(canonical_json(list(history)).encode("utf-8")).hexdigest()


def _classify_state(record: dict, synthetic: bool = False) -> str:
    if record["turn"] == 1:
        return "root"
    if synthetic or record["facts"]["posterior_count"] == 1:
        return "recovery_singleton"
    if record["turn"] == 2:
        return "turn_2"
    return "later_on_policy"


def _synthetic_recovery_states(
    train_secrets: Sequence[str], universe: Sequence[str], count: int, seed: int
) -> list[dict]:
    """Create distinct, legal off-policy histories using training secrets only."""
    oracle = GreedyPartitionOracle(universe)
    rng = random.Random(seed ^ 0xDA66E2)
    records: dict[str, dict] = {}
    attempts = 0
    while len(records) < count and attempts < max(1000, count * 80):
        secret = train_secrets[attempts % len(train_secrets)]
        depth = rng.randint(1, 5)
        guesses = rng.sample(list(universe), k=min(depth, len(universe)))
        history: list[tuple[str, str]] = []
        for guess in guesses:
            if guess == secret:
                continue
            history.append((guess, score_wordle(secret, guess)))
            key = _state_key(history)
            if key not in records:
                facts = _facts(oracle, history, secret)
                records[key] = {
                    "schema_version": "wordle-balanced-state-v1",
                    "split": "common_train",
                    "state_id": f"common_train-recovery-{key[:16]}",
                    "episode_id": f"recovery-{attempts:06d}",
                    "secret_answer": secret,
                    "history": [{"guess": word, "feedback": feedback} for word, feedback in history],
                    "turn": len(history) + 1,
                    "facts": facts,
                    "state_type": "recovery_singleton",
                    "prior_policy": "varied_legal",
                }
            if len(records) >= count:
                break
        attempts += 1
    return sorted(records.values(), key=lambda row: row["state_id"])


def _balanced_select(
    pools: dict[str, list[dict]], total: int, seed: int, state_cap: int, target_cap: int
) -> list[tuple[dict, str]]:
    if state_cap < 1 or target_cap < 1:
        raise ValueError("state_cap and target_cap must be positive")
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    quotas = {kind: round(total * fraction) for kind, fraction in BALANCED_MIXTURE.items()}
    quotas["turn_2"] += total - sum(quotas.values())
    selected: list[tuple[dict, str]] = []
    state_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    def take(kind: str, limit: int) -> None:
        candidates = pools.get(kind, [])
        cursor = 0
        while sum(selected_kind == kind for _, selected_kind in selected) < limit and candidates:
            record = candidates[cursor % len(candidates)]
            cursor += 1
            state_id = record["state_id"]
            target = record["facts"]["oracle_action"]
            if state_counts[state_id] < state_cap and target_counts[target] < target_cap:
                selected.append((record, kind))
                state_counts[state_id] += 1
                target_counts[target] += 1
            if cursor >= len(candidates) * state_cap:
                break

    for kind in BALANCED_MIXTURE:
        take(kind, quotas[kind])
    # Scarce root states and per-target caps can make the requested proportions
    # infeasible. Fill from distinct underrepresented states and report achieved mix.
    for kind in ("turn_2", "later_on_policy", "recovery_singleton", "root"):
        if len(selected) >= total:
            break
        take(kind, sum(selected_kind == kind for _, selected_kind in selected) + total - len(selected))
    return selected[:total]


def prepare_balanced(
    universe_size: int = 128,
    train_secret_count: int = 96,
    states: int = 512,
    seed: int = 2026,
    state_cap: int = 4,
    target_cap: int = 8,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build a versioned, inference-shaped curriculum with auditable caps."""
    if not 0 < train_secret_count < universe_size:
        raise ValueError("train_secret_count must be between zero and universe_size")
    directory = ROOT / "data" / "common-curriculum-002" / f"u{universe_size}-train{train_secret_count}"
    rows_path, manifest_path = directory / "train.jsonl", directory / "manifest.json"
    if rows_path.exists() and manifest_path.exists() and not force:
        return rows_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    universe = ranked_common_words(universe_size)
    shuffled = list(universe)
    random.Random(seed).shuffle(shuffled)
    train_secrets = sorted(shuffled[:train_secret_count])
    dev_secrets = sorted(shuffled[train_secret_count:])
    if set(train_secrets) & set(dev_secrets):
        raise AssertionError("train/dev secret leakage")
    canonical_count = max(states, min(states * 2, 4096))
    canonical = generate_canonical_states(
        train_secrets, "common_train", canonical_count, seed=seed, answer_vocabulary=universe
    )
    recovery = _synthetic_recovery_states(train_secrets, universe, max(states, 128), seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    for record in canonical:
        pools[_classify_state(record)].append(record)
    pools["recovery_singleton"].extend(recovery)
    selected = _balanced_select(pools, states, seed, state_cap, target_cap)
    rows, state_manifest = [], []
    target_frequency = Counter(record["facts"]["oracle_action"] for record, _ in selected)
    for index, (record, state_type) in enumerate(selected):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        target = record["facts"]["oracle_action"]
        prompt = _explicit_feedback_messages(history)
        row = {
            "example_id": f"balanced-{index:06d}-{record['state_id']}",
            "state_id": record["state_id"],
            "source_state": record,
            "state_type": state_type,
            "turn": record["turn"],
            "posterior_size": record["facts"]["posterior_count"],
            "target_word": target,
            "target_frequency": target_frequency[target],
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
        }
        rows.append(row)
        state_manifest.append({key: row[key] for key in (
            "example_id", "state_id", "state_type", "turn", "posterior_size", "target_word", "target_frequency"
        )})
    labelled_secrets = {row["source_state"]["secret_answer"] for row in rows}
    if not labelled_secrets <= set(train_secrets) or labelled_secrets & set(dev_secrets):
        raise AssertionError("balanced labels contain a held-out secret")
    write_jsonl(directory / "canonical.jsonl", canonical)
    write_jsonl(directory / "recovery_states.jsonl", recovery)
    write_jsonl(rows_path, rows)
    state_manifest_path = write_jsonl(directory / "state_manifest.jsonl", state_manifest)
    write_json(directory / "universe.json", universe)
    write_json(directory / "train_secrets.json", train_secrets)
    write_json(directory / "dev_secrets.json", dev_secrets)
    achieved = Counter(row["state_type"] for row in rows)
    manifest = {
        "curriculum_id": BALANCED_CURRICULUM_ID,
        "universe_size": universe_size,
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "rendered_examples": len(rows),
        "unique_source_states": len({row["state_id"] for row in rows}),
        "state_copy_cap": state_cap,
        "target_word_cap": target_cap,
        "requested_mixture": BALANCED_MIXTURE,
        "achieved_composition": dict(sorted(achieved.items())),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "target_frequency_distribution": dict(sorted(target_frequency.items())),
        "prompt_version": "explicit-constraints-v2-compact",
        "prompt_renderer": "wordle_lab.experiments.intervention_sweep._explicit_feedback_messages",
        "rendered_sha256": sha256_file(rows_path),
        "state_manifest_sha256": sha256_file(state_manifest_path),
        "seed": seed,
        "labelled_episode_secrets": "training split only",
    }
    write_json(manifest_path, manifest)
    return rows_path, manifest


def _targeted_state_type(history: Sequence[tuple[str, str]], posterior_count: int) -> str:
    if not history:
        return "format_root"
    if posterior_count == 1:
        return "true_singleton"
    if len(history) == 1:
        return "turn_2"
    if posterior_count <= 4:
        return "low_posterior"
    return "later_broad"


def _targeted_state_pools(
    train_secrets: Sequence[str], universe: Sequence[str], total: int, seed: int
) -> dict[str, list[dict]]:
    """Generate varied, training-secret-only states grouped by actual difficulty."""
    oracle = GreedyPartitionOracle(universe)
    rng = random.Random(seed ^ 0xC003)
    pools: dict[str, dict[str, dict]] = {kind: {} for kind in TARGETED_MIXTURE}

    def add(secret: str, history: list[tuple[str, str]], source_policy: str) -> None:
        facts = _facts(oracle, history, secret)
        state_type = _targeted_state_type(history, int(facts["posterior_count"]))
        key = _state_key(history)
        pools[state_type].setdefault(
            key,
            {
                "schema_version": "wordle-targeted-state-v1",
                "split": "common_train",
                "state_id": f"common_train-targeted-{key[:16]}",
                "episode_id": f"targeted-{secret}-{key[:10]}",
                "secret_answer": secret,
                "history": [{"guess": guess, "feedback": feedback} for guess, feedback in history],
                "turn": len(history) + 1,
                "facts": facts,
                "state_type": state_type,
                "prior_policy": source_policy,
            },
        )

    add(train_secrets[0], [], "format_anchor")
    secrets = list(train_secrets)
    guesses = list(universe)
    rng.shuffle(secrets)
    rng.shuffle(guesses)
    # Exhaustive one-guess states provide broad feedback-pattern coverage and
    # many minimal singleton demonstrations without using held-out episodes.
    for secret in secrets:
        for guess in guesses:
            if guess != secret:
                add(secret, [(guess, score_wordle(secret, guess))], "varied_opener_exhaustive")

    required = {
        kind: max(32, round(total * fraction * 3))
        for kind, fraction in TARGETED_MIXTURE.items()
        if kind not in {"format_root", "turn_2"}
    }
    max_attempts = max(20_000, total * 80)
    for attempt in range(max_attempts):
        if all(len(pools[kind]) >= count for kind, count in required.items()):
            break
        secret = secrets[attempt % len(secrets)]
        depth = 2 if rng.random() < 0.7 else 3
        sampled = rng.sample([word for word in universe if word != secret], k=depth)
        history = [(guess, score_wordle(secret, guess)) for guess in sampled]
        add(secret, history, "varied_legal_multiturn")
    return {kind: sorted(records.values(), key=lambda row: row["state_id"]) for kind, records in pools.items()}


def _targeted_select(
    pools: dict[str, list[dict]], total: int, seed: int, target_cap: int
) -> list[tuple[dict, str]]:
    """Select exact posterior-bucket quotas while balancing action labels."""
    if target_cap < 1:
        raise ValueError("target_cap must be positive")
    quotas = {kind: round(total * fraction) for kind, fraction in TARGETED_MIXTURE.items()}
    quotas["turn_2"] += total - sum(quotas.values())
    rng = random.Random(seed ^ 0x5E1EC7)
    selected: list[tuple[dict, str]] = []
    target_counts: Counter[str] = Counter()

    for kind, quota in quotas.items():
        candidates = list(pools.get(kind, []))
        if kind == "format_root":
            if not candidates:
                raise RuntimeError("targeted curriculum has no root format anchor")
            selected.extend((candidates[0], kind) for _ in range(quota))
            continue
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in candidates:
            grouped[record["facts"]["oracle_action"]].append(record)
        for records in grouped.values():
            rng.shuffle(records)
        target_order = list(grouped)
        rng.shuffle(target_order)
        chosen = 0
        while chosen < quota:
            progressed = False
            target_order.sort(key=lambda target: target_counts[target])
            for target in target_order:
                if chosen >= quota:
                    break
                if target_counts[target] >= target_cap or not grouped[target]:
                    continue
                selected.append((grouped[target].pop(), kind))
                target_counts[target] += 1
                chosen += 1
                progressed = True
            if not progressed:
                raise RuntimeError(
                    f"targeted pool cannot fill {kind} quota {quota}; selected {chosen}, "
                    f"pool={len(candidates)}, target_cap={target_cap}"
                )
    return selected


def prepare_targeted(
    universe_size: int = 128,
    train_secret_count: int = 96,
    states: int = 1024,
    seed: int = 2026,
    target_cap: int = 16,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build curriculum 003 with true singleton and varied-opener supervision."""
    if not 0 < train_secret_count < universe_size:
        raise ValueError("train_secret_count must be between zero and universe_size")
    directory = ROOT / "data" / "common-curriculum-003" / f"u{universe_size}-train{train_secret_count}"
    rows_path, manifest_path = directory / "train.jsonl", directory / "manifest.json"
    if rows_path.exists() and manifest_path.exists() and not force:
        return rows_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    universe = ranked_common_words(universe_size)
    shuffled = list(universe)
    random.Random(seed).shuffle(shuffled)
    train_secrets = sorted(shuffled[:train_secret_count])
    dev_secrets = sorted(shuffled[train_secret_count:])
    if set(train_secrets) & set(dev_secrets):
        raise AssertionError("train/dev secret leakage")
    pools = _targeted_state_pools(train_secrets, universe, states, seed)
    selected = _targeted_select(pools, states, seed, target_cap)
    rows, state_manifest = [], []
    target_frequency = Counter(record["facts"]["oracle_action"] for record, _ in selected)
    for index, (record, state_type) in enumerate(selected):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        target = record["facts"]["oracle_action"]
        row = {
            "example_id": f"targeted-{index:06d}-{record['state_id']}",
            "state_id": record["state_id"],
            "source_state": record,
            "state_type": state_type,
            "turn": record["turn"],
            "posterior_size": record["facts"]["posterior_count"],
            "target_word": target,
            "target_frequency": target_frequency[target],
            "prompt": _strict_explicit_feedback_messages(history),
            "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
        }
        rows.append(row)
        state_manifest.append({key: row[key] for key in (
            "example_id", "state_id", "state_type", "turn", "posterior_size", "target_word", "target_frequency"
        )})
    labelled_secrets = {row["source_state"]["secret_answer"] for row in rows}
    if not labelled_secrets <= set(train_secrets) or labelled_secrets & set(dev_secrets):
        raise AssertionError("targeted labels contain a held-out episode secret")
    if any(row["posterior_size"] != 1 for row in rows if row["state_type"] == "true_singleton"):
        raise AssertionError("true_singleton bucket contains a non-singleton state")
    non_root_counts = Counter(row["target_word"] for row in rows if row["state_type"] != "format_root")
    if non_root_counts and max(non_root_counts.values()) > target_cap:
        raise AssertionError("target word cap exceeded")
    write_jsonl(directory / "candidate_states.jsonl", [record for kind in pools.values() for record in kind])
    write_jsonl(rows_path, rows)
    state_manifest_path = write_jsonl(directory / "state_manifest.jsonl", state_manifest)
    write_json(directory / "universe.json", universe)
    write_json(directory / "train_secrets.json", train_secrets)
    write_json(directory / "dev_secrets.json", dev_secrets)
    manifest = {
        "curriculum_id": TARGETED_CURRICULUM_ID,
        "universe_size": universe_size,
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "rendered_examples": len(rows),
        "unique_source_states": len({row["state_id"] for row in rows}),
        "format_root_copies": sum(row["state_type"] == "format_root" for row in rows),
        "non_root_state_copy_cap": 1,
        "non_root_target_word_cap": target_cap,
        "requested_mixture": TARGETED_MIXTURE,
        "achieved_composition": dict(sorted(Counter(row["state_type"] for row in rows).items())),
        "posterior_distribution": dict(sorted(Counter(str(row["posterior_size"]) for row in rows).items())),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "target_frequency_distribution": dict(sorted(target_frequency.items())),
        "candidate_pool_composition": {kind: len(records) for kind, records in pools.items()},
        "prompt_version": "strict-explicit-constraints-v3-five-ascii",
        "prompt_renderer": "wordle_lab.experiments.intervention_sweep._strict_explicit_feedback_messages",
        "rendered_sha256": sha256_file(rows_path),
        "state_manifest_sha256": sha256_file(state_manifest_path),
        "seed": seed,
        "labelled_episode_secrets": "training split only",
    }
    write_json(manifest_path, manifest)
    return rows_path, manifest


def prepare_balanced_strict(
    universe_size: int = 128,
    train_secret_count: int = 96,
    states: int = 512,
    seed: int = 2026,
    force: bool = False,
) -> tuple[Path, dict]:
    """Rerender the proven balanced states with the strict five-letter prompt."""
    source_path, source_manifest = prepare_balanced(
        universe_size, train_secret_count, states, seed
    )
    source_dir = source_path.parent
    directory = ROOT / "data" / "common-curriculum-004" / f"u{universe_size}-train{train_secret_count}"
    rows_path, manifest_path = directory / "train.jsonl", directory / "manifest.json"
    if rows_path.exists() and manifest_path.exists() and not force:
        return rows_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(source_path)
    for row in rows:
        history = [(item["guess"], item["feedback"]) for item in row["source_state"]["history"]]
        row["prompt"] = _strict_explicit_feedback_messages(history)
    write_jsonl(rows_path, rows)
    canonical_path = write_jsonl(directory / "canonical.jsonl", read_jsonl(source_dir / "canonical.jsonl"))
    state_manifest_path = write_jsonl(
        directory / "state_manifest.jsonl",
        [{key: row[key] for key in (
            "example_id", "state_id", "state_type", "turn", "posterior_size", "target_word", "target_frequency"
        )} for row in rows],
    )
    for name in ("universe.json", "train_secrets.json", "dev_secrets.json"):
        write_json(directory / name, json.loads((source_dir / name).read_text(encoding="utf-8")))
    manifest = {
        **source_manifest,
        "curriculum_id": BALANCED_STRICT_CURRICULUM_ID,
        "parent_curriculum_id": source_manifest["curriculum_id"],
        "parent_rendered_sha256": source_manifest["rendered_sha256"],
        "prompt_version": "strict-explicit-constraints-v3-five-ascii",
        "prompt_renderer": "wordle_lab.experiments.intervention_sweep._strict_explicit_feedback_messages",
        "rendered_sha256": sha256_file(rows_path),
        "canonical_sha256": sha256_file(canonical_path),
        "state_manifest_sha256": sha256_file(state_manifest_path),
    }
    write_json(manifest_path, manifest)
    return rows_path, manifest


def prepare_balanced_strict_anchored(
    universe_size: int = 128,
    train_secret_count: int = 96,
    states: int = 512,
    seed: int = 2026,
    force: bool = False,
) -> tuple[Path, dict]:
    """Add explicit root-format exposure to the strict balanced curriculum."""
    source_path, source_manifest = prepare_balanced_strict(
        universe_size, train_secret_count, states, seed
    )
    source_dir = source_path.parent
    directory = ROOT / "data" / "common-curriculum-005" / f"u{universe_size}-train{train_secret_count}"
    rows_path, manifest_path = directory / "train.jsonl", directory / "manifest.json"
    if rows_path.exists() and manifest_path.exists() and not force:
        return rows_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(source_path)
    root_rows = [row for row in rows if row["state_type"] == "root"]
    if not root_rows:
        raise RuntimeError("balanced strict curriculum has no root row")
    desired_root_copies = 51
    for index in range(max(0, desired_root_copies - len(root_rows))):
        anchor = dict(root_rows[index % len(root_rows)])
        anchor["example_id"] = f"format-root-anchor-{index:04d}-{anchor['state_id']}"
        anchor["state_type"] = "format_root_anchor"
        rows.append(anchor)
    random.Random(seed ^ 0xA11CE).shuffle(rows)
    target_frequency = Counter(row["target_word"] for row in rows)
    for row in rows:
        row["target_frequency"] = target_frequency[row["target_word"]]
    write_jsonl(rows_path, rows)
    canonical_path = write_jsonl(directory / "canonical.jsonl", read_jsonl(source_dir / "canonical.jsonl"))
    state_manifest_path = write_jsonl(
        directory / "state_manifest.jsonl",
        [{key: row[key] for key in (
            "example_id", "state_id", "state_type", "turn", "posterior_size", "target_word", "target_frequency"
        )} for row in rows],
    )
    for name in ("universe.json", "train_secrets.json", "dev_secrets.json"):
        write_json(directory / name, json.loads((source_dir / name).read_text(encoding="utf-8")))
    manifest = {
        **source_manifest,
        "curriculum_id": BALANCED_STRICT_ANCHORED_CURRICULUM_ID,
        "parent_curriculum_id": source_manifest["curriculum_id"],
        "parent_rendered_sha256": source_manifest["rendered_sha256"],
        "rendered_examples": len(rows),
        "unique_source_states": len({row["state_id"] for row in rows}),
        "format_root_copies": sum(row["state_type"] in {"root", "format_root_anchor"} for row in rows),
        "format_root_state_cap_exception": True,
        "target_word_cap_excludes_format_root": True,
        "achieved_composition": dict(sorted(Counter(row["state_type"] for row in rows).items())),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "target_frequency_distribution": dict(sorted(target_frequency.items())),
        "rendered_sha256": sha256_file(rows_path),
        "canonical_sha256": sha256_file(canonical_path),
        "state_manifest_sha256": sha256_file(state_manifest_path),
    }
    write_json(manifest_path, manifest)
    return rows_path, manifest


def train_and_evaluate(
    universe_size: int = 512,
    train_secret_count: int = 384,
    states: int = 2048,
    max_steps: int = 400,
    dev_games: int = 25,
    seed: int = 2026,
    learning_rate: float = 1e-4,
    parent_run_id: str | None = None,
    dataset_version: str = "current",
    word_token_weight: float = 1.0,
) -> tuple[str, dict]:
    if dataset_version not in {"current", "balanced", "targeted", "balanced_strict", "balanced_strict_anchored"}:
        raise ValueError("unknown common curriculum dataset version")
    if dataset_version == "balanced_strict_anchored":
        rows_path, data_manifest = prepare_balanced_strict_anchored(universe_size, train_secret_count, states, seed)
        prompt_builder = _strict_explicit_feedback_messages
    elif dataset_version == "balanced_strict":
        rows_path, data_manifest = prepare_balanced_strict(universe_size, train_secret_count, states, seed)
        prompt_builder = _strict_explicit_feedback_messages
    elif dataset_version == "targeted":
        rows_path, data_manifest = prepare_targeted(universe_size, train_secret_count, states, seed)
        prompt_builder = _strict_explicit_feedback_messages
    elif dataset_version == "balanced":
        rows_path, data_manifest = prepare_balanced(universe_size, train_secret_count, states, seed)
        prompt_builder = _explicit_feedback_messages
    else:
        rows_path, data_manifest = prepare(universe_size, train_secret_count, states, seed)
        prompt_builder = _explicit_feedback_messages
    rows = read_jsonl(rows_path)
    spec = {
        "method": "sft",
        "representation": f"common_{dataset_version}_curriculum",
        "seed": seed,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": int(os.environ.get("WORDLE_MICRO_BATCH_SIZE", "4")),
        "gradient_accumulation_steps": int(os.environ.get("WORDLE_GRADIENT_ACCUMULATION_STEPS", "1")),
        "max_length": 320,
        "prompt_version": data_manifest["prompt_version"],
        "word_token_weight": word_token_weight,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "curriculum": data_manifest,
        "model": model_metadata(),
    }
    if parent_run_id:
        parent_checkpoint = ARTIFACTS / "runs" / parent_run_id / "checkpoints" / "final"
        if not parent_checkpoint.exists():
            raise FileNotFoundError(f"parent adapter not found: {parent_checkpoint}")
        spec.update({"parent_run_id": parent_run_id, "parent_checkpoint": str(parent_checkpoint)})
    run_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    loss_name = "word" if word_token_weight > 1 else "completion"
    run_id = f"sft-common-{dataset_version}-{loss_name}-s{seed}-{run_hash}"
    run_dir = ARTIFACTS / "runs" / run_id
    if (run_dir / "summary.json").exists():
        return run_id, json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", data_manifest)
    set_seed(seed)
    model, accounting = train_sft(rows, run_dir, spec)
    tokenizer = load_tokenizer(run_dir / "checkpoints" / "final")
    common_dir = rows_path.parent
    universe = json.loads((common_dir / "universe.json").read_text(encoding="utf-8"))
    dev_answers = json.loads((common_dir / "dev_secrets.json").read_text(encoding="utf-8"))[:dev_games]
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original_messages = generation.inference_messages
    original_config = dict(generation.GENERATION_CONFIG)
    try:
        generation.inference_messages = prompt_builder
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update({"do_sample": False, "max_new_tokens": 128, "use_cache": True})
        games, summary = evaluate(model, tokenizer, dev_answers, allowed, universe)
        valid_guesses = [turn["guess"] for game in games for turn in game["turns"] if turn["valid"]]
        summary.update(
            {
                "run_id": run_id,
                "curriculum_id": data_manifest["curriculum_id"],
                "dev_secret_split": "held-out",
                "unique_guesses": len(set(valid_guesses)),
                "accounting": accounting,
            }
        )
        write_jsonl(run_dir / "games.jsonl", games)
        dev_records = generate_canonical_states(
            dev_answers,
            "common_dev_diagnostic",
            max(len(dev_answers), min(len(dev_answers) * 4, 256)),
            seed=seed,
            answer_vocabulary=universe,
        )
        training_records_path = (
            common_dir / "candidate_states.jsonl"
            if data_manifest["curriculum_id"] == TARGETED_CURRICULUM_ID
            else common_dir / "canonical.jsonl"
        )
        training_records = read_jsonl(training_records_path)
        _, diagnostic_summary = run_state_diagnostics(
            model, tokenizer, dev_records, training_records, allowed, universe, run_dir
        )
        summary["state_diagnostics"] = diagnostic_summary
        write_json(run_dir / "summary.json", summary)
    finally:
        generation.inference_messages = original_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(original_config)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return run_id, summary


def evaluate_saved_checkpoint(
    run_id: str,
    checkpoint: str,
    dev_games: int = 32,
    decoder: str = "greedy",
    prompt_variant: str = "curriculum",
) -> dict:
    run_dir = ARTIFACTS / "runs" / run_id
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    checkpoint_dir = run_dir / "checkpoints" / checkpoint
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    if decoder not in DECODING_VARIANTS:
        raise ValueError(f"unknown decoder: {decoder}")
    if prompt_variant not in {"curriculum", "explicit", "strict"}:
        raise ValueError(f"unknown prompt variant: {prompt_variant}")
    curriculum = spec["curriculum"]
    if curriculum.get("curriculum_id") == BALANCED_STRICT_ANCHORED_CURRICULUM_ID:
        curriculum_folder = "common-curriculum-005"
        prompt_builder = _strict_explicit_feedback_messages
        training_records_name = "canonical.jsonl"
    elif curriculum.get("curriculum_id") == BALANCED_STRICT_CURRICULUM_ID:
        curriculum_folder = "common-curriculum-004"
        prompt_builder = _strict_explicit_feedback_messages
        training_records_name = "canonical.jsonl"
    elif curriculum.get("curriculum_id") == TARGETED_CURRICULUM_ID:
        curriculum_folder = "common-curriculum-003"
        prompt_builder = _strict_explicit_feedback_messages
        training_records_name = "candidate_states.jsonl"
    elif curriculum.get("curriculum_id") == BALANCED_CURRICULUM_ID:
        curriculum_folder = "common-curriculum-002"
        prompt_builder = _explicit_feedback_messages
        training_records_name = "canonical.jsonl"
    else:
        curriculum_folder = "common-curriculum-001"
        prompt_builder = _explicit_feedback_messages
        training_records_name = "canonical.jsonl"
    if prompt_variant == "explicit":
        prompt_builder = _explicit_feedback_messages
    elif prompt_variant == "strict":
        prompt_builder = _strict_explicit_feedback_messages
    common_dir = ROOT / "data" / curriculum_folder / f"u{curriculum['universe_size']}-train{curriculum['train_secret_count']}"
    universe = json.loads((common_dir / "universe.json").read_text(encoding="utf-8"))
    dev_answers = json.loads((common_dir / "dev_secrets.json").read_text(encoding="utf-8"))[:dev_games]
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else run_dir / "checkpoints" / "final"
    tokenizer = load_tokenizer(tokenizer_source)
    model = load_adapter(checkpoint_dir)
    original_messages = generation.inference_messages
    original_config = dict(generation.GENERATION_CONFIG)
    try:
        generation.inference_messages = prompt_builder
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(DECODING_VARIANTS[decoder])
        games, summary = evaluate(model, tokenizer, dev_answers, allowed, universe)
        valid_guesses = [turn["guess"] for game in games for turn in game["turns"] if turn["valid"]]
        summary.update(
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "decoder": decoder,
                "prompt_variant": prompt_variant,
                "unique_guesses": len(set(valid_guesses)),
            }
        )
        artifact_suffix = f"{checkpoint}-{decoder}" + ("" if prompt_variant == "curriculum" else f"-{prompt_variant}")
        write_jsonl(run_dir / f"dose-{artifact_suffix}-games.jsonl", games)
        dev_records = generate_canonical_states(
            dev_answers, "common_dev_diagnostic", max(len(dev_answers), min(len(dev_answers) * 4, 256)),
            seed=int(spec["seed"]), answer_vocabulary=universe,
        )
        _, diagnostic_summary = run_state_diagnostics(
            model, tokenizer, dev_records, read_jsonl(common_dir / training_records_name), allowed, universe,
            run_dir / f"dose-{artifact_suffix}",
        )
        summary["state_diagnostics"] = diagnostic_summary
        write_json(run_dir / f"dose-{artifact_suffix}-summary.json", summary)
        return summary
    finally:
        generation.inference_messages = original_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(original_config)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Common-word, explicit-feedback SFT curriculum pilot")
    parser.add_argument("command", choices=("prepare", "prepare-balanced", "prepare-targeted", "prepare-balanced-strict", "prepare-balanced-strict-anchored", "train", "evaluate"))
    parser.add_argument("--universe-size", type=int, default=512)
    parser.add_argument("--train-secrets", type=int, default=384)
    parser.add_argument("--states", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dev-games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--decoder", choices=tuple(DECODING_VARIANTS), default="greedy")
    parser.add_argument("--prompt-variant", choices=("curriculum", "explicit", "strict"), default="curriculum")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--dataset-version", choices=("current", "balanced", "targeted", "balanced_strict", "balanced_strict_anchored"), default="current")
    parser.add_argument("--word-token-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path, result = prepare(args.universe_size, args.train_secrets, args.states, args.seed, force=args.force)
        result = {**result, "path": str(path)}
    elif args.command == "prepare-balanced":
        path, result = prepare_balanced(args.universe_size, args.train_secrets, args.states, args.seed, force=args.force)
        result = {**result, "path": str(path)}
    elif args.command == "prepare-targeted":
        path, result = prepare_targeted(args.universe_size, args.train_secrets, args.states, args.seed, force=args.force)
        result = {**result, "path": str(path)}
    elif args.command == "prepare-balanced-strict":
        path, result = prepare_balanced_strict(args.universe_size, args.train_secrets, args.states, args.seed, force=args.force)
        result = {**result, "path": str(path)}
    elif args.command == "prepare-balanced-strict-anchored":
        path, result = prepare_balanced_strict_anchored(args.universe_size, args.train_secrets, args.states, args.seed, force=args.force)
        result = {**result, "path": str(path)}
    elif args.command == "train":
        run_id, summary = train_and_evaluate(
            args.universe_size,
            args.train_secrets,
            args.states,
            args.steps,
            args.dev_games,
            args.seed,
            args.learning_rate,
            args.parent_run_id,
            args.dataset_version,
            args.word_token_weight,
        )
        result = {"run_id": run_id, **summary}
    else:
        if not args.run_id:
            parser.error("evaluate requires --run-id")
        result = evaluate_saved_checkpoint(args.run_id, args.checkpoint, args.dev_games, args.decoder, args.prompt_variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
