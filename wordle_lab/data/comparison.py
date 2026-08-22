from __future__ import annotations

"""Matched, audited data for the Gemma 3 270M representation study."""

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from wordle_lab.common import ROOT, canonical_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from wordle_lab.data.builders import render_representation
from wordle_lab.data.canonical import _facts, generate_canonical_states
from wordle_lab.experiments.common_curriculum import ranked_common_words
from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION, assert_supported_model, load_tokenizer
from wordle_lab.protocol.env import posterior_candidates, score_wordle
from wordle_lab.protocol.oracle import GreedyPartitionOracle


DATASET_ID = "GEMMA-270M-REPRESENTATION-COMPARISON-001"
PARTITIONS = {
    "reasoning_single_step": "state_rationale",
    "non_reasoning_single_step": "state_direct",
    "non_reasoning_multi_step": "episode_multiturn",
}
DEFAULT_TURN_QUOTAS = {1: 128, 2: 1024, 3: 1024, 4: 819, 5: 614, 6: 487}


def default_directory(universe_size: int = 128, train_secret_count: int = 96, states: int = 4096) -> Path:
    return ROOT / "data" / "gemma-270m-comparison-v1" / f"u{universe_size}-train{train_secret_count}-n{states}"


def _key(history: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(canonical_json(list(history)).encode("utf-8")).hexdigest()


def _record(oracle: GreedyPartitionOracle, secret: str, history: list[tuple[str, str]], policy: str) -> dict:
    key = _key(history)
    return {
        "schema_version": "wordle-comparison-state-v1",
        "split": "common_train",
        "state_id": f"common-train-{key[:20]}",
        "secret_answer": secret,
        "history": [{"guess": guess, "feedback": feedback} for guess, feedback in history],
        "turn": len(history) + 1,
        "facts": _facts(oracle, history, secret),
        "source_policy": policy,
    }


def _turn_quotas(states: int) -> dict[int, int]:
    if states < 128:
        raise ValueError("comparison data requires at least 128 source rows")
    if states == sum(DEFAULT_TURN_QUOTAS.values()):
        return dict(DEFAULT_TURN_QUOTAS)
    weights = {1: 0.03125, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.11875}
    quotas = {turn: round(states * value) for turn, value in weights.items()}
    quotas[2] += states - sum(quotas.values())
    return quotas


def _candidate_pools(
    train_secrets: Sequence[str], universe: Sequence[str], quotas: dict[int, int], seed: int
) -> dict[int, list[dict]]:
    oracle = GreedyPartitionOracle(universe)
    rng = random.Random(seed ^ 0xC0A4)
    pools: dict[int, dict[str, dict]] = {turn: {} for turn in range(1, 7)}
    # Root observations are identical, but rotate their episode-secret provenance
    # so duplicated format exposure does not distort the training-secret audit.
    for secret in train_secrets:
        pools[1][f"root-{secret}"] = _record(oracle, secret, [], "root_format_anchor")
    desired = {turn: max(quota * 2, quota + 128) for turn, quota in quotas.items() if turn > 1}
    attempts = 0
    max_attempts = max(50_000, sum(quotas.values()) * 40)
    while attempts < max_attempts and any(len(pools[t]) < desired[t] for t in desired):
        secret = train_secrets[attempts % len(train_secrets)]
        guesses = list(universe)
        rng.shuffle(guesses)
        history: list[tuple[str, str]] = []
        for guess in guesses:
            if guess == secret:
                continue
            history.append((guess, score_wordle(secret, guess)))
            turn = len(history) + 1
            if turn > 6:
                break
            key = _key(history)
            pools[turn].setdefault(key, _record(oracle, secret, history, "varied_legal_unsolved_history"))
        attempts += 1
    short = {turn: (len(pools[turn]), desired[turn]) for turn in desired if len(pools[turn]) < desired[turn]}
    if short:
        raise RuntimeError(f"could not generate enough comparison candidates: {short}")
    return {turn: list(records.values()) for turn, records in pools.items()}


def _select(pools: dict[int, list[dict]], quotas: dict[int, int], seed: int) -> list[dict]:
    selected: list[dict] = []
    target_counts: Counter[str] = Counter()
    secret_counts: Counter[str] = Counter()
    for turn, quota in quotas.items():
        if turn == 1:
            anchors = pools[1]
            selected.extend(dict(anchors[index % len(anchors)]) for index in range(quota))
            target_counts[anchors[0]["facts"]["oracle_action"]] += quota
            for index in range(quota):
                secret_counts[anchors[index % len(anchors)]["secret_answer"]] += 1
            continue
        ranked = sorted(
            pools[turn],
            key=lambda row: hashlib.sha256(f"{seed}:{row['state_id']}".encode()).hexdigest(),
        )
        for _ in range(quota):
            if not ranked:
                raise RuntimeError(f"turn {turn} candidate pool exhausted")
            best_index = min(
                range(len(ranked)),
                key=lambda index: (
                    target_counts[ranked[index]["facts"]["oracle_action"]],
                    secret_counts[ranked[index]["secret_answer"]],
                    index,
                ),
            )
            row = ranked.pop(best_index)
            selected.append(row)
            target_counts[row["facts"]["oracle_action"]] += 1
            secret_counts[row["secret_answer"]] += 1
    random.Random(seed ^ 0x51EC7).shuffle(selected)
    return selected


def _enrich(rendered: list[dict], sources: list[dict], partition: str) -> list[dict]:
    output = []
    for source, row in zip(sources, rendered, strict=True):
        comparison_id = source["comparison_id"]
        output.append({
            **row,
            "schema_version": "wordle-comparison-example-v1",
            "example_id": f"{comparison_id}-{partition}",
            "comparison_id": comparison_id,
            "source_state_id": source["state_id"],
            "partition": partition,
            "target_word": source["facts"]["oracle_action"],
            "posterior_size": source["facts"]["posterior_count"],
        })
    return output


def build_comparison_bundle(
    output_dir: str | Path | None = None,
    *,
    universe_size: int = 128,
    train_secret_count: int = 96,
    states: int = 4096,
    dev_states: int = 512,
    seed: int = 2026,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build three matched views. Dev probes are evaluation-only and never rendered as training rows."""
    assert_supported_model()
    if not 0 < train_secret_count < universe_size:
        raise ValueError("train_secret_count must be between zero and universe_size")
    directory = Path(output_dir) if output_dir else default_directory(universe_size, train_secret_count, states)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not force:
        return directory, read_json(manifest_path)
    universe = ranked_common_words(universe_size)
    shuffled = list(universe)
    random.Random(seed).shuffle(shuffled)
    train_secrets = sorted(shuffled[:train_secret_count])
    dev_secrets = sorted(shuffled[train_secret_count:])
    if set(train_secrets) & set(dev_secrets):
        raise AssertionError("train/dev secret leakage")
    quotas = _turn_quotas(states)
    selected = _select(_candidate_pools(train_secrets, universe, quotas, seed), quotas, seed)
    sources = [{**row, "comparison_id": f"train-{index:06d}"} for index, row in enumerate(selected)]
    write_jsonl(directory / "source_states.jsonl", sources)
    paths: dict[str, Path] = {}
    for partition, representation in PARTITIONS.items():
        rendered = _enrich(render_representation(sources, representation), sources, partition)
        paths[partition] = write_jsonl(directory / f"{partition}.jsonl", rendered)
    dev = generate_canonical_states(
        dev_secrets, "common_dev_probe", dev_states, seed=seed, answer_vocabulary=universe
    )
    write_jsonl(directory / "dev_probe_states.jsonl", dev)
    write_json(directory / "universe.json", universe)
    write_json(directory / "train_secrets.json", train_secrets)
    write_json(directory / "dev_secrets.json", dev_secrets)
    manifest = {
        "dataset_id": DATASET_ID,
        "model": {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION, "exclusive": True},
        "seed": seed,
        "universe_size": len(universe),
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "source_states": len(sources),
        "rendered_training_rows": len(sources) * len(PARTITIONS),
        "rows_per_partition": len(sources),
        "dev_probe_states": len(dev),
        "dev_probe_role": "evaluation_only_never_training",
        "partitions": PARTITIONS,
        "comparison_control": "identical comparison_id, source state, and oracle target across all partitions",
        "turn_quotas": {str(key): value for key, value in quotas.items()},
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in sources).items())),
        "unique_source_states": len({row["state_id"] for row in sources}),
        "unique_oracle_targets": len({row["facts"]["oracle_action"] for row in sources}),
        "target_distribution": dict(sorted(Counter(row["facts"]["oracle_action"] for row in sources).items())),
        "training_secret_distribution": dict(sorted(Counter(row["secret_answer"] for row in sources).items())),
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "source_states_sha256": sha256_file(directory / "source_states.jsonl"),
        "dev_probe_states_sha256": sha256_file(directory / "dev_probe_states.jsonl"),
        "word_list_source": "data/wordlists/tabatkins_wordle_list_pinned.txt ranked by pinned wordfreq dependency",
    }
    write_json(manifest_path, manifest)
    audit = audit_comparison_bundle(directory, include_token_lengths=True)
    manifest["audit"] = audit
    write_json(manifest_path, manifest)
    return directory, manifest


def _terminal_word(row: dict) -> str:
    content = row["completion"][0]["content"].strip()
    marker = "Final answer:"
    if marker not in content:
        raise AssertionError(f"missing terminal answer in {row['example_id']}")
    return content.rsplit(marker, 1)[1].strip().split()[0].upper()


def audit_comparison_bundle(directory: str | Path, *, include_token_lengths: bool = False) -> dict[str, Any]:
    """Recompute labels/history and prove that the three partitions are matched."""
    directory = Path(directory)
    sources = read_jsonl(directory / "source_states.jsonl")
    universe = read_json(directory / "universe.json")
    train_secrets = set(read_json(directory / "train_secrets.json"))
    dev_secrets = set(read_json(directory / "dev_secrets.json"))
    if not sources:
        raise AssertionError("empty comparison source data")
    if train_secrets & dev_secrets:
        raise AssertionError("train/dev secret leakage")
    oracle = GreedyPartitionOracle(universe)
    source_by_id = {row["comparison_id"]: row for row in sources}
    if len(source_by_id) != len(sources):
        raise AssertionError("duplicate comparison_id")
    for row in sources:
        secret = row["secret_answer"]
        if secret not in train_secrets or secret in dev_secrets:
            raise AssertionError(f"non-training secret in {row['comparison_id']}")
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        if row["turn"] != len(history) + 1 or not 1 <= row["turn"] <= 6:
            raise AssertionError(f"bad turn in {row['comparison_id']}")
        if any(score_wordle(secret, guess) != feedback for guess, feedback in history):
            raise AssertionError(f"incorrect feedback in {row['comparison_id']}")
        if any(feedback == "GGGGG" for _, feedback in history):
            raise AssertionError(f"post-solve training state in {row['comparison_id']}")
        if secret not in posterior_candidates(history, universe):
            raise AssertionError(f"secret removed from posterior in {row['comparison_id']}")
        if row["facts"] != _facts(oracle, history, secret):
            raise AssertionError(f"stale oracle facts in {row['comparison_id']}")

    ids = set(source_by_id)
    token_lengths: dict[str, dict[str, int]] = {}
    tokenizer = load_tokenizer() if include_token_lengths else None
    for partition, representation in PARTITIONS.items():
        rows = read_jsonl(directory / f"{partition}.jsonl")
        if len(rows) != len(sources) or {row["comparison_id"] for row in rows} != ids:
            raise AssertionError(f"unmatched rows in {partition}")
        lengths = []
        for row in rows:
            source = source_by_id[row["comparison_id"]]
            target = source["facts"]["oracle_action"]
            if row["source_state_id"] != source["state_id"] or row["target_word"] != target:
                raise AssertionError(f"source/target mismatch in {row['example_id']}")
            if _terminal_word(row) != target:
                raise AssertionError(f"completion mismatch in {row['example_id']}")
            if row["representation"] != representation:
                raise AssertionError(f"representation mismatch in {row['example_id']}")
            content = row["completion"][0]["content"]
            has_reasoning = "Choice rationale:" in content and "Action assessment:" in content
            if has_reasoning != (partition == "reasoning_single_step"):
                raise AssertionError(f"reasoning boundary violation in {row['example_id']}")
            if partition == "non_reasoning_multi_step":
                assistants = [message["content"] for message in row["prompt"] if message["role"] == "assistant"]
                expected = [f"Final answer: {item['guess']}" for item in source["history"]]
                if assistants != expected:
                    raise AssertionError(f"multi-step history mismatch in {row['example_id']}")
            if tokenizer is not None:
                full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
                lengths.append(len(tokenizer(full, add_special_tokens=False)["input_ids"]))
        if lengths:
            token_lengths[partition] = {"max": max(lengths), "mean": round(sum(lengths) / len(lengths))}
    dev = read_jsonl(directory / "dev_probe_states.jsonl")
    if any(row["secret_answer"] not in dev_secrets for row in dev):
        raise AssertionError("dev probes are not held out")
    return {
        "status": "passed",
        "checks": [
            "held_out_secret_split", "feedback_recomputed", "no_post_solve_states",
            "secret_in_posterior", "oracle_facts_recomputed", "matched_comparison_ids",
            "matched_targets", "reasoning_partition_separation", "multi_step_history_fidelity",
            "dev_probes_evaluation_only",
        ],
        "source_rows": len(sources),
        "rendered_rows": len(sources) * len(PARTITIONS),
        "token_lengths": token_lengths,
    }
