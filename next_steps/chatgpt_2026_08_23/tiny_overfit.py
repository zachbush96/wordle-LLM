from __future__ import annotations

"""Thirty-two-state memorization diagnostics recommended before larger runs."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import torch

from wordle_lab.common import (
    ROOT,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from wordle_lab.methods.sft import Collator, CompletionDataset
from wordle_lab.models import load_adapter, load_base_model, load_tokenizer
from wordle_lab.protocol.env import is_five_ascii_letters, normalize_word, posterior_candidates, score_wordle
from wordle_lab.protocol.generation import stop_token_ids
from wordle_lab.protocol.parsing import parse_terminal_answer


TINY_OVERFIT_ID = "GEMMA-270M-TINY-OVERFIT-001"
SUITE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "common-curriculum-002" / "u128-train96"
DEFAULT_OUTPUT = SUITE_ROOT / "generated" / "tiny_overfit"
EXPECTED_SOURCE_DIRECTORY = "data/common-curriculum-002/u128-train96"
EXPECTED_SOURCE_HASHES = {
    "manifest.json": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
    "train.jsonl": "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f",
    "state_manifest.jsonl": "4ab23b5cd883d8ad9b542befadc23c2aec3a3d631b78f239bb551ca998fd6a3c",
    "universe.json": "1256cd1c1075246251cafb4d01612dae26a73808a4915c3d88006478f3f736ac",
    "train_secrets.json": "e8ace1e06a6f35a1b600702099c029e232b6124fe76265a5cf4da2d981386a4e",
    "dev_secrets.json": "e94dea81d06f464a55ea7463b36837c998d1e405ef3f1e6e0500c78ea627c8a2",
}
COMMON_ROW_FIELDS = {
    "example_id",
    "state_id",
    "split",
    "task",
    "turn",
    "posterior_size",
    "history",
    "target_word",
    "prompt",
    "completion",
}
GENERAL_ROW_FIELDS = COMMON_ROW_FIELDS | {
    "pair_id",
    "pair_member",
    "feedback_hamming_distance",
}


def _history(row: dict[str, Any]) -> list[dict[str, str]]:
    source = row.get("source_state", row)
    return [
        {"guess": str(item["guess"]).upper(), "feedback": str(item["feedback"]).upper()}
        for item in source["history"]
    ]


def _history_tuples(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [(item["guess"], item["feedback"]) for item in _history(row)]


def _source_contract(
    source_dir: Path = DEFAULT_SOURCE,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Verify the exact balanced-002 source and recompute its Wordle facts."""

    source_dir = Path(source_dir).resolve()
    observed_hashes: dict[str, str] = {}
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        path = source_dir / name
        if not path.is_file():
            raise RuntimeError(f"tiny-overfit source artifact is missing: {path}")
        observed_hashes[name] = sha256_file(path)
        if observed_hashes[name] != expected:
            raise RuntimeError(
                f"tiny-overfit source hash mismatch for {name}: "
                f"expected {expected}, observed {observed_hashes[name]}"
            )

    manifest = read_json(source_dir / "manifest.json")
    expected_manifest = {
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "rendered_examples": 512,
        "rendered_sha256": EXPECTED_SOURCE_HASHES["train.jsonl"],
        "state_manifest_sha256": EXPECTED_SOURCE_HASHES["state_manifest.jsonl"],
        "universe_size": 128,
        "train_secret_count": 96,
        "dev_secret_count": 32,
        "labelled_episode_secrets": "training split only",
        "seed": 2026,
    }
    drift = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"tiny-overfit source manifest drifted: {json.dumps(drift, sort_keys=True)}")

    universe = [normalize_word(word) for word in read_json(source_dir / "universe.json")]
    train_secrets = [normalize_word(word) for word in read_json(source_dir / "train_secrets.json")]
    dev_secrets = [normalize_word(word) for word in read_json(source_dir / "dev_secrets.json")]
    if len(universe) != 128 or len(set(universe)) != len(universe) or not all(
        is_five_ascii_letters(word) for word in universe
    ):
        raise RuntimeError("tiny-overfit source universe is not 128 unique five-letter words")
    if len(train_secrets) != 96 or len(set(train_secrets)) != len(train_secrets):
        raise RuntimeError("tiny-overfit source train split is not 96 unique words")
    if len(dev_secrets) != 32 or len(set(dev_secrets)) != len(dev_secrets):
        raise RuntimeError("tiny-overfit source dev split is not 32 unique words")
    if set(train_secrets) & set(dev_secrets):
        raise RuntimeError("tiny-overfit source train and dev secret splits overlap")
    if set(train_secrets) | set(dev_secrets) != set(universe):
        raise RuntimeError("tiny-overfit source splits do not partition the declared universe")

    rows = read_jsonl(source_dir / "train.jsonl")
    state_manifest = read_jsonl(source_dir / "state_manifest.jsonl")
    if len(rows) != 512 or len(state_manifest) != len(rows):
        raise RuntimeError("tiny-overfit source must contain exactly 512 training rows and manifest rows")
    if len({str(row.get("example_id", "")) for row in rows}) != len(rows):
        raise RuntimeError("tiny-overfit source example_id values are not unique")

    projected_manifest: list[dict[str, Any]] = []
    train_set, dev_set, universe_set = set(train_secrets), set(dev_secrets), set(universe)
    feedback_count = 0
    for index, row in enumerate(rows):
        source = row.get("source_state")
        if not isinstance(source, dict):
            raise RuntimeError(f"source row {index} has no canonical source_state")
        history = _history_tuples(row)
        secret = normalize_word(source.get("secret_answer", ""))
        target = normalize_word(row.get("target_word", ""))
        state_id = str(row.get("state_id", ""))
        if source.get("split") != "common_train" or secret not in train_set or secret in dev_set:
            raise RuntimeError(f"source row {index} is not a training-split member")
        if source.get("state_id") != state_id:
            raise RuntimeError(f"source row {index} state identity mismatch")
        history_digest = sha256_text(canonical_json(history))[:16]
        if not state_id.startswith("common_train-") or not state_id.endswith(history_digest):
            raise RuntimeError(f"source row {index} state_id does not match its public history")
        turn = len(history) + 1
        if row.get("turn") != turn or source.get("turn") != turn:
            raise RuntimeError(f"source row {index} turn does not match its history")
        for guess, feedback in history:
            if score_wordle(secret, guess) != feedback:
                raise RuntimeError(f"source row {index} contains feedback that does not match its train secret")
            if feedback == "GGGGG":
                raise RuntimeError(f"source row {index} occurs after the training game was solved")
            feedback_count += 1
        posterior = posterior_candidates(history, universe)
        declared_posterior = row.get("posterior_size")
        facts_posterior = source.get("facts", {}).get("posterior_count")
        if len(posterior) != declared_posterior or len(posterior) != facts_posterior or secret not in posterior:
            raise RuntimeError(f"source row {index} posterior facts do not recompute")
        guesses = {guess for guess, _ in history}
        if (
            target not in universe_set
            or not is_five_ascii_letters(target)
            or target in guesses
            or target not in posterior
        ):
            raise RuntimeError(f"source row {index} target is not a legal posterior-consistent action")
        expected_completion = [{"role": "assistant", "content": f"Final answer: {target}"}]
        if row.get("completion") != expected_completion:
            raise RuntimeError(f"source row {index} completion does not encode its declared target")
        projected_manifest.append(
            {
                "example_id": row["example_id"],
                "posterior_size": row["posterior_size"],
                "state_id": row["state_id"],
                "state_type": row["state_type"],
                "target_frequency": row["target_frequency"],
                "target_word": row["target_word"],
                "turn": row["turn"],
            }
        )
    if state_manifest != projected_manifest:
        raise RuntimeError("tiny-overfit source state_manifest does not exactly describe train.jsonl")

    audit = {
        "status": "passed",
        "curriculum_id": manifest["curriculum_id"],
        "source_directory": EXPECTED_SOURCE_DIRECTORY,
        "hashes": observed_hashes,
        "training_rows": len(rows),
        "unique_source_states": len({row["state_id"] for row in rows}),
        "feedback_rows_recomputed": len(rows),
        "feedback_tiles_recomputed": feedback_count,
        "target_legality_rows_recomputed": len(rows),
        "posterior_rows_recomputed": len(rows),
        "train_dev_overlap": 0,
        "locked_test_access": False,
    }
    return rows, universe, audit


def audit_tiny_overfit_source(source_dir: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Public read-only audit for the exact balanced-002 training source."""

    _, _, audit = _source_contract(source_dir)
    return audit


def _feedback_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def select_contrast_pairs(rows: Sequence[dict[str, Any]], pair_count: int = 16) -> list[dict[str, Any]]:
    """Select disjoint, nearly identical turn-two states with different feedback/targets."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        history = _history(row)
        if len(history) == 1 and int(row.get("turn", 2)) == 2:
            unique.setdefault(str(row["state_id"]), row)
    candidates: list[tuple[int, str, str, dict[str, Any], dict[str, Any]]] = []
    for left, right in combinations(unique.values(), 2):
        left_history, right_history = _history(left), _history(right)
        if left_history[0]["guess"] != right_history[0]["guess"]:
            continue
        if left_history[0]["feedback"] == right_history[0]["feedback"]:
            continue
        if str(left["target_word"]).upper() == str(right["target_word"]).upper():
            continue
        distance = _feedback_distance(left_history[0]["feedback"], right_history[0]["feedback"])
        candidates.append((distance, str(left["state_id"]), str(right["state_id"]), left, right))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for distance, _, _, left, right in candidates:
        ids = {str(left["state_id"]), str(right["state_id"])}
        if ids & used:
            continue
        pair_id = f"contrast-{len(selected) // 2:02d}"
        for member, row in enumerate((left, right)):
            selected.append(_strip_training_row(row, "state_action", pair_id=pair_id, pair_member=member))
            selected[-1]["feedback_hamming_distance"] = distance
        used.update(ids)
        if len(selected) == pair_count * 2:
            break
    if len(selected) != pair_count * 2:
        raise RuntimeError(f"needed {pair_count} contrast pairs; found {len(selected) // 2}")
    return selected


def _strip_training_row(
    row: dict[str, Any],
    task: str,
    *,
    pair_id: str | None = None,
    pair_member: int | None = None,
) -> dict[str, Any]:
    target = str(row.get("target_word") or row["completion"][0]["content"].rsplit(":", 1)[-1]).strip().upper()
    clean = {
        "example_id": f"tiny-{task}-{row['state_id']}",
        "state_id": str(row["state_id"]),
        "split": "training_only_memorization",
        "task": task,
        "turn": int(row.get("turn", len(_history(row)) + 1)),
        "posterior_size": int(row.get("posterior_size", row.get("source_state", {}).get("facts", {}).get("posterior_count", 0))),
        "history": _history(row),
        "target_word": target,
        "prompt": row["prompt"],
        "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
    }
    if pair_id is not None:
        clean.update({"pair_id": pair_id, "pair_member": int(pair_member)})
    return clean


def select_singletons(rows: Sequence[dict[str, Any]], count: int = 32, seed: int = 2026) -> list[dict[str, Any]]:
    """Select target-balanced, unique, training-only posterior-size-one states."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        posterior_size = int(row.get("posterior_size", row.get("source_state", {}).get("facts", {}).get("posterior_count", 0)))
        if posterior_size == 1:
            unique.setdefault(str(row["state_id"]), row)
    ordered = sorted(
        unique.values(),
        key=lambda row: hashlib.sha256(f"{seed}:{row['state_id']}".encode("utf-8")).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    remaining = list(ordered)
    while remaining and len(selected) < count:
        remaining.sort(
            key=lambda row: (
                target_counts[str(row.get("target_word") or row["completion"][0]["content"]).upper()],
                hashlib.sha256(f"{seed}:{row['state_id']}".encode("utf-8")).hexdigest(),
            )
        )
        row = remaining.pop(0)
        clean = _strip_training_row(row, "singleton")
        target_counts[clean["target_word"]] += 1
        selected.append(clean)
    if len(selected) != count:
        raise RuntimeError(f"needed {count} singleton states; found {len(selected)}")
    return selected


def _audit_emitted_row(
    row: dict[str, Any],
    *,
    task: str,
    source_by_state: dict[str, list[dict[str, Any]]],
    universe: Sequence[str],
) -> list[str]:
    expected_fields = GENERAL_ROW_FIELDS if task == "state_action" else COMMON_ROW_FIELDS
    if set(row) != expected_fields:
        raise RuntimeError(
            f"tiny-overfit {task} row has unexpected fields: "
            f"missing={sorted(expected_fields - set(row))}, extra={sorted(set(row) - expected_fields)}"
        )
    state_id = str(row.get("state_id", ""))
    source_candidates = source_by_state.get(state_id, [])
    if not source_candidates:
        raise RuntimeError(f"tiny-overfit row {state_id!r} is not a member of the audited train source")
    if row.get("split") != "training_only_memorization" or row.get("task") != task:
        raise RuntimeError(f"tiny-overfit row {state_id} has the wrong split or task")
    if row.get("example_id") != f"tiny-{task}-{state_id}":
        raise RuntimeError(f"tiny-overfit row {state_id} has a noncanonical example_id")

    history = _history_tuples(row)
    source = source_candidates[0]["source_state"]
    secret = normalize_word(source["secret_answer"])
    for guess, feedback in history:
        if score_wordle(secret, guess) != feedback:
            raise RuntimeError(f"tiny-overfit row {state_id} feedback does not recompute from its train source")
    posterior = posterior_candidates(history, universe)
    target = normalize_word(row.get("target_word", ""))
    if row.get("turn") != len(history) + 1:
        raise RuntimeError(f"tiny-overfit row {state_id} turn does not match its history")
    if row.get("posterior_size") != len(posterior):
        raise RuntimeError(f"tiny-overfit row {state_id} posterior size does not recompute")
    if (
        target not in set(universe)
        or not is_five_ascii_letters(target)
        or target in {guess for guess, _ in history}
        or target not in posterior
    ):
        raise RuntimeError(f"tiny-overfit row {state_id} target is not a legal posterior-consistent action")
    if row.get("completion") != [{"role": "assistant", "content": f"Final answer: {target}"}]:
        raise RuntimeError(f"tiny-overfit row {state_id} completion does not encode its target")
    if task == "singleton" and posterior != [target]:
        raise RuntimeError(
            f"tiny-overfit singleton row {state_id} does not recompute to exactly its declared target"
        )
    exact_membership = any(
        _history_tuples(candidate) == history
        and normalize_word(candidate["target_word"]) == target
        and candidate["prompt"] == row["prompt"]
        for candidate in source_candidates
    )
    if not exact_membership:
        raise RuntimeError(f"tiny-overfit row {state_id} is not an exact prompt/action member of train.jsonl")
    return posterior


def audit_tiny_overfit_bundle(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Fail closed on source provenance and every emitted memorization row."""

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    source_rows, universe, source_audit = _source_contract(source_dir)
    manifest_path = output_dir / "manifest.json"
    general_path = output_dir / "general_32.jsonl"
    singleton_path = output_dir / "singleton_32.jsonl"
    for path in (manifest_path, general_path, singleton_path):
        if not path.is_file():
            raise RuntimeError(f"tiny-overfit bundle artifact is missing: {path}")
    manifest = read_json(manifest_path)
    general = read_jsonl(general_path)
    singletons = read_jsonl(singleton_path)
    if len(general) != 32 or len(singletons) != 32:
        raise RuntimeError("tiny-overfit bundle must contain exactly 32 rows in each cell")
    general_ids = {str(row.get("state_id", "")) for row in general}
    singleton_ids = {str(row.get("state_id", "")) for row in singletons}
    if len(general_ids) != 32 or len(singleton_ids) != 32:
        raise RuntimeError("tiny-overfit cells must contain unique state IDs")
    if general_ids & singleton_ids:
        raise RuntimeError("tiny-overfit general and singleton cells are not disjoint")

    source_by_state: dict[str, list[dict[str, Any]]] = {}
    for source_row in source_rows:
        source_by_state.setdefault(str(source_row["state_id"]), []).append(source_row)
    for row in general:
        _audit_emitted_row(
            row,
            task="state_action",
            source_by_state=source_by_state,
            universe=universe,
        )
    for row in singletons:
        _audit_emitted_row(
            row,
            task="singleton",
            source_by_state=source_by_state,
            universe=universe,
        )

    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in general:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    if len(pairs) != 16 or any(len(pair) != 2 for pair in pairs.values()):
        raise RuntimeError("tiny-overfit general cell must contain 16 complete contrast pairs")
    for pair_id, pair in pairs.items():
        left, right = sorted(pair, key=lambda row: int(row["pair_member"]))
        if [left["pair_member"], right["pair_member"]] != [0, 1]:
            raise RuntimeError(f"tiny-overfit contrast pair {pair_id} has invalid member indices")
        left_history, right_history = _history(left), _history(right)
        distance = _feedback_distance(left_history[0]["feedback"], right_history[0]["feedback"])
        if (
            left_history[0]["guess"] != right_history[0]["guess"]
            or left_history[0]["feedback"] == right_history[0]["feedback"]
            or left["target_word"] == right["target_word"]
            or left["feedback_hamming_distance"] != distance
            or right["feedback_hamming_distance"] != distance
        ):
            raise RuntimeError(f"tiny-overfit contrast pair {pair_id} does not recompute")

    expected_general = select_contrast_pairs(source_rows, 16)
    expected_general_ids = {row["state_id"] for row in expected_general}
    expected_singletons = select_singletons(
        [row for row in source_rows if str(row["state_id"]) not in expected_general_ids],
        32,
    )
    if general != expected_general or singletons != expected_singletons:
        raise RuntimeError("tiny-overfit emitted cells differ from the deterministic train-only selection")

    general_hash, singleton_hash = sha256_file(general_path), sha256_file(singleton_path)
    expected_manifest_fields = {
        "experiment_id": TINY_OVERFIT_ID,
        "protocol_id": "WORDLE-PROTOCOL-002",
        "purpose": "training-set memorization diagnostic; no generalization claim",
        "locked_test_access": False,
        "source_split": "common-curriculum-002 training only",
        "source_directory": EXPECTED_SOURCE_DIRECTORY,
        "source_rows_sha256": EXPECTED_SOURCE_HASHES["train.jsonl"],
        "source_manifest_sha256": EXPECTED_SOURCE_HASHES["manifest.json"],
        "source_declared_rows_sha256": EXPECTED_SOURCE_HASHES["train.jsonl"],
        "universe_sha256": EXPECTED_SOURCE_HASHES["universe.json"],
        "train_secrets_sha256": EXPECTED_SOURCE_HASHES["train_secrets.json"],
        "dev_secrets_sha256": EXPECTED_SOURCE_HASHES["dev_secrets.json"],
        "cells": {
            "general_32": {"rows": 32, "contrast_pairs": 16, "sha256": general_hash},
            "singleton_32": {
                "rows": 32,
                "posterior_size": 1,
                "unique_targets": len({row["target_word"] for row in singletons}),
                "sha256": singleton_hash,
            },
        },
        "row_schema": [
            "example_id",
            "state_id",
            "split",
            "task",
            "turn",
            "posterior_size",
            "history",
            "target_word",
            "prompt",
            "completion",
        ],
        "dataset_sha256": hashlib.sha256((general_hash + singleton_hash).encode("ascii")).hexdigest(),
    }
    if manifest != expected_manifest_fields:
        raise RuntimeError("tiny-overfit manifest does not exactly bind the audited source and emitted rows")
    return {
        "status": "passed",
        "experiment_id": TINY_OVERFIT_ID,
        "source": source_audit,
        "bundle_directory": str(output_dir),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "general_rows": len(general),
        "singleton_rows": len(singletons),
        "general_singleton_overlap": 0,
        "selected_feedback_rows_recomputed": len(general) + len(singletons),
        "selected_target_legality_rows_recomputed": len(general) + len(singletons),
        "singleton_posteriors_recomputed": len(singletons),
        "universe_sha256": EXPECTED_SOURCE_HASHES["universe.json"],
        "locked_test_access": False,
    }


def load_audited_cell(rows_path: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Load one declared cell and its universe through the bundle provenance."""

    rows_path = Path(rows_path).resolve()
    bundle_dir = rows_path.parent
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("tiny-overfit rows require their sibling audited manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("source_directory") != EXPECTED_SOURCE_DIRECTORY:
        raise RuntimeError("tiny-overfit manifest does not declare the frozen balanced-002 source")
    source_dir = (ROOT / manifest["source_directory"]).resolve()
    audit = audit_tiny_overfit_bundle(source_dir, bundle_dir)
    cell_by_file = {"general_32.jsonl": "general_32", "singleton_32.jsonl": "singleton_32"}
    cell = cell_by_file.get(rows_path.name)
    if cell is None or rows_path.parent != bundle_dir:
        raise RuntimeError("rows must be one of the two files declared by the tiny-overfit bundle")
    if sha256_file(rows_path) != manifest["cells"][cell]["sha256"]:
        raise RuntimeError("tiny-overfit rows hash does not match the bundle manifest")
    rows = read_jsonl(rows_path)
    universe = [normalize_word(word) for word in read_json(source_dir / "universe.json")]
    return rows, universe, {**audit, "cell": cell, "cell_sha256": sha256_file(rows_path)}


def build_tiny_overfit_bundle(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Create two deterministic 32-row memorization cells and their audit manifest."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        audit_tiny_overfit_bundle(source_dir, output_dir)
        return read_json(manifest_path)
    source_rows, _, _ = _source_contract(source_dir)
    general = select_contrast_pairs(source_rows, 16)
    general_ids = {row["state_id"] for row in general}
    singletons = select_singletons(
        [row for row in source_rows if str(row["state_id"]) not in general_ids],
        32,
    )
    general_path = write_jsonl(output_dir / "general_32.jsonl", general)
    singleton_path = write_jsonl(output_dir / "singleton_32.jsonl", singletons)
    source_manifest = read_json(source_dir / "manifest.json")
    if general_ids & set(row["state_id"] for row in singletons):
        raise AssertionError("general and singleton memorization cells must use distinct states")
    manifest = {
        "experiment_id": TINY_OVERFIT_ID,
        "protocol_id": "WORDLE-PROTOCOL-002",
        "purpose": "training-set memorization diagnostic; no generalization claim",
        "locked_test_access": False,
        "source_split": "common-curriculum-002 training only",
        "source_directory": EXPECTED_SOURCE_DIRECTORY,
        "source_rows_sha256": sha256_file(source_dir / "train.jsonl"),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "source_declared_rows_sha256": source_manifest["rendered_sha256"],
        "universe_sha256": sha256_file(source_dir / "universe.json"),
        "train_secrets_sha256": sha256_file(source_dir / "train_secrets.json"),
        "dev_secrets_sha256": sha256_file(source_dir / "dev_secrets.json"),
        "cells": {
            "general_32": {
                "rows": len(general),
                "contrast_pairs": len({row["pair_id"] for row in general}),
                "sha256": sha256_file(general_path),
            },
            "singleton_32": {
                "rows": len(singletons),
                "posterior_size": 1,
                "unique_targets": len({row["target_word"] for row in singletons}),
                "sha256": sha256_file(singleton_path),
            },
        },
        "row_schema": [
            "example_id",
            "state_id",
            "split",
            "task",
            "turn",
            "posterior_size",
            "history",
            "target_word",
            "prompt",
            "completion",
        ],
        "dataset_sha256": hashlib.sha256(
            (sha256_file(general_path) + sha256_file(singleton_path)).encode("ascii")
        ).hexdigest(),
    }
    write_json(manifest_path, manifest)
    audit_tiny_overfit_bundle(source_dir, output_dir)
    return manifest


def _generate_prompts(model, tokenizer, rows: Sequence[dict[str, Any]], batch_size: int = 16) -> list[str]:
    device = next(model.parameters()).device
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            inputs = tokenizer(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=64,
                    use_cache=True,
                    eos_token_id=stop_token_ids(tokenizer),
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = inputs["input_ids"].shape[1]
            outputs.extend(
                tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                for row in generated
            )
    finally:
        tokenizer.padding_side = previous_padding
    return outputs


def candidate_word_scores(
    model,
    tokenizer,
    prompt: list[dict[str, str]],
    candidates: Sequence[str],
    *,
    max_length: int = 320,
    batch_size: int = 16,
) -> dict[str, float]:
    """Score only action-word tokens and normalize over the declared universe."""
    candidate_rows = [
        {
            "example_id": f"rank-{word}",
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": f"Final answer: {word}"}],
            "target_word": word,
        }
        for word in candidates
    ]
    dataset = CompletionDataset(candidate_rows, tokenizer, max_length, word_token_weight=2.0)
    collator = Collator(tokenizer.pad_token_id)
    device = next(model.parameters()).device
    log_scores: list[float] = []
    model.eval()
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(len(dataset), start + batch_size))]
        batch = {key: value.to(device) for key, value in collator(samples).items()}
        weights = batch.pop("loss_weights")
        with torch.inference_mode():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        labels = batch["labels"][:, 1:]
        word_mask = weights[:, 1:].gt(1.0) & labels.ne(-100)
        safe_labels = labels.masked_fill(~word_mask, 0)
        observed = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        scores = (observed * word_mask).sum(-1)
        log_scores.extend(float(value) for value in scores.cpu())
        del logits, log_probs
    probabilities = torch.softmax(torch.tensor(log_scores, dtype=torch.float64), dim=0).tolist()
    return {
        str(word).upper(): float(probability)
        for word, probability in zip(candidates, probabilities, strict=True)
    }


def evaluate_memorization(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    universe: Sequence[str],
    *,
    rank_batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure natural exact recall plus target rank/probability on training states."""
    started = time.perf_counter()
    allowed = [str(word).upper() for word in universe]
    generated = _generate_prompts(model, tokenizer, rows)
    details: list[dict[str, Any]] = []
    for row, raw in zip(rows, generated, strict=True):
        target = str(row["target_word"]).upper()
        parsed = parse_terminal_answer(raw, allowed)
        probabilities = candidate_word_scores(
            model,
            tokenizer,
            row["prompt"],
            allowed,
            batch_size=rank_batch_size,
        )
        ranking = sorted(allowed, key=lambda word: (-probabilities[word], word))
        details.append(
            {
                "example_id": row["example_id"],
                "state_id": row["state_id"],
                "task": row["task"],
                "pair_id": row.get("pair_id"),
                "target_word": target,
                "raw_output": raw,
                "parse_status": parsed["status"],
                "parsed_guess": parsed["parsed_guess"],
                "exact_match": parsed["status"] == "ok" and parsed["parsed_guess"] == target,
                "target_rank": ranking.index(target) + 1,
                "target_probability": probabilities[target],
                "top_ranked_word": ranking[0],
                "top_ranked_probability": probabilities[ranking[0]],
            }
        )
    pairs: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        if detail.get("pair_id"):
            pairs.setdefault(detail["pair_id"], []).append(detail)
    ranks = [int(row["target_rank"]) for row in details]
    summary = {
        "experiment_id": TINY_OVERFIT_ID,
        "items": len(details),
        "task": details[0]["task"] if details else None,
        "natural_exact_accuracy": sum(row["exact_match"] for row in details) / len(details) if details else None,
        "terminal_compliance": sum(row["parse_status"] == "ok" for row in details) / len(details) if details else None,
        "mean_target_rank": statistics.mean(ranks) if ranks else None,
        "median_target_rank": statistics.median(ranks) if ranks else None,
        "mean_reciprocal_rank": statistics.mean(1.0 / rank for rank in ranks) if ranks else None,
        "mean_target_probability": statistics.mean(row["target_probability"] for row in details) if details else None,
        "top_1_rank_accuracy": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else None,
        "contrast_pair_exact_accuracy": (
            sum(len(pair) == 2 and all(item["exact_match"] for item in pair) for pair in pairs) / len(pairs)
            if pairs else None
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    return details, summary


def _validated_detail_map(
    rows: Sequence[dict[str, Any]],
    details: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    expected = {str(row["example_id"]): row for row in rows}
    observed: dict[str, dict[str, Any]] = {}
    for index, detail in enumerate(details):
        example_id = str(detail.get("example_id", ""))
        if not example_id or example_id in observed:
            raise RuntimeError(f"{label} detail row {index} has a missing or duplicate example_id")
        observed[example_id] = dict(detail)
    if set(observed) != set(expected):
        raise RuntimeError(f"{label} details do not cover exactly the audited memorization rows")
    for example_id, row in expected.items():
        detail = observed[example_id]
        identity = {
            "state_id": row["state_id"],
            "task": row["task"],
            "pair_id": row.get("pair_id"),
            "target_word": row["target_word"],
        }
        if any(detail.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"{label} detail identity mismatch for {example_id}")
        rank = detail.get("target_rank")
        probability = detail.get("target_probability")
        exact = detail.get("exact_match")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 128:
            raise RuntimeError(f"{label} target rank is invalid for {example_id}")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise RuntimeError(f"{label} target probability is invalid for {example_id}")
        if not isinstance(exact, bool):
            raise RuntimeError(f"{label} exact-match value is invalid for {example_id}")
    return observed


def compare_memorization_details(
    rows: Sequence[dict[str, Any]],
    base_details: Sequence[dict[str, Any]],
    adapter_details: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join identical-state base/adapter results and compute signed deltas."""

    base = _validated_detail_map(rows, base_details, "base")
    adapter = _validated_detail_map(rows, adapter_details, "adapter")
    joined: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row["example_id"])
        before, after = base[example_id], adapter[example_id]
        base_rank, adapter_rank = int(before["target_rank"]), int(after["target_rank"])
        base_probability = float(before["target_probability"])
        adapter_probability = float(after["target_probability"])
        base_exact, adapter_exact = bool(before["exact_match"]), bool(after["exact_match"])
        joined.append(
            {
                "example_id": example_id,
                "state_id": row["state_id"],
                "task": row["task"],
                "pair_id": row.get("pair_id"),
                "target_word": row["target_word"],
                "base_target_rank": base_rank,
                "adapter_target_rank": adapter_rank,
                "target_rank_delta": adapter_rank - base_rank,
                "target_rank_improvement": base_rank - adapter_rank,
                "base_target_probability": base_probability,
                "adapter_target_probability": adapter_probability,
                "target_probability_delta": adapter_probability - base_probability,
                "base_exact_match": base_exact,
                "adapter_exact_match": adapter_exact,
                "exact_match_delta": int(adapter_exact) - int(base_exact),
            }
        )

    def aggregate(prefix: str) -> dict[str, Any]:
        ranks = [int(row[f"{prefix}_target_rank"]) for row in joined]
        probabilities = [float(row[f"{prefix}_target_probability"]) for row in joined]
        exact = [bool(row[f"{prefix}_exact_match"]) for row in joined]
        return {
            "items": len(joined),
            "mean_target_rank": statistics.mean(ranks),
            "median_target_rank": statistics.median(ranks),
            "mean_reciprocal_rank": statistics.mean(1.0 / rank for rank in ranks),
            "mean_target_probability": statistics.mean(probabilities),
            "natural_exact_accuracy": sum(exact) / len(exact),
            "exact_match_count": sum(exact),
            "top_1_rank_accuracy": sum(rank == 1 for rank in ranks) / len(ranks),
        }

    if not joined:
        raise RuntimeError("memorization comparison requires at least one audited row")
    before_summary, after_summary = aggregate("base"), aggregate("adapter")
    summary = {
        "experiment_id": TINY_OVERFIT_ID,
        "comparison": "paired_base_vs_adapter_on_identical_training_states",
        "items": len(joined),
        "base": before_summary,
        "adapter": after_summary,
        "deltas": {
            "mean_target_rank": after_summary["mean_target_rank"] - before_summary["mean_target_rank"],
            "mean_target_rank_improvement": before_summary["mean_target_rank"] - after_summary["mean_target_rank"],
            "mean_reciprocal_rank": after_summary["mean_reciprocal_rank"] - before_summary["mean_reciprocal_rank"],
            "mean_target_probability": after_summary["mean_target_probability"] - before_summary["mean_target_probability"],
            "natural_exact_accuracy": after_summary["natural_exact_accuracy"] - before_summary["natural_exact_accuracy"],
            "exact_match_count": after_summary["exact_match_count"] - before_summary["exact_match_count"],
            "top_1_rank_accuracy": after_summary["top_1_rank_accuracy"] - before_summary["top_1_rank_accuracy"],
        },
        "state_changes": {
            "rank_improved": sum(row["target_rank_improvement"] > 0 for row in joined),
            "rank_unchanged": sum(row["target_rank_improvement"] == 0 for row in joined),
            "rank_worsened": sum(row["target_rank_improvement"] < 0 for row in joined),
            "exact_gained": sum(row["exact_match_delta"] == 1 for row in joined),
            "exact_lost": sum(row["exact_match_delta"] == -1 for row in joined),
        },
        "locked_test_access": False,
    }
    return joined, summary


def evaluate_checkpoint(
    rows_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    rank_batch_size: int = 16,
    evaluation_label: str | None = None,
) -> dict[str, Any]:
    rows_path = Path(rows_path).resolve()
    rows, universe, audit = load_audited_cell(rows_path)
    if checkpoint is None:
        tokenizer = load_tokenizer()
        model = load_base_model(training=False)
        label = evaluation_label or "base"
    else:
        checkpoint = Path(checkpoint).resolve()
        tokenizer = load_tokenizer(checkpoint)
        model = load_adapter(checkpoint)
        label = evaluation_label or checkpoint.name
    if not label or not label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("evaluation label must contain only letters, numbers, hyphens, or underscores")
    try:
        details, summary = evaluate_memorization(model, tokenizer, rows, universe, rank_batch_size=rank_batch_size)
        output_dir = Path(output_dir)
        items_path = write_jsonl(output_dir / f"{label}-items.jsonl", details)
        summary.update(
            {
                "evaluation_label": label,
                "checkpoint": "base" if checkpoint is None else str(checkpoint),
                "rows_sha256": sha256_file(rows_path),
                "bundle_manifest_sha256": audit["bundle_manifest_sha256"],
                "universe_sha256": audit["universe_sha256"],
                "items_sha256": sha256_file(items_path),
                "locked_test_access": False,
            }
        )
        write_json(output_dir / f"{label}-summary.json", summary)
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def compare_checkpoints(
    rows_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    base_items_path: Path | None = None,
    adapter_items_path: Path | None = None,
    rank_batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate or consume base/adapter outputs, then persist a paired comparison."""

    rows_path, output_dir = Path(rows_path).resolve(), Path(output_dir).resolve()
    rows, _, audit = load_audited_cell(rows_path)
    consume = base_items_path is not None or adapter_items_path is not None
    if checkpoint is not None and consume:
        raise ValueError("compare accepts either --checkpoint or precomputed item files, not both")
    if checkpoint is None and not (base_items_path is not None and adapter_items_path is not None):
        raise ValueError("compare requires --checkpoint or both --base-items and --adapter-items")
    if checkpoint is not None:
        evaluation_dir = output_dir / "evaluations"
        evaluate_checkpoint(
            rows_path,
            evaluation_dir,
            checkpoint=None,
            rank_batch_size=rank_batch_size,
            evaluation_label="base",
        )
        evaluate_checkpoint(
            rows_path,
            evaluation_dir,
            checkpoint=checkpoint,
            rank_batch_size=rank_batch_size,
            evaluation_label="adapter",
        )
        base_items_path = evaluation_dir / "base-items.jsonl"
        adapter_items_path = evaluation_dir / "adapter-items.jsonl"
        mode = "evaluated_base_and_adapter"
    else:
        base_items_path = Path(base_items_path).resolve()
        adapter_items_path = Path(adapter_items_path).resolve()
        mode = "consumed_precomputed_items"
    if not base_items_path.is_file() or not adapter_items_path.is_file():
        raise FileNotFoundError("base and adapter item files must both exist")
    base_details, adapter_details = read_jsonl(base_items_path), read_jsonl(adapter_items_path)
    joined, summary = compare_memorization_details(rows, base_details, adapter_details)
    joined_path = write_jsonl(output_dir / "pre_post_items.jsonl", joined)
    summary.update(
        {
            "mode": mode,
            "adapter_checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
            "base_items_path": str(base_items_path),
            "adapter_items_path": str(adapter_items_path),
            "rows_sha256": sha256_file(rows_path),
            "bundle_manifest_sha256": audit["bundle_manifest_sha256"],
            "universe_sha256": audit["universe_sha256"],
            "base_items_sha256": sha256_file(base_items_path),
            "adapter_items_sha256": sha256_file(adapter_items_path),
            "joined_items_sha256": sha256_file(joined_path),
        }
    )
    summary_path = write_json(output_dir / "pre_post_summary.json", summary)
    return {
        **summary,
        "joined_items_path": str(joined_path),
        "summary_path": str(summary_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or evaluate the 32-state Gemma memorization suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--force", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    audit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--rows", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--rank-batch-size", type=int, default=16)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--rows", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--checkpoint", type=Path)
    compare_parser.add_argument("--base-items", type=Path)
    compare_parser.add_argument("--adapter-items", type=Path)
    compare_parser.add_argument("--rank-batch-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_tiny_overfit_bundle(args.source_dir, args.output_dir, force=args.force)
    elif args.command == "audit":
        result = audit_tiny_overfit_bundle(args.source_dir, args.output_dir)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.rows,
            args.output_dir,
            checkpoint=args.checkpoint,
            rank_batch_size=args.rank_batch_size,
        )
    else:
        try:
            result = compare_checkpoints(
                args.rows,
                args.output_dir,
                checkpoint=args.checkpoint,
                base_items_path=args.base_items,
                adapter_items_path=args.adapter_items,
                rank_batch_size=args.rank_batch_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
