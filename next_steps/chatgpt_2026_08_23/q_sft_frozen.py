from __future__ import annotations

"""Frozen-target Q-SFT continuation for the historical balanced-002 parent.

The offline target is a conservative, bounded Wordle value recurrence.  For a
state with ``P`` possible answers and ``k`` valid guesses remaining:

``V_0(P) = 0``
``V_k(P) = 1/P + (1 - 1/P) * gamma * V_(k-1)(P)``

This is the discounted probability of eventually solving under a deliberately
weak surrogate that gives each future attempt probability ``1/P`` and assumes a
miss does not shrink the state.  It is in [0, 1], increases as the posterior
shrinks, and increases with additional remaining guesses.  The snapshot builder
uses only the training row's public ``posterior_size`` and ``turn`` fields.  It
never serializes a secret, candidate list, oracle/evaluator field, or answer set.

Building and auditing that train-only bundle is independent of parent
eligibility.  Dry-run and training additionally require hash-pinned development
evidence to meet the declared legality thresholds; otherwise they return the
explicit ``blocked_prerequisite_legality_gate_failed`` status before training.
"""

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
DEFAULT_CONFIG = MODULE_DIR / "q_sft_frozen_config.json"
DEFAULT_BUNDLE = MODULE_DIR / "generated" / "q_sft_frozen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics  # noqa: E402
from wordle_lab.common import (  # noqa: E402
    ARTIFACTS,
    DATA,
    canonical_json,
    git_commit,
    read_json,
    read_jsonl,
    set_seed,
    sha256_file,
    sha256_text,
    source_tree_sha256,
    write_json,
    write_jsonl,
)
from wordle_lab.data.canonical import generate_canonical_states  # noqa: E402
from wordle_lab.methods.q_sft import train_q_sft, validate_q_sft_rows  # noqa: E402
from wordle_lab.models import (  # noqa: E402
    SUPPORTED_MODEL_ID,
    SUPPORTED_REVISION,
    load_adapter,
    load_tokenizer,
)
from wordle_lab.protocol.evaluator import evaluate  # noqa: E402
from wordle_lab.protocol.retention import evaluate_retention  # noqa: E402

from next_steps.chatgpt_2026_08_23.experiment_guardrails import (  # noqa: E402
    GuardrailViolation,
    assert_locked_test_closed,
    assert_protocol_lock_unchanged,
    build_artifact_manifest,
    normalize_gate_metrics,
    verify_artifact_manifest,
)


EXPERIMENT_ID = "QSFT-BALANCED-002-FROZEN-001"
TARGET_ID = "WORDLE-CONSERVATIVE-BELLMAN-001"
CURRICULUM_ID = "COMMON-WORD-CURRICULUM-002"
BEHAVIOR_POLICY_ID = "BALANCED-002-EMPIRICAL-UNIFORM-001"
SNAPSHOT_FIELDS = {
    "comparison_id",
    "source_state_id",
    "behavior_policy_id",
    "behavior_action",
    "behavior_probability",
    "behavior_support_size",
    "behavior_support_sha256",
    "posterior_size",
    "turn",
    "bellman_target",
}
JOINED_FIELDS = SNAPSHOT_FIELDS | {"example_id", "prompt", "completion"}
BEHAVIOR_PROVENANCE_CONTRACT = {
    "policy_id": BEHAVIOR_POLICY_ID,
    "source_state_id": "top-level state_id exactly equal to source_state.state_id",
    "behavior_action": "top-level target_word exactly equal to the completion final answer",
    "behavior_support": "distinct behavior_action values grouped by source_state_id",
    "behavior_probability": "1 / behavior_support_size",
    "support_commitment": "sha256(canonical_json(sorted(behavior_support)))",
    "uniform": True,
}
FORBIDDEN_EMITTED_FIELDS = {
    "allowed_words",
    "answer",
    "candidate",
    "candidate_words",
    "candidates",
    "evaluator",
    "locked_test_answer",
    "oracle",
    "posterior_candidates",
    "secret",
    "secret_answer",
    "source_state",
    "test_answer",
}
Q_SFT_REQUIRED_ARTIFACTS = (
    "spec.json",
    "dataset_manifest.json",
    "train_metrics.jsonl",
    "accounting.json",
    "summary.json",
    "games.jsonl",
    "retention.jsonl",
    "gate_metrics.json",
)
PREREQUISITE_THRESHOLDS = {
    "terminal_compliance_minimum": 0.99,
    "turn_2_posterior_violation_maximum_exclusive": 0.30,
    "singleton_answer_accuracy_minimum": 0.80,
}
PARENT_DEV_EVIDENCE = {
    "summary": {
        "path": "summary.json",
        "sha256": "5501c697996717e9a67be75e90f1ee57dbaefa90a29899b733bbdd8f0d093b9d",
    },
    "games": {
        "path": "games.jsonl",
        "sha256": "f0df429c6af2a80799dedb3abdadeb64672e392bc610aeb7db6ddff198f8defc",
    },
    "diagnostic_summary": {
        "path": "diagnostics/7b309ade0477/summary.json",
        "sha256": "05079bbeed9be50efc4f48e4acc3b68756f1ff64146808dbba20ac1e0896c859",
    },
    "diagnostic_items": {
        "path": "diagnostics/7b309ade0477/items.jsonl",
        "sha256": "a0204ff22d4376aedb8519d0c9b66c72e2e5d2ff55a4f51c984584c1863f8593",
    },
}
BLOCKED_PREREQUISITE_STATUS = "blocked_prerequisite_legality_gate_failed"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(canonical_json(dict(row)) for row in rows)
    return hashlib.sha256((payload + ("\n" if payload else "")).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise GuardrailViolation(f"{label} must be a SHA-256 digest")
    return normalized


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if the frozen Q-SFT experiment contract drifts."""

    exact = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": "WORDLE-PROTOCOL-002",
        "protocol_sha256": "afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a",
        "model": {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION},
        "method": "q_sft",
        "objective": "bellman_likelihood_uniform_wce",
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    mismatches = {key: {"expected": expected, "actual": config.get(key)} for key, expected in exact.items() if config.get(key) != expected}
    target = config.get("target", {})
    target_exact = {
        "target_id": TARGET_ID,
        "formula": "V0=0; Vk=1/P+(1-1/P)*gamma*V(k-1)",
        "gamma": 0.99,
        "universe_size": 128,
        "max_valid_guesses": 6,
        "inputs": ["posterior_size", "turn"],
    }
    if target != target_exact:
        mismatches["target"] = {"expected": target_exact, "actual": target}
    behavior_provenance = config.get("behavior_provenance", {})
    if behavior_provenance != BEHAVIOR_PROVENANCE_CONTRACT:
        mismatches["behavior_provenance"] = {
            "expected": BEHAVIOR_PROVENANCE_CONTRACT,
            "actual": behavior_provenance,
        }
    data = config.get("data", {})
    data_exact = {
        "curriculum_id": CURRICULUM_ID,
        "directory": "data/common-curriculum-002/u128-train96",
        "training_file": "train.jsonl",
        "training_rows": 512,
        "source_manifest_sha256": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
        "source_training_sha256": "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f",
        "state_manifest_sha256": "4ab23b5cd883d8ad9b542befadc23c2aec3a3d631b78f239bb551ca998fd6a3c",
    }
    if data != data_exact:
        mismatches["data"] = {"expected": data_exact, "actual": data}
    warm = config.get("warm_start", {})
    warm_required = {
        "required": True,
        "expected_parent_run_id": "sft-common-balanced-word-s2026-0649b4deeb",
        "expected_checkpoint": "final",
        "expected_parent_method": "sft",
        "expected_parent_curriculum_id": CURRICULUM_ID,
        "expected_parent_seed": 2026,
        "expected_parent_word_token_weight": 8.0,
        "expected_parent_spec_sha256": "655972a100f33ca26d0e7834f602de9856f88aed89107aed67e6426f9d8c95bc",
        "expected_parent_dataset_manifest_sha256": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
        "expected_parent_adapter_tree_sha256": "074f3a7fe657e34a50cb67f1bf121d61a7c5d5978234c2cc203604ba20e8b833",
    }
    for key, expected in warm_required.items():
        if warm.get(key) != expected:
            mismatches[f"warm_start.{key}"] = {"expected": expected, "actual": warm.get(key)}
    _require_sha256(warm.get("expected_parent_adapter_tree_sha256"), "warm_start.expected_parent_adapter_tree_sha256")
    training = config.get("training", {})
    training_exact = {
        "seed": 2026,
        "max_steps": 100,
        "learning_rate": 0.00005,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "discount": 0.99,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
    }
    if training != training_exact:
        mismatches["training"] = {"expected": training_exact, "actual": training}
    evaluation = config.get("evaluation", {})
    evaluation_exact = {
        "split": "dev",
        "dev_games": 32,
        "diagnostic_items": 128,
        "retention_probes": "data/protocol-002/retention_probes_v1.jsonl",
        "natural_generation": True,
        "prompt_version": "reasoning-envelope-v1",
        "decoder": "frozen_greedy",
    }
    if evaluation != evaluation_exact:
        mismatches["evaluation"] = {"expected": evaluation_exact, "actual": evaluation}
    if mismatches:
        raise GuardrailViolation(f"frozen Q-SFT configuration drifted: {json.dumps(mismatches, sort_keys=True)}")
    assert_locked_test_closed(config, require_explicit=True)
    return dict(config)


def conservative_bellman_target(
    posterior_size: int,
    turn: int,
    *,
    gamma: float = 0.99,
    max_valid_guesses: int = 6,
) -> float:
    """Return the monotone bounded recurrence documented in this module."""

    if isinstance(posterior_size, bool) or not isinstance(posterior_size, int) or posterior_size < 1:
        raise ValueError("posterior_size must be a positive integer")
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= max_valid_guesses:
        raise ValueError("turn must be between one and max_valid_guesses")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    immediate = 1.0 / posterior_size
    value = 0.0
    remaining = max_valid_guesses - turn + 1
    for _ in range(remaining):
        value = immediate + (1.0 - immediate) * float(gamma) * value
    if not 0.0 <= value <= 1.0:
        raise AssertionError("bounded Bellman recurrence escaped [0, 1]")
    return round(value, 12)


def audit_target_contract(
    *,
    gamma: float = 0.99,
    universe_size: int = 128,
    max_valid_guesses: int = 6,
) -> dict[str, Any]:
    """Exhaustively verify boundedness and both monotonicity directions."""

    grid = {
        (posterior_size, turn): conservative_bellman_target(
            posterior_size,
            turn,
            gamma=gamma,
            max_valid_guesses=max_valid_guesses,
        )
        for posterior_size in range(1, universe_size + 1)
        for turn in range(1, max_valid_guesses + 1)
    }
    bounded = all(0.0 <= value <= 1.0 for value in grid.values())
    posterior_monotone = all(
        grid[(posterior_size, turn)] >= grid[(posterior_size + 1, turn)]
        for posterior_size in range(1, universe_size)
        for turn in range(1, max_valid_guesses + 1)
    )
    remaining_turns_monotone = all(
        grid[(posterior_size, turn)] >= grid[(posterior_size, turn + 1)]
        for posterior_size in range(1, universe_size + 1)
        for turn in range(1, max_valid_guesses)
    )
    if not bounded or not posterior_monotone or not remaining_turns_monotone:
        raise GuardrailViolation("frozen Bellman target failed its bounded monotonicity contract")
    return {
        "status": "passed",
        "states_checked": len(grid),
        "bounded_0_1": bounded,
        "nonincreasing_with_posterior_size": posterior_monotone,
        "nondecreasing_with_guesses_remaining": remaining_turns_monotone,
        "minimum": min(grid.values()),
        "maximum": max(grid.values()),
    }


def _assert_no_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if str(key).lower() in FORBIDDEN_EMITTED_FIELDS)
        if forbidden:
            raise GuardrailViolation(f"forbidden emitted fields at {path}: {forbidden}")
        for key, nested in value.items():
            _assert_no_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_no_forbidden_keys(nested, f"{path}[{index}]")


_WORDLE_ACTION = re.compile(r"[A-Z]{5}")


def _behavior_action_from_source(row: Mapping[str, Any], index: int) -> str:
    action = row.get("target_word")
    if not isinstance(action, str) or not _WORDLE_ACTION.fullmatch(action):
        raise GuardrailViolation(f"balanced-002 row {index} has an invalid behavior action")
    nonempty_lines = [
        line.strip()
        for message in row["completion"]
        for line in message["content"].splitlines()
        if line.strip()
    ]
    if not nonempty_lines:
        raise GuardrailViolation(f"balanced-002 row {index} has an empty completion")
    match = re.fullmatch(r"Final answer:\s*([A-Za-z]{5})", nonempty_lines[-1], flags=re.IGNORECASE)
    if match is None or match.group(1).upper() != action:
        raise GuardrailViolation(f"balanced-002 row {index} behavior action/completion mismatch")
    return action


def _behavior_support_sha256(actions: Sequence[str]) -> str:
    return sha256_text(canonical_json(sorted(actions)))


def _validate_training_source_row(row: Mapping[str, Any], index: int, universe_size: int) -> None:
    required = {
        "example_id",
        "state_id",
        "target_word",
        "posterior_size",
        "turn",
        "prompt",
        "completion",
        "source_state",
    }
    missing = sorted(required - set(row))
    if missing:
        raise GuardrailViolation(f"balanced-002 row {index} is missing fields: {missing}")
    source_state = row["source_state"]
    if not isinstance(source_state, Mapping) or source_state.get("split") != "common_train":
        raise GuardrailViolation(f"balanced-002 row {index} is not training-only")
    state_id = row["state_id"]
    if not isinstance(state_id, str) or not state_id or source_state.get("state_id") != state_id:
        raise GuardrailViolation(f"balanced-002 row {index} has unstable source_state_id provenance")
    posterior_size = row["posterior_size"]
    if isinstance(posterior_size, bool) or not isinstance(posterior_size, int) or not 1 <= posterior_size <= universe_size:
        raise GuardrailViolation(f"balanced-002 row {index} has invalid posterior_size")
    turn = row["turn"]
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= 6 or source_state.get("turn") != turn:
        raise GuardrailViolation(f"balanced-002 row {index} has invalid or inconsistent turn")
    if not isinstance(row["prompt"], list) or not row["prompt"] or not isinstance(row["completion"], list) or not row["completion"]:
        raise GuardrailViolation(f"balanced-002 row {index} has an invalid prompt/completion envelope")
    for message in row["prompt"] + row["completion"]:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise GuardrailViolation(f"balanced-002 row {index} has an invalid chat message")
    if any(message["role"] != "assistant" for message in row["completion"]):
        raise GuardrailViolation(f"balanced-002 row {index} completion must be from the assistant")
    _behavior_action_from_source(row, index)


def _source_behavior_supports(by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    actions_by_state: defaultdict[str, set[str]] = defaultdict(set)
    for index, comparison_id in enumerate(sorted(by_id)):
        row = by_id[comparison_id]
        state_id = str(row["state_id"])
        action = _behavior_action_from_source(row, index)
        actions_by_state[state_id].add(action)
    return {state_id: sorted(actions) for state_id, actions in sorted(actions_by_state.items())}


def build_frozen_snapshots(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    gamma: float = 0.99,
    universe_size: int = 128,
    max_valid_guesses: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic, provenance-complete snapshots from training rows."""

    rows = [dict(row) for row in source_rows]
    if not rows:
        raise GuardrailViolation("balanced-002 source rows are empty")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _validate_training_source_row(row, index, universe_size)
        comparison_id = str(row["example_id"])
        if comparison_id in by_id:
            raise GuardrailViolation(f"duplicate balanced-002 example_id: {comparison_id}")
        by_id[comparison_id] = row
    behavior_supports = _source_behavior_supports(by_id)
    snapshots = [
        {
            "comparison_id": comparison_id,
            "source_state_id": str(row["state_id"]),
            "behavior_policy_id": BEHAVIOR_POLICY_ID,
            "behavior_action": str(row["target_word"]),
            "behavior_probability": round(1.0 / len(behavior_supports[str(row["state_id"])]), 12),
            "behavior_support_size": len(behavior_supports[str(row["state_id"])]),
            "behavior_support_sha256": _behavior_support_sha256(
                behavior_supports[str(row["state_id"])]
            ),
            "posterior_size": int(row["posterior_size"]),
            "turn": int(row["turn"]),
            "bellman_target": conservative_bellman_target(
                int(row["posterior_size"]),
                int(row["turn"]),
                gamma=gamma,
                max_valid_guesses=max_valid_guesses,
            ),
        }
        for comparison_id, row in sorted(by_id.items())
    ]
    validate_snapshot_rows(snapshots)
    target_values = [row["bellman_target"] for row in snapshots]
    target_contract = audit_target_contract(
        gamma=gamma,
        universe_size=universe_size,
        max_valid_guesses=max_valid_guesses,
    )
    public_envelope = [
        {
            "comparison_id": comparison_id,
            "source_state_id": by_id[comparison_id]["state_id"],
            "behavior_action": by_id[comparison_id]["target_word"],
            "posterior_size": by_id[comparison_id]["posterior_size"],
            "turn": by_id[comparison_id]["turn"],
            "prompt": by_id[comparison_id]["prompt"],
            "completion": by_id[comparison_id]["completion"],
        }
        for comparison_id in sorted(by_id)
    ]
    audit = {
        "status": "passed",
        "target_id": TARGET_ID,
        "formula": "V0=0; Vk=1/P+(1-1/P)*gamma*V(k-1)",
        "gamma": gamma,
        "bounds": [min(target_values), max(target_values)],
        "contract_audit": target_contract,
        "rows": len(snapshots),
        "snapshot_rows_sha256": _jsonl_sha256(snapshots),
        "public_source_envelope_sha256": _jsonl_sha256(public_envelope),
        "behavior_provenance": _behavior_provenance_audit(snapshots),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "checks": [
            "training_split_only",
            "unique_comparison_ids",
            "stable_source_state_ids",
            "uniform_behavior_probabilities",
            "behavior_support_commitments_recomputed",
            "target_inputs_preserved_and_recomputed",
            "bounded_targets",
            "deterministic_sorted_output",
            "no_secret_candidate_or_evaluator_fields",
        ],
        "locked_test_access": False,
    }
    return snapshots, audit


def validate_snapshot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float = 0.99,
    universe_size: int = 128,
    max_valid_guesses: int = 6,
) -> None:
    if not rows:
        raise GuardrailViolation("snapshot rows are empty")
    seen = set()
    by_state: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if set(row) != SNAPSHOT_FIELDS:
            raise GuardrailViolation(f"snapshot {index} must contain exactly {sorted(SNAPSHOT_FIELDS)}")
        comparison_id = row["comparison_id"]
        if not isinstance(comparison_id, str) or not comparison_id or comparison_id in seen:
            raise GuardrailViolation(f"snapshot {index} has an invalid or duplicate comparison_id")
        seen.add(comparison_id)
        source_state_id = row["source_state_id"]
        if not isinstance(source_state_id, str) or not source_state_id:
            raise GuardrailViolation(f"snapshot {index} has an invalid source_state_id")
        if row["behavior_policy_id"] != BEHAVIOR_POLICY_ID:
            raise GuardrailViolation(f"snapshot {index} has an invalid behavior_policy_id")
        action = row["behavior_action"]
        if not isinstance(action, str) or not _WORDLE_ACTION.fullmatch(action):
            raise GuardrailViolation(f"snapshot {index} has an invalid behavior_action")
        support_size = row["behavior_support_size"]
        if isinstance(support_size, bool) or not isinstance(support_size, int) or support_size < 1:
            raise GuardrailViolation(f"snapshot {index} has an invalid behavior_support_size")
        probability = row["behavior_probability"]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 < float(probability) <= 1.0
        ):
            raise GuardrailViolation(f"snapshot {index} has an invalid behavior_probability")
        _require_sha256(row["behavior_support_sha256"], f"snapshot {index} behavior_support_sha256")
        posterior_size = row["posterior_size"]
        if (
            isinstance(posterior_size, bool)
            or not isinstance(posterior_size, int)
            or not 1 <= posterior_size <= universe_size
        ):
            raise GuardrailViolation(f"snapshot {index} has an invalid posterior_size")
        turn = row["turn"]
        if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= max_valid_guesses:
            raise GuardrailViolation(f"snapshot {index} has an invalid turn")
        target = row["bellman_target"]
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(float(target)) or not 0 <= float(target) <= 1:
            raise GuardrailViolation(f"snapshot {index} has an invalid bellman_target")
        expected_target = conservative_bellman_target(
            posterior_size,
            turn,
            gamma=gamma,
            max_valid_guesses=max_valid_guesses,
        )
        if float(target) != expected_target:
            raise GuardrailViolation(f"snapshot {index} bellman_target does not match its declared inputs")
        _assert_no_forbidden_keys(row)
        by_state[source_state_id].append(row)

    for source_state_id, state_rows in by_state.items():
        actions = sorted({str(row["behavior_action"]) for row in state_rows})
        expected_size = len(actions)
        expected_probability = round(1.0 / expected_size, 12)
        expected_support_sha256 = _behavior_support_sha256(actions)
        if len({(row["posterior_size"], row["turn"]) for row in state_rows}) != 1:
            raise GuardrailViolation(f"source_state_id {source_state_id} has inconsistent target inputs")
        for row in state_rows:
            if row["behavior_support_size"] != expected_size:
                raise GuardrailViolation(f"source_state_id {source_state_id} behavior support size mismatch")
            if row["behavior_support_sha256"] != expected_support_sha256:
                raise GuardrailViolation(f"source_state_id {source_state_id} behavior support hash mismatch")
            if float(row["behavior_probability"]) != expected_probability:
                raise GuardrailViolation(f"source_state_id {source_state_id} behavior probability is not uniform")
        probability_by_action = {
            action: float(next(row for row in state_rows if row["behavior_action"] == action)["behavior_probability"])
            for action in actions
        }
        if not math.isclose(
            sum(probability_by_action.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise GuardrailViolation(f"source_state_id {source_state_id} behavior probabilities do not sum to one")


def _behavior_provenance_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_snapshot_rows(rows)
    by_state: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["source_state_id"])].append(row)
    state_commitments = [
        {
            "source_state_id": source_state_id,
            "behavior_support_size": state_rows[0]["behavior_support_size"],
            "behavior_support_sha256": state_rows[0]["behavior_support_sha256"],
            "behavior_probability": state_rows[0]["behavior_probability"],
        }
        for source_state_id, state_rows in sorted(by_state.items())
    ]
    support_sizes = [int(row["behavior_support_size"]) for row in state_commitments]
    probabilities = [float(row["behavior_probability"]) for row in rows]
    return {
        "status": "passed",
        "policy_id": BEHAVIOR_POLICY_ID,
        "source_states": len(state_commitments),
        "snapshot_rows": len(rows),
        "behavior_actions": sum(
            len({str(row["behavior_action"]) for row in state_rows})
            for state_rows in by_state.values()
        ),
        "unique_behavior_actions": len({str(row["behavior_action"]) for row in rows}),
        "support_size_bounds": [min(support_sizes), max(support_sizes)],
        "support_size_distribution": dict(sorted(Counter(str(size) for size in support_sizes).items())),
        "probability_bounds": [min(probabilities), max(probabilities)],
        "state_supports_sha256": _jsonl_sha256(state_commitments),
        "checks": [
            "duplicate_samples_share_state_action_metadata",
            "uniform_probability_equals_reciprocal_support_size",
            "probabilities_sum_to_one_per_state",
            "support_hashes_recomputed_from_snapshot_actions",
        ],
        "locked_test_access": False,
    }


def join_snapshots_to_training_rows(
    source_rows: Iterable[Mapping[str, Any]], snapshots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join targets and sanitized behavior provenance while dropping private state."""

    validate_snapshot_rows(snapshots)
    snapshot_by_id = {row["comparison_id"]: row for row in snapshots}
    source_by_id: dict[str, Mapping[str, Any]] = {}
    materialized_source = [dict(row) for row in source_rows]
    for row in materialized_source:
        comparison_id = str(row.get("example_id", ""))
        if not comparison_id or comparison_id in source_by_id:
            raise GuardrailViolation("source rows require unique non-empty example_id values")
        source_by_id[comparison_id] = row
    if set(source_by_id) != set(snapshot_by_id):
        raise GuardrailViolation("snapshot/source comparison_id sets do not match exactly")
    expected_snapshots, _ = build_frozen_snapshots(materialized_source)
    if {row["comparison_id"]: row for row in expected_snapshots} != snapshot_by_id:
        raise GuardrailViolation("snapshot behavior provenance or target differs from deterministic source rebuild")
    joined = [
        {
            "example_id": comparison_id,
            **dict(snapshot_by_id[comparison_id]),
            "prompt": source_by_id[comparison_id]["prompt"],
            "completion": source_by_id[comparison_id]["completion"],
        }
        for comparison_id in sorted(source_by_id)
    ]
    for index, row in enumerate(joined):
        if set(row) != JOINED_FIELDS:
            raise GuardrailViolation(f"joined row {index} must contain exactly {sorted(JOINED_FIELDS)}")
        _assert_no_forbidden_keys(row)
    validate_q_sft_rows(joined, discount=0.99)
    return joined


def _source_context(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    data = config["data"]
    directory = _resolve(data["directory"])
    source_path = directory / data["training_file"]
    manifest_path = directory / "manifest.json"
    if sha256_file(source_path) != data["source_training_sha256"]:
        raise GuardrailViolation("balanced-002 training file hash mismatch")
    if sha256_file(manifest_path) != data["source_manifest_sha256"]:
        raise GuardrailViolation("balanced-002 source manifest hash mismatch")
    if sha256_file(directory / "state_manifest.jsonl") != data["state_manifest_sha256"]:
        raise GuardrailViolation("balanced-002 state manifest hash mismatch")
    source_manifest = read_json(manifest_path)
    if source_manifest.get("curriculum_id") != CURRICULUM_ID or source_manifest.get("rendered_examples") != data["training_rows"]:
        raise GuardrailViolation("balanced-002 source manifest identity/scale mismatch")
    if source_manifest.get("rendered_sha256") != data["source_training_sha256"]:
        raise GuardrailViolation("balanced-002 source manifest training hash mismatch")
    rows = read_jsonl(source_path)
    if len(rows) != data["training_rows"]:
        raise GuardrailViolation("balanced-002 source row count mismatch")
    return directory, rows, source_manifest


def _tree_sha256(directory: Path) -> tuple[str, dict[str, str]]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise GuardrailViolation(f"adapter directory contains no files: {directory}")
    hashes = {path.relative_to(directory).as_posix(): sha256_file(path) for path in files}
    return sha256_text(canonical_json(hashes)), hashes


def validate_parent_adapter(parent_adapter: str | Path, warm_start: Mapping[str, Any]) -> dict[str, Any]:
    """Require the explicit, exact Gemma-only balanced-002 SFT parent."""

    if parent_adapter is None:
        raise GuardrailViolation("--parent-adapter is required; Q-SFT may not start from an implicit parent")
    adapter = Path(parent_adapter).resolve()
    if not adapter.is_dir() or adapter.name != warm_start["expected_checkpoint"] or adapter.parent.name != "checkpoints":
        raise GuardrailViolation("parent adapter must be the explicit expected checkpoint directory")
    run_dir = adapter.parent.parent
    if run_dir.name != warm_start["expected_parent_run_id"]:
        raise GuardrailViolation("parent adapter run id does not match the frozen configuration")
    for forbidden in ("test_summary.json", "test_games.jsonl"):
        if (run_dir / forbidden).exists():
            raise GuardrailViolation("parent run contains a locked-test artifact")
    adapter_config_path = adapter / "adapter_config.json"
    model_files = [adapter / "adapter_model.safetensors", adapter / "adapter_model.bin"]
    model_path = next((path for path in model_files if path.is_file()), None)
    if not adapter_config_path.is_file() or model_path is None:
        raise GuardrailViolation("parent adapter is missing PEFT config or weights")
    adapter_config = read_json(adapter_config_path)
    base = str(adapter_config.get("base_model_name_or_path", "")).replace("\\", "/").rstrip("/").lower()
    accepted_base = base == SUPPORTED_MODEL_ID.lower() or base.endswith("/models/base/google--gemma-3-270m-it")
    if not accepted_base or adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise GuardrailViolation("parent must be a Gemma 3 270M causal-LM LoRA adapter")
    spec_path = run_dir / "spec.json"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    if not spec_path.is_file() or not dataset_manifest_path.is_file():
        raise GuardrailViolation("parent run is missing spec.json or dataset_manifest.json")
    if sha256_file(spec_path) != warm_start["expected_parent_spec_sha256"]:
        raise GuardrailViolation("parent spec hash mismatch")
    if sha256_file(dataset_manifest_path) != warm_start["expected_parent_dataset_manifest_sha256"]:
        raise GuardrailViolation("parent dataset manifest hash mismatch")
    spec = read_json(spec_path)
    model = spec.get("model", {})
    expected_model = {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION}
    if {key: model.get(key) for key in expected_model} != expected_model:
        raise GuardrailViolation("parent run did not use the pinned Gemma model/revision")
    parent_checks = {
        "method": spec.get("method") == warm_start["expected_parent_method"],
        "curriculum": spec.get("curriculum", {}).get("curriculum_id") == warm_start["expected_parent_curriculum_id"],
        "seed": spec.get("seed") == warm_start["expected_parent_seed"],
        "word_token_weight": spec.get("word_token_weight") == warm_start["expected_parent_word_token_weight"],
    }
    if not all(parent_checks.values()):
        raise GuardrailViolation(f"parent recipe mismatch: {[key for key, passed in parent_checks.items() if not passed]}")
    tree_hash, file_hashes = _tree_sha256(adapter)
    if tree_hash != warm_start["expected_parent_adapter_tree_sha256"]:
        raise GuardrailViolation("parent adapter tree hash mismatch")
    return {
        "status": "passed",
        "parent_run_id": run_dir.name,
        "parent_checkpoint": adapter.name,
        "parent_adapter_tree_sha256": tree_hash,
        "parent_adapter_files": file_hashes,
        "parent_spec_sha256": sha256_file(spec_path),
        "parent_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "model_id": SUPPORTED_MODEL_ID,
        "model_revision": SUPPORTED_REVISION,
        "locked_test_access": False,
    }


def _unit_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GuardrailViolation(f"parent development metric is missing or invalid: {label}")
    metric = float(value)
    if not 0.0 <= metric <= 1.0:
        raise GuardrailViolation(f"parent development metric is outside [0, 1]: {label}")
    return metric


def _blocked_prerequisite_evidence(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": BLOCKED_PREREQUISITE_STATUS,
        "passed": False,
        "reason": reason,
        "thresholds": dict(PREREQUISITE_THRESHOLDS),
        **details,
        "locked_test_access": False,
    }


def evaluate_prerequisite_thresholds(
    terminal_compliance: float,
    turn_2_posterior_violation_rate: float,
    singleton_answer_accuracy: float,
) -> dict[str, Any]:
    """Apply the declared prerequisite thresholds, including the strict turn-2 bound."""

    terminal = _unit_metric(terminal_compliance, "terminal_compliance")
    turn_two = _unit_metric(turn_2_posterior_violation_rate, "turn_2_posterior_violation_rate")
    singleton = _unit_metric(singleton_answer_accuracy, "singleton_answer_accuracy")
    checks = {
        "terminal_compliance": {
            "observed": terminal,
            "comparator": ">=",
            "threshold": PREREQUISITE_THRESHOLDS["terminal_compliance_minimum"],
            "passed": terminal >= PREREQUISITE_THRESHOLDS["terminal_compliance_minimum"],
        },
        "turn_2_posterior_violation_rate": {
            "observed": turn_two,
            "comparator": "<",
            "threshold": PREREQUISITE_THRESHOLDS["turn_2_posterior_violation_maximum_exclusive"],
            "passed": turn_two
            < PREREQUISITE_THRESHOLDS["turn_2_posterior_violation_maximum_exclusive"],
        },
        "singleton_answer_accuracy": {
            "observed": singleton,
            "comparator": ">=",
            "threshold": PREREQUISITE_THRESHOLDS["singleton_answer_accuracy_minimum"],
            "passed": singleton >= PREREQUISITE_THRESHOLDS["singleton_answer_accuracy_minimum"],
        },
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def assess_parent_prerequisite_legality(
    parent_adapter: str | Path,
    warm_start: Mapping[str, Any],
    *,
    validated_parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen dev-only legality prerequisite for Q-SFT."""

    adapter = Path(parent_adapter).resolve()
    parent = dict(validated_parent or validate_parent_adapter(adapter, warm_start))
    run_dir = adapter.parent.parent
    observed_hashes: dict[str, dict[str, Any]] = {}
    for label, declaration in PARENT_DEV_EVIDENCE.items():
        relative = Path(declaration["path"])
        if relative.is_absolute() or ".." in relative.parts or any(
            part.lower() in {"test", "locked_test", "locked-test"} for part in relative.parts
        ):
            raise GuardrailViolation("Q-SFT parent evidence declaration crossed the locked-test boundary")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise GuardrailViolation("Q-SFT parent evidence path escapes its run") from exc
        if not path.is_file():
            return _blocked_prerequisite_evidence(
                "frozen_parent_development_evidence_missing",
                parent_run_id=parent["parent_run_id"],
                evidence_available=False,
                missing_evidence=relative.as_posix(),
                evidence=observed_hashes,
            )
        observed = sha256_file(path)
        observed_hashes[label] = {
            "path": str(path.relative_to(ROOT).as_posix()),
            "expected_sha256": declaration["sha256"],
            "observed_sha256": observed,
            "matched": observed == declaration["sha256"],
        }
        if observed != declaration["sha256"]:
            return _blocked_prerequisite_evidence(
                "frozen_parent_development_evidence_hash_mismatch",
                parent_run_id=parent["parent_run_id"],
                evidence_available=False,
                evidence=observed_hashes,
            )

    summary = read_json(run_dir / PARENT_DEV_EVIDENCE["summary"]["path"])
    diagnostic = read_json(run_dir / PARENT_DEV_EVIDENCE["diagnostic_summary"]["path"])
    try:
        assert_locked_test_closed(summary)
        assert_locked_test_closed(diagnostic)
        if (
            summary.get("run_id") != parent["parent_run_id"]
            or summary.get("curriculum_id") != CURRICULUM_ID
            or summary.get("dev_secret_split") != "held-out"
            or summary.get("n_games") != 32
        ):
            raise GuardrailViolation("parent development summary identity or split mismatch")
        if summary.get("state_diagnostics") != diagnostic:
            raise GuardrailViolation("parent embedded and standalone diagnostic summaries differ")
        if diagnostic.get("artifact_id") != "7b309ade0477" or diagnostic.get("items") != 128:
            raise GuardrailViolation("parent diagnostic identity or scale mismatch")
        if diagnostic.get("items_sha256") != PARENT_DEV_EVIDENCE["diagnostic_items"]["sha256"]:
            raise GuardrailViolation("parent diagnostic item hash declaration mismatch")
        turn_two = diagnostic.get("by_turn", {}).get("2", {})
        singleton_bucket = diagnostic.get("by_posterior_size", {}).get("1", {})
        if turn_two.get("items") != 58 or singleton_bucket.get("items") != 74:
            raise GuardrailViolation("parent prerequisite diagnostic coverage mismatch")
        terminal = _unit_metric(summary.get("terminal_marker_compliance"), "terminal_compliance")
        turn_two_violation = _unit_metric(
            turn_two.get("posterior_constraint_violation_rate"),
            "turn_2_posterior_violation_rate",
        )
        singleton = _unit_metric(diagnostic.get("singleton_answer_accuracy"), "singleton_answer_accuracy")
        if singleton != _unit_metric(
            singleton_bucket.get("singleton_answer_accuracy"),
            "posterior_size_1.singleton_answer_accuracy",
        ):
            raise GuardrailViolation("parent singleton metrics disagree")
    except GuardrailViolation as exc:
        return _blocked_prerequisite_evidence(
            "frozen_parent_development_evidence_invalid",
            parent_run_id=parent["parent_run_id"],
            evidence_available=False,
            evidence=observed_hashes,
            validation_error=str(exc),
        )

    threshold_result = evaluate_prerequisite_thresholds(terminal, turn_two_violation, singleton)
    checks = threshold_result["checks"]
    failed = threshold_result["failed_checks"]
    if failed:
        return _blocked_prerequisite_evidence(
            "parent_does_not_meet_declared_development_thresholds",
            parent_run_id=parent["parent_run_id"],
            evidence_available=True,
            evidence=observed_hashes,
            diagnostic_coverage={"turn_2_items": 58, "singleton_items": 74, "total_items": 128},
            checks=checks,
            failed_checks=failed,
        )
    return {
        "status": "prerequisite_legality_gate_passed",
        "passed": True,
        "reason": None,
        "parent_run_id": parent["parent_run_id"],
        "evidence_available": True,
        "evidence": observed_hashes,
        "diagnostic_coverage": {"turn_2_items": 58, "singleton_items": 74, "total_items": 128},
        "thresholds": dict(PREREQUISITE_THRESHOLDS),
        "checks": checks,
        "failed_checks": [],
        "locked_test_access": False,
    }


def _blocked_q_sft_result(
    parent: Mapping[str, Any],
    prerequisite: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": BLOCKED_PREREQUISITE_STATUS,
        "experiment_id": EXPERIMENT_ID,
        "parent": dict(parent),
        "prerequisite_legality_gate": dict(prerequisite),
        "training_started": False,
        "run_directory_created": False,
        "next_required_action": (
            "Select a frozen Gemma parent with development evidence meeting every legality threshold; "
            "do not run Q-SFT from this historical parent."
        ),
        "locked_test_access": False,
    }


def build_bundle(
    config: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_BUNDLE,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    force: bool = False,
) -> dict[str, Any]:
    """Persist deterministic snapshots, sanitized training rows, and hashes."""

    validate_config(config)
    protocol_audit = assert_protocol_lock_unchanged()
    source_dir, source_rows, source_manifest = _source_context(config)
    snapshots, target_audit = build_frozen_snapshots(source_rows, **{
        "gamma": config["target"]["gamma"],
        "universe_size": config["target"]["universe_size"],
        "max_valid_guesses": config["target"]["max_valid_guesses"],
    })
    joined = join_snapshots_to_training_rows(source_rows, snapshots)
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        return audit_bundle(config, output, config_path=config_path)
    output.mkdir(parents=True, exist_ok=True)
    snapshots_path = write_jsonl(output / "snapshots.jsonl", snapshots)
    joined_path = write_jsonl(output / "training_rows.jsonl", joined)
    manifest = {
        "schema_version": "q-sft-frozen-bundle-v2",
        "experiment_id": EXPERIMENT_ID,
        "protocol": protocol_audit,
        "source": {
            "curriculum_id": source_manifest["curriculum_id"],
            "directory": str(source_dir.relative_to(ROOT).as_posix()),
            "training_rows": len(source_rows),
            "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "source_training_sha256": sha256_file(source_dir / config["data"]["training_file"]),
            "state_manifest_sha256": sha256_file(source_dir / "state_manifest.jsonl"),
        },
        "target": target_audit,
        "behavior_contract": dict(config["behavior_provenance"]),
        "behavior_provenance": target_audit["behavior_provenance"],
        "files": {
            "snapshots.jsonl": {"rows": len(snapshots), "sha256": sha256_file(snapshots_path)},
            "training_rows.jsonl": {"rows": len(joined), "sha256": sha256_file(joined_path)},
        },
        "config_sha256": sha256_file(config_path),
        "emitted_snapshot_fields": sorted(SNAPSHOT_FIELDS),
        "emitted_training_fields": sorted(JOINED_FIELDS),
        "forbidden_fields_absent": True,
        "locked_test_access": False,
    }
    manifest["manifest_content_sha256"] = sha256_text(canonical_json(manifest))
    write_json(manifest_path, manifest)
    return audit_bundle(config, output, config_path=config_path)


def audit_bundle(
    config: Mapping[str, Any],
    bundle_dir: str | Path = DEFAULT_BUNDLE,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Rebuild targets in memory and verify an existing frozen snapshot bundle."""

    validate_config(config)
    assert_protocol_lock_unchanged()
    output = Path(bundle_dir).resolve()
    manifest = read_json(output / "manifest.json")
    content_hash = manifest.pop("manifest_content_sha256", None)
    if content_hash != sha256_text(canonical_json(manifest)):
        raise GuardrailViolation("Q-SFT bundle manifest content hash mismatch")
    if manifest.get("schema_version") != "q-sft-frozen-bundle-v2" or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise GuardrailViolation("Q-SFT bundle manifest identity mismatch")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise GuardrailViolation("Q-SFT bundle config hash mismatch")
    assert_locked_test_closed(manifest, require_explicit=True)
    source_dir, source_rows, _ = _source_context(config)
    expected_snapshots, target_audit = build_frozen_snapshots(
        source_rows,
        gamma=config["target"]["gamma"],
        universe_size=config["target"]["universe_size"],
        max_valid_guesses=config["target"]["max_valid_guesses"],
    )
    expected_joined = join_snapshots_to_training_rows(source_rows, expected_snapshots)
    expected_source = {
        "curriculum_id": CURRICULUM_ID,
        "directory": str(source_dir.relative_to(ROOT).as_posix()),
        "training_rows": len(source_rows),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "source_training_sha256": sha256_file(source_dir / config["data"]["training_file"]),
        "state_manifest_sha256": sha256_file(source_dir / "state_manifest.jsonl"),
    }
    if (
        manifest.get("source") != expected_source
        or manifest.get("target") != target_audit
        or manifest.get("behavior_contract") != config["behavior_provenance"]
        or manifest.get("behavior_provenance") != target_audit["behavior_provenance"]
    ):
        raise GuardrailViolation("Q-SFT bundle source or target provenance mismatch")
    protocol = manifest.get("protocol", {})
    current_protocol = assert_protocol_lock_unchanged()
    if protocol != current_protocol:
        raise GuardrailViolation("Q-SFT bundle protocol provenance mismatch")
    if manifest.get("emitted_snapshot_fields") != sorted(SNAPSHOT_FIELDS):
        raise GuardrailViolation("Q-SFT bundle snapshot-field declaration mismatch")
    if manifest.get("emitted_training_fields") != sorted(JOINED_FIELDS) or manifest.get("forbidden_fields_absent") is not True:
        raise GuardrailViolation("Q-SFT bundle training-field declaration mismatch")
    snapshots_path = output / "snapshots.jsonl"
    joined_path = output / "training_rows.jsonl"
    snapshots = read_jsonl(snapshots_path)
    joined = read_jsonl(joined_path)
    if snapshots != expected_snapshots or joined != expected_joined:
        raise GuardrailViolation("Q-SFT bundle content differs from deterministic rebuild")
    for name, path, row_count in (
        ("snapshots.jsonl", snapshots_path, len(snapshots)),
        ("training_rows.jsonl", joined_path, len(joined)),
    ):
        entry = manifest.get("files", {}).get(name, {})
        if entry != {"rows": row_count, "sha256": sha256_file(path)}:
            raise GuardrailViolation(f"Q-SFT bundle artifact hash/count mismatch: {name}")
    validate_snapshot_rows(snapshots)
    validate_q_sft_rows(joined, discount=config["training"]["discount"])
    return {
        "status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "bundle_dir": str(output),
        "rows": len(joined),
        "snapshot_sha256": sha256_file(snapshots_path),
        "training_rows_sha256": sha256_file(joined_path),
        "bundle_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_training_sha256": sha256_file(source_dir / config["data"]["training_file"]),
        "target": target_audit,
        "behavior_provenance": target_audit["behavior_provenance"],
        "checks": [
            "protocol_lock_unchanged",
            "source_hashes_exact",
            "deterministic_target_rebuild",
            "behavior_provenance_recomputed",
            "join_ids_exact",
            "core_q_sft_validation_passed",
            "secret_candidate_evaluator_fields_absent",
        ],
        "locked_test_access": False,
    }


def dry_run(
    config: Mapping[str, Any],
    *,
    parent_adapter: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate every input and compute hashes without writing or training."""

    validate_config(config)
    protocol = assert_protocol_lock_unchanged()
    parent = validate_parent_adapter(parent_adapter, config["warm_start"])
    prerequisite = assess_parent_prerequisite_legality(
        parent_adapter,
        config["warm_start"],
        validated_parent=parent,
    )
    if not prerequisite["passed"]:
        return {
            **_blocked_q_sft_result(parent, prerequisite),
            "config_sha256": sha256_file(config_path),
            "protocol": protocol,
        }
    _, source_rows, _ = _source_context(config)
    snapshots, target = build_frozen_snapshots(
        source_rows,
        gamma=config["target"]["gamma"],
        universe_size=config["target"]["universe_size"],
        max_valid_guesses=config["target"]["max_valid_guesses"],
    )
    joined = join_snapshots_to_training_rows(source_rows, snapshots)
    return {
        "status": "dry_run_passed",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "protocol": protocol,
        "parent": parent,
        "prerequisite_legality_gate": prerequisite,
        "training_rows": len(joined),
        "training_rows_sha256": _jsonl_sha256(joined),
        "target": target,
        "canonical_evaluation_policy": "natural_generation_under_WORDLE-PROTOCOL-002",
        "locked_test_access": False,
    }


def _training_spec(
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    prerequisite: Mapping[str, Any],
    bundle: Mapping[str, Any],
    parent_path: Path,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "method": "q_sft",
        "objective": config["objective"],
        "representation": "common_balanced_curriculum",
        "seed": training["seed"],
        "max_steps": training["max_steps"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "max_length": training["max_length"],
        "warmup_fraction": training["warmup_fraction"],
        "max_grad_norm": training["max_grad_norm"],
        "discount": training["discount"],
        "lora": training["lora"],
        "parent_checkpoint": str(parent_path),
        "parent_provenance": dict(parent),
        "prerequisite_legality_gate": dict(prerequisite),
        "protocol_id": config["protocol_id"],
        "protocol_sha256": assert_protocol_lock_unchanged()["protocol_sha256"],
        "model": config["model"],
        "implementation_sha256": sha256_file(__file__),
        "data": {
            "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
            "training_rows_sha256": bundle["training_rows_sha256"],
            "source_training_sha256": bundle["source_training_sha256"],
            "target_id": TARGET_ID,
            "behavior_policy_id": bundle["behavior_provenance"]["policy_id"],
            "behavior_state_supports_sha256": bundle["behavior_provenance"]["state_supports_sha256"],
        },
        "canonical_evaluation_policy": "natural_generation_under_WORDLE-PROTOCOL-002",
        "locked_test_access": False,
        "candidate_injection": False,
        "reranking": False,
        "output_repair": False,
    }


def train(
    config: Mapping[str, Any],
    *,
    parent_adapter: str | Path,
    bundle_dir: str | Path = DEFAULT_BUNDLE,
    output_root: str | Path = ARTIFACTS / "runs",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Warm-start the existing core Q-SFT trainer from the explicit parent."""

    validate_config(config)
    parent_path = Path(parent_adapter).resolve()
    parent = validate_parent_adapter(parent_path, config["warm_start"])
    prerequisite = assess_parent_prerequisite_legality(
        parent_path,
        config["warm_start"],
        validated_parent=parent,
    )
    if not prerequisite["passed"]:
        return _blocked_q_sft_result(parent, prerequisite)
    bundle = audit_bundle(config, bundle_dir, config_path=config_path)
    spec = _training_spec(config, parent, prerequisite, bundle, parent_path)
    run_digest = sha256_text(canonical_json({key: value for key, value in spec.items() if key != "parent_checkpoint"}))[:10]
    run_id = f"q-sft-balanced-frozen-s{spec['seed']}-{run_digest}"
    run_dir = Path(output_root).resolve() / run_id
    if run_dir.exists():
        raise GuardrailViolation(f"Q-SFT run directory already exists; refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    spec.update(
        {
            "run_id": run_id,
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "git_commit": git_commit(),
            "source_tree_sha256": source_tree_sha256(),
        }
    )
    write_json(run_dir / "spec.json", spec)
    bundle_path = Path(bundle_dir).resolve()
    write_json(run_dir / "dataset_manifest.json", read_json(bundle_path / "manifest.json"))
    rows = read_jsonl(bundle_path / "training_rows.jsonl")
    set_seed(spec["seed"])
    model, accounting = train_q_sft(rows, parent_path, run_dir, spec)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "status": "trained_pending_dev_evaluation",
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "split": "dev",
        "locked_test_access": False,
        "accounting": accounting,
    }
    write_json(run_dir / "summary.json", summary)
    return {**summary, "run_dir": str(run_dir)}


def evaluate_run(
    config: Mapping[str, Any],
    *,
    parent_adapter: str | Path,
    run_dir: str | Path,
    checkpoint: str = "final",
    bundle_dir: str | Path = DEFAULT_BUNDLE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Evaluate by unchanged natural generation on development data only."""

    validate_config(config)
    assert_protocol_lock_unchanged()
    parent = validate_parent_adapter(parent_adapter, config["warm_start"])
    bundle = audit_bundle(config, bundle_dir, config_path=config_path)
    run = Path(run_dir).resolve()
    spec = read_json(run / "spec.json")
    assert_locked_test_closed(spec, require_explicit=True)
    expected = {
        "method": "q_sft",
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": assert_protocol_lock_unchanged()["protocol_sha256"],
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise GuardrailViolation("Q-SFT run spec identity/protocol mismatch")
    if spec.get("parent_provenance", {}).get("parent_adapter_tree_sha256") != parent["parent_adapter_tree_sha256"]:
        raise GuardrailViolation("Q-SFT run parent provenance mismatch")
    if spec.get("data", {}).get("bundle_manifest_sha256") != bundle["bundle_manifest_sha256"]:
        raise GuardrailViolation("Q-SFT run data provenance mismatch")
    checkpoint_dir = (run / "checkpoints" / checkpoint).resolve()
    try:
        checkpoint_dir.relative_to(run)
    except ValueError as exc:
        raise GuardrailViolation("checkpoint path escapes the Q-SFT run") from exc
    if not checkpoint_dir.is_dir():
        raise GuardrailViolation(f"Q-SFT checkpoint is missing: {checkpoint_dir}")

    source_dir, source_rows, _ = _source_context(config)
    universe = read_json(source_dir / "universe.json")
    dev_answers = read_json(source_dir / "dev_secrets.json")[: config["evaluation"]["dev_games"]]
    allowed = [
        line.strip().upper()
        for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_records = generate_canonical_states(
        dev_answers,
        "q_sft_frozen_dev_diagnostic",
        config["evaluation"]["diagnostic_items"],
        seed=config["training"]["seed"],
        answer_vocabulary=universe,
    )
    training_records = [row["source_state"] for row in source_rows]
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_adapter(checkpoint_dir)
    try:
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        write_jsonl(run / "games.jsonl", games)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            training_records,
            allowed,
            universe,
            run,
        )
        retention_rows, retention = evaluate_retention(
            model,
            tokenizer,
            read_jsonl(_resolve(config["evaluation"]["retention_probes"])),
        )
        write_jsonl(run / "retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": EXPERIMENT_ID,
            "run_id": spec["run_id"],
            "recipe_id": EXPERIMENT_ID,
            "seed": spec["seed"],
            "split": "dev",
            "locked_test_access": False,
            "protocol_id": spec["protocol_id"],
            "protocol_sha256": spec["protocol_sha256"],
            "model_id": SUPPORTED_MODEL_ID,
            "model_revision": SUPPORTED_REVISION,
            "dataset_manifest_sha256": bundle["bundle_manifest_sha256"],
            "checkpoint": checkpoint,
            "canonical_evaluation_policy": "natural_generation_under_WORDLE-PROTOCOL-002",
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
            "parent_provenance": parent,
        }
        write_json(run / "summary.json", summary)
        gate_metrics = normalize_gate_metrics(summary)
        write_json(run / "gate_metrics.json", gate_metrics)
        provenance = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": spec["protocol_id"],
            "protocol_sha256": spec["protocol_sha256"],
            "model_id": SUPPORTED_MODEL_ID,
            "model_revision": SUPPORTED_REVISION,
            "seed": spec["seed"],
            "split": "dev",
            "locked_test_access": False,
            "dataset_manifest_sha256": bundle["bundle_manifest_sha256"],
            "source_tree_sha256": spec["source_tree_sha256"],
            "git_commit": spec["git_commit"],
        }
        artifact_manifest = build_artifact_manifest(
            run,
            provenance=provenance,
            required_artifacts=Q_SFT_REQUIRED_ARTIFACTS,
        )
        write_json(run / "artifact_manifest.json", artifact_manifest)
        verify_artifact_manifest(run, artifact_manifest, required_artifacts=Q_SFT_REQUIRED_ARTIFACTS)
        return {**summary, "gate_metrics": gate_metrics, "artifact_manifest": artifact_manifest}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-target, train-only balanced-002 Q-SFT experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write deterministic frozen snapshots and sanitized joined rows")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE)
    build.add_argument("--force", action="store_true")
    dry = commands.add_parser(
        "dry-run",
        help="validate protocol, parent, and prerequisite development evidence without training",
    )
    dry.add_argument("--parent-adapter", type=Path, required=True)
    train_parser = commands.add_parser("train", help="run the existing core Q-SFT trainer")
    train_parser.add_argument("--parent-adapter", type=Path, required=True)
    train_parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    train_parser.add_argument("--output-root", type=Path, default=ARTIFACTS / "runs")
    evaluate_parser = commands.add_parser("evaluate", help="evaluate one Q-SFT checkpoint on development only")
    evaluate_parser.add_argument("--parent-adapter", type=Path, required=True)
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", default="final")
    evaluate_parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "build":
        result = build_bundle(config, args.output_dir, config_path=args.config, force=args.force)
    elif args.command == "dry-run":
        result = dry_run(config, parent_adapter=args.parent_adapter, config_path=args.config)
    elif args.command == "train":
        result = train(
            config,
            parent_adapter=args.parent_adapter,
            bundle_dir=args.bundle_dir,
            output_root=args.output_root,
            config_path=args.config,
        )
    else:
        result = evaluate_run(
            config,
            parent_adapter=args.parent_adapter,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            bundle_dir=args.bundle_dir,
            config_path=args.config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result.get("status") == BLOCKED_PREREQUISITE_STATUS else 0


if __name__ == "__main__":
    raise SystemExit(main())
