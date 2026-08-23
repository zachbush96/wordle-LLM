from __future__ import annotations

"""Legality-first Wordle policy data and natural-generation evaluation."""

import argparse
import gc
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from wordle_lab.analysis.state_diagnostics import build_probe_items, run_state_diagnostics, score_probe_outputs
from wordle_lab.common import ARTIFACTS, DATA, ROOT, canonical_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.models import load_adapter, load_tokenizer, model_metadata
from wordle_lab.protocol import generation
from wordle_lab.protocol.env import posterior_candidates
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention


CONSTRAINT_POLICY_ID = "GEMMA-CONSTRAINT-FIRST-POLICY-001"
SUITE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "common-curriculum-002" / "u128-train96"
DEFAULT_OUTPUT = SUITE_ROOT / "generated" / "constraint_first"
CONSTRAINT_FIRST_CHECKPOINTS = (
    "step-000150",
    "step-000300",
    "step-000450",
    "step-000600",
)
CONSTRAINT_FIRST_DEV_GAMES = 32
CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS = 128
CONSTRAINT_FIRST_RETENTION_PROBES = 200
SAMPLED_MULTI_LABEL_OBJECTIVE = (
    "word-focused SFT over up to four deterministic legal labels per state; "
    "sampled support approximation, not a set-normalized loss"
)

# These single-seed development thresholds and their dose-selection order were
# added after the completed run had begun. They are therefore a transparent
# post-hoc analysis policy, not a preregistered part of that training condition.
# Passing them selects a replication candidate only; it never opens the locked
# test. The retention floor tolerates some movement from the 0.30 base-model
# score while rejecting catastrophic collapse, and the gameplay floor requires
# matching the historical 8/32 balanced-002 result rather than declaring
# success from legality alone.
CONSTRAINT_FIRST_GATE_THRESHOLDS = {
    "terminal_marker_compliance": {"group": "format", "op": ">=", "value": 0.99},
    "invalid_guess_rate": {"group": "legality", "op": "<=", "value": 0.01},
    "repeat_guess_rate": {"group": "legality", "op": "<=", "value": 0.10},
    "gameplay_constraint_violation_rate": {"group": "legality", "op": "<=", "value": 0.30},
    "posterior_constraint_violation_rate": {"group": "posterior", "op": "<=", "value": 0.30},
    "turn_2_posterior_constraint_violation_rate": {"group": "posterior", "op": "<", "value": 0.30},
    "singleton_answer_accuracy": {"group": "singleton", "op": ">=", "value": 0.80},
    "retention_overall_score": {"group": "retention", "op": ">=", "value": 0.20},
    "win_rate": {"group": "gameplay", "op": ">=", "value": 0.25},
}


def constraint_first_evaluation_policy() -> dict[str, Any]:
    """Return evaluation metadata kept deliberately outside the training spec."""
    return {
        "schema_version": "constraint-first-post-hoc-evaluation-policy-v1",
        "registration_status": "post_hoc_after_training_started",
        "preregistered": False,
        "training_objective": {
            "family": "sampled_multi_label_word_focused_sft",
            "description": SAMPLED_MULTI_LABEL_OBJECTIVE,
            "set_normalized_loss": False,
        },
        "checkpoints": list(CONSTRAINT_FIRST_CHECKPOINTS),
        "gate_thresholds": CONSTRAINT_FIRST_GATE_THRESHOLDS,
        "selection": (
            "among gate-passing doses, minimize overall then turn-2 posterior violation; "
            "maximize singleton accuracy, format compliance, retention, and win rate; "
            "then minimize gameplay violations/repeats and prefer the earlier dose"
        ),
        "interpretation_boundary": (
            "post-hoc single-seed development comparison; thresholds and selection order "
            "were not used to define or alter the completed training run"
        ),
        "locked_test_access": False,
    }


def _history(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(item["guess"]).upper(), str(item["feedback"]).upper())
        for item in row["source_state"]["history"]
    ]


def _hash_words(words: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(words)).encode("utf-8")).hexdigest()


def build_constraint_first_bundle(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    seed: int = 2026,
    singleton_multiplier: int = 2,
    legal_labels_per_state: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Approximate a set-valued legal objective with multiple labels per state."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return read_json(manifest_path)
    if singleton_multiplier < 1:
        raise ValueError("singleton_multiplier must be at least 1")
    if legal_labels_per_state < 2:
        raise ValueError("legal_labels_per_state must be at least 2")
    source_rows = read_jsonl(source_dir / "train.jsonl")
    universe = [str(word).upper() for word in read_json(source_dir / "universe.json")]
    train_secrets = set(read_json(source_dir / "train_secrets.json"))
    dev_secrets = set(read_json(source_dir / "dev_secrets.json"))
    source_manifest = read_json(source_dir / "manifest.json")
    if train_secrets & dev_secrets:
        raise AssertionError("source train/dev overlap")
    if sha256_file(source_dir / "train.jsonl") != source_manifest["rendered_sha256"]:
        raise AssertionError("balanced-002 source hash drift")

    unique_sources: dict[str, dict[str, Any]] = {}
    for source_row in source_rows:
        state_id = str(source_row["state_id"])
        previous = unique_sources.setdefault(state_id, source_row)
        if canonical_json(previous["source_state"]) != canonical_json(source_row["source_state"]):
            raise AssertionError(f"source state id collision: {state_id}")

    occurrences: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    source_targets: dict[str, set[str]] = defaultdict(set)
    posterior_sizes: dict[str, int] = {}
    for state_id, source_row in sorted(unique_sources.items()):
        source = source_row["source_state"]
        if source["secret_answer"] not in train_secrets or source["secret_answer"] in dev_secrets:
            raise AssertionError("constraint-first source is not training-only")
        history = _history(source_row)
        posterior = [word for word in posterior_candidates(history, universe) if word not in {guess for guess, _ in history}]
        if not posterior:
            raise AssertionError(f"no legal posterior action for {source_row['state_id']}")
        posterior_sizes[state_id] = len(posterior)
        offset = int(hashlib.sha256(f"{seed}:{state_id}".encode("utf-8")).hexdigest(), 16) % len(posterior)
        target_count = 1 if len(posterior) == 1 else min(legal_labels_per_state, len(posterior))
        for occurrence in range(target_count):
            occurrences[state_id] += 1
            target = posterior[(offset + occurrence) % len(posterior)]
            source_targets[state_id].add(target)
            base = {
                "state_id": state_id,
                "split": "training_only",
                "task": "constraint_first_full_policy",
                "turn": len(history) + 1,
                "history": [{"guess": guess, "feedback": feedback} for guess, feedback in history],
                "posterior_size": len(posterior),
                "acceptable_action_count": len(posterior),
                "acceptable_action_set_sha256": _hash_words(posterior),
                "target_word": target,
                "target_policy": "deterministic_multi_label_posterior_consistent_nonrepeat",
                "prompt": source_row["prompt"],
                "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
                "source_state_sha256": hashlib.sha256(canonical_json(source).encode("utf-8")).hexdigest(),
                "locked_test_access": False,
            }
            copies = singleton_multiplier if len(posterior) == 1 else 1
            for copy in range(copies):
                rows.append({"example_id": f"constraint-{state_id}-{occurrence:02d}-{copy:02d}", **base})

    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['example_id']}".encode("utf-8")).hexdigest())
    rows_path = write_jsonl(output_dir / "train.jsonl", rows)
    singleton_rows = [row for row in rows if row["posterior_size"] == 1]
    manifest = {
        "experiment_id": CONSTRAINT_POLICY_ID,
        "protocol_id": "WORDLE-PROTOCOL-002",
        "locked_test_access": False,
        "source_directory": str(source_dir.relative_to(ROOT)).replace("\\", "/"),
        "source_curriculum_id": source_manifest["curriculum_id"],
        "source_rows_sha256": sha256_file(source_dir / "train.jsonl"),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "universe_sha256": sha256_file(source_dir / "universe.json"),
        "train_secrets_sha256": sha256_file(source_dir / "train_secrets.json"),
        "dev_secrets_sha256": sha256_file(source_dir / "dev_secrets.json"),
        "canonical_sha256": sha256_file(source_dir / "canonical.jsonl"),
        "allowed_words_sha256": sha256_file(ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt"),
        "retention_probes_sha256": sha256_file(DATA / "retention_probes_v1.jsonl"),
        "protocol_lock_file_sha256": sha256_file(DATA / "protocol_lock.json"),
        "rows": len(rows),
        "rows_sha256": sha256_file(rows_path),
        "unique_source_states": len(unique_sources),
        "singleton_rows": len(singleton_rows),
        "singleton_source_states": len({row["state_id"] for row in singleton_rows}),
        "singleton_multiplier": singleton_multiplier,
        "legal_labels_per_state": legal_labels_per_state,
        "states_with_multiple_legal_labels": sum(len(targets) > 1 for targets in source_targets.values()),
        "non_singleton_source_states": sum(size > 1 for size in posterior_sizes.values()),
        "states_meeting_declared_label_coverage": sum(
            len(source_targets[state_id]) == (1 if size == 1 else min(legal_labels_per_state, size))
            for state_id, size in posterior_sizes.items()
        ),
        "target_distribution": dict(sorted(Counter(row["target_word"] for row in rows).items())),
        "objective": SAMPLED_MULTI_LABEL_OBJECTIVE,
        "evaluation_primary": [
            "terminal_compliance",
            "posterior_constraint_violation_rate",
            "turn_2_posterior_constraint_violation_rate",
            "repeat_rate",
            "singleton_answer_accuracy",
        ],
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def constraint_policy_spec(
    rows_path: Path = DEFAULT_OUTPUT / "train.jsonl",
    *,
    steps: int = 600,
    seed: int = 2026,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    from wordle_lab.methods.unsloth_sft import UNSLOTH_WEIGHTED_BACKEND_ID, validate_unsloth_objective

    manifest = read_json(rows_path.parent / "manifest.json")
    protocol = read_json(DATA / "protocol_lock.json")
    if sha256_file(rows_path) != manifest["rows_sha256"]:
        raise AssertionError("constraint-first row hash drift")
    source_dir = ROOT / manifest["source_directory"]
    source_files = {
        "source_manifest_sha256": source_dir / "manifest.json",
        "source_rows_sha256": source_dir / "train.jsonl",
        "universe_sha256": source_dir / "universe.json",
        "train_secrets_sha256": source_dir / "train_secrets.json",
        "dev_secrets_sha256": source_dir / "dev_secrets.json",
        "canonical_sha256": source_dir / "canonical.jsonl",
        "allowed_words_sha256": ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt",
        "retention_probes_sha256": DATA / "retention_probes_v1.jsonl",
        "protocol_lock_file_sha256": DATA / "protocol_lock.json",
    }
    drift = {
        key: {"expected": manifest.get(key), "actual": sha256_file(path)}
        for key, path in source_files.items()
        if manifest.get(key) != sha256_file(path)
    }
    if drift:
        raise AssertionError(f"constraint-first source/evaluation provenance drift: {drift}")
    spec = {
        "experiment_id": CONSTRAINT_POLICY_ID,
        "method": "unsloth_constraint_first_sft",
        "backend": UNSLOTH_WEIGHTED_BACKEND_ID,
        "representation": "constraint_first_full_policy",
        "seed": seed,
        "max_steps": steps,
        "learning_rate": learning_rate,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": 320,
        "word_token_weight": 8.0,
        "quantization": "none_16bit",
        "gradient_checkpointing": "unsloth",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "model": model_metadata(),
        "data": {
            "path": str(rows_path),
            "rows": manifest["rows"],
            "sha256": manifest["rows_sha256"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_rows_sha256": manifest["source_rows_sha256"],
            "universe_sha256": manifest["universe_sha256"],
            "train_secrets_sha256": manifest["train_secrets_sha256"],
            "dev_secrets_sha256": manifest["dev_secrets_sha256"],
            "canonical_sha256": manifest["canonical_sha256"],
        },
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_lock_file_sha256": manifest["protocol_lock_file_sha256"],
        "evaluation_contract": {
            "allowed_words_sha256": manifest["allowed_words_sha256"],
            "retention_probes_sha256": manifest["retention_probes_sha256"],
            "canonical_sha256": manifest["canonical_sha256"],
            "dev_secrets_sha256": manifest["dev_secrets_sha256"],
            "universe_sha256": manifest["universe_sha256"],
        },
        "locked_test_access": False,
        "candidate_injection": False,
        "reranking": False,
        "output_repair": False,
    }
    validate_unsloth_objective(spec)
    return spec


def _require_frozen_constraint_evaluation_request(source_dir: Path, dev_games: int) -> Path:
    """Reject any evaluation request outside the frozen development contract."""
    source_dir = Path(source_dir)
    if source_dir.resolve() != DEFAULT_SOURCE.resolve():
        raise RuntimeError("constraint-first evaluation source must be the preregistered balanced-002 directory")
    if isinstance(dev_games, bool) or dev_games != CONSTRAINT_FIRST_DEV_GAMES:
        raise RuntimeError(
            f"constraint-first evaluation requires exactly {CONSTRAINT_FIRST_DEV_GAMES} frozen development games"
        )
    return source_dir


def _constraint_evaluation_context(
    run_dir: Path,
    source_dir: Path = DEFAULT_SOURCE,
    dev_games: int = CONSTRAINT_FIRST_DEV_GAMES,
) -> dict[str, Any]:
    """Load and hash the complete, immutable development evaluation input set."""
    run_dir = Path(run_dir)
    source_dir = _require_frozen_constraint_evaluation_request(source_dir, dev_games)
    spec = read_json(run_dir / "spec.json")
    if spec.get("experiment_id") != CONSTRAINT_POLICY_ID or spec.get("locked_test_access") is not False:
        raise RuntimeError("not a locked-test-free constraint-first run")
    expected_spec = constraint_policy_spec(DEFAULT_OUTPUT / "train.jsonl")
    if spec != expected_spec:
        raise RuntimeError("constraint-first run spec differs from the fully reconstructed contract")

    paths = {
        "allowed_words_sha256": ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt",
        "canonical_sha256": source_dir / "canonical.jsonl",
        "dev_secrets_sha256": source_dir / "dev_secrets.json",
        "retention_probes_sha256": DATA / "retention_probes_v1.jsonl",
        "universe_sha256": source_dir / "universe.json",
    }
    observed_contract = {key: sha256_file(path) for key, path in paths.items()}
    if spec.get("evaluation_contract") != observed_contract:
        raise RuntimeError("constraint-first evaluation input hashes differ from the run contract")

    protocol_lock_path = DATA / "protocol_lock.json"
    protocol_lock = read_json(protocol_lock_path)
    if (
        spec.get("protocol_id") != protocol_lock.get("protocol_id")
        or spec.get("protocol_sha256") != protocol_lock.get("protocol_sha256")
        or spec.get("protocol_lock_file_sha256") != sha256_file(protocol_lock_path)
    ):
        raise RuntimeError("constraint-first protocol binding differs from the frozen protocol lock")

    dev_answers = read_json(paths["dev_secrets_sha256"])
    if (
        not isinstance(dev_answers, list)
        or len(dev_answers) != CONSTRAINT_FIRST_DEV_GAMES
        or len(set(dev_answers)) != CONSTRAINT_FIRST_DEV_GAMES
        or any(not isinstance(word, str) or len(word) != 5 or word != word.upper() for word in dev_answers)
    ):
        raise RuntimeError("constraint-first development file must contain exactly 32 unique uppercase five-letter words")
    universe = read_json(paths["universe_sha256"])
    allowed = [
        line.strip().upper()
        for line in paths["allowed_words_sha256"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retention_probes = read_jsonl(paths["retention_probes_sha256"])
    if len(retention_probes) != CONSTRAINT_FIRST_RETENTION_PROBES:
        raise RuntimeError(
            f"constraint-first retention input must contain exactly {CONSTRAINT_FIRST_RETENTION_PROBES} probes"
        )
    probes = generate_canonical_states(
        dev_answers,
        "constraint_first_dev",
        CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS,
        seed=int(spec["seed"]),
        answer_vocabulary=universe,
    )
    if len(probes) != CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS:
        raise RuntimeError("constraint-first diagnostic generator did not produce exactly 128 frozen probes")
    training_records = read_jsonl(source_dir / "canonical.jsonl")
    diagnostic_inputs = build_probe_items(probes, training_records)
    binding = {
        "schema_version": "constraint-first-evaluation-inputs-v1",
        "run_id": run_dir.name,
        "experiment_id": CONSTRAINT_POLICY_ID,
        "split": "dev",
        "dev_games": CONSTRAINT_FIRST_DEV_GAMES,
        "diagnostic_items": CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS,
        "retention_probes": CONSTRAINT_FIRST_RETENTION_PROBES,
        "dev_answer_order_sha256": hashlib.sha256(canonical_json(dev_answers).encode("utf-8")).hexdigest(),
        "diagnostic_inputs_sha256": hashlib.sha256(
            canonical_json(diagnostic_inputs).encode("utf-8")
        ).hexdigest(),
        "evaluation_contract": observed_contract,
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": spec["protocol_sha256"],
        "protocol_lock_file_sha256": spec["protocol_lock_file_sha256"],
        "locked_test_access": False,
    }
    return {
        "spec": spec,
        "dev_answers": dev_answers,
        "universe": universe,
        "allowed": allowed,
        "retention_probes": retention_probes,
        "probes": probes,
        "training_records": training_records,
        "diagnostic_inputs": diagnostic_inputs,
        "binding": binding,
    }


def _resolve_declared_artifact_path(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _recomputed_gameplay_gate_metrics(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for game in games for turn in game.get("turns", [])]
    valid = [turn for turn in turns if turn.get("valid")]
    return {
        "n_games": len(games),
        "wins": sum(bool(game.get("won")) for game in games),
        "win_rate": sum(bool(game.get("won")) for game in games) / len(games),
        "terminal_marker_compliance": sum(bool(turn.get("format_valid")) for turn in turns) / max(1, len(turns)),
        "invalid_guess_rate": sum(not bool(turn.get("valid")) for turn in turns) / max(1, len(turns)),
        "repeat_guess_rate": sum(bool(turn.get("repeat")) for turn in valid) / max(1, len(valid)),
        "constraint_violation_rate": sum(bool(turn.get("constraint_violation")) for turn in valid)
        / max(1, len(valid)),
    }


def validate_reused_constraint_summary(
    run_dir: Path,
    checkpoint: str,
    summary: dict[str, Any],
    *,
    source_dir: Path = DEFAULT_SOURCE,
    dev_games: int = CONSTRAINT_FIRST_DEV_GAMES,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate a saved dose summary against its run and raw evaluation artifacts."""
    run_dir = Path(run_dir)
    if checkpoint not in CONSTRAINT_FIRST_CHECKPOINTS:
        raise RuntimeError(f"unexpected constraint-first checkpoint: {checkpoint}")
    context = evaluation_context or _constraint_evaluation_context(run_dir, source_dir, dev_games)
    spec = context["spec"]
    if not (run_dir / "checkpoints" / checkpoint).is_dir():
        raise RuntimeError(f"constraint-first checkpoint directory is missing: {checkpoint}")
    training_summary = read_json(run_dir / "summary.json")
    if (
        training_summary.get("status") != "trained"
        or training_summary.get("locked_test_access") is not False
        or _resolve_declared_artifact_path(training_summary.get("run_dir")) != run_dir.resolve()
    ):
        raise RuntimeError("constraint-first training summary does not bind this run directory")
    if (
        summary.get("status") != "dev_evaluated"
        or summary.get("experiment_id") != CONSTRAINT_POLICY_ID
        or summary.get("checkpoint") != checkpoint
        or summary.get("split") != "dev"
        or summary.get("locked_test_access") is not False
    ):
        raise RuntimeError("constraint-first reused summary has the wrong run, checkpoint, split, or lock identity")
    if summary.get("evaluation_policy") is not None and summary["evaluation_policy"] != constraint_first_evaluation_policy():
        raise RuntimeError("constraint-first reused summary evaluation policy drift")
    optional_bindings = {
        "run_id": run_dir.name,
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": spec["protocol_sha256"],
        "protocol_lock_file_sha256": spec["protocol_lock_file_sha256"],
        "evaluation_contract": spec["evaluation_contract"],
        "evaluation_input_contract": context["binding"],
    }
    for key, expected in optional_bindings.items():
        if key in summary and summary[key] != expected:
            raise RuntimeError(f"constraint-first reused summary {key} binding mismatch")

    games_path = run_dir / f"eval-{checkpoint}-games.jsonl"
    retention_path = run_dir / f"eval-{checkpoint}-retention.jsonl"
    games = read_jsonl(games_path)
    expected_answers = context["dev_answers"]
    if (
        len(games) != CONSTRAINT_FIRST_DEV_GAMES
        or [game.get("game_id") for game in games] != list(range(CONSTRAINT_FIRST_DEV_GAMES))
        or [game.get("answer") for game in games] != expected_answers
    ):
        raise RuntimeError("constraint-first gameplay artifact is not the exact frozen 32-game development set")
    recomputed_gameplay = _recomputed_gameplay_gate_metrics(games)
    gameplay = summary.get("gameplay")
    if not isinstance(gameplay, dict) or any(gameplay.get(key) != value for key, value in recomputed_gameplay.items()):
        raise RuntimeError("constraint-first gameplay summary differs from recomputed raw game metrics")

    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("items") != CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS:
        raise RuntimeError("constraint-first diagnostics do not contain exactly 128 items")
    artifact_id = diagnostics.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise RuntimeError("constraint-first diagnostics are missing an artifact id")
    diagnostics_dir = run_dir / f"eval-{checkpoint}" / "diagnostics" / artifact_id
    if "diagnostics_dir" in summary and _resolve_declared_artifact_path(summary["diagnostics_dir"]) != diagnostics_dir.resolve():
        raise RuntimeError("constraint-first diagnostic directory is not bound to this run/checkpoint")
    diagnostic_items_path = diagnostics_dir / "items.jsonl"
    diagnostic_summary_path = diagnostics_dir / "summary.json"
    diagnostic_rows = read_jsonl(diagnostic_items_path)
    if len(diagnostic_rows) != CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS:
        raise RuntimeError("constraint-first raw diagnostic artifact does not contain exactly 128 items")
    rescored_rows, rescored_summary = score_probe_outputs(
        context["diagnostic_inputs"],
        [{"raw_output": row.get("raw_output", "")} for row in diagnostic_rows],
        context["allowed"],
        context["universe"],
    )
    if diagnostic_rows != rescored_rows:
        raise RuntimeError("constraint-first raw diagnostic rows differ from the frozen rescoring contract")
    expected_diagnostic_summary = {
        **rescored_summary,
        "artifact_id": artifact_id,
        "items_sha256": sha256_file(diagnostic_items_path),
    }
    if read_json(diagnostic_summary_path) != expected_diagnostic_summary or diagnostics != expected_diagnostic_summary:
        raise RuntimeError("constraint-first diagnostic summary differs from recomputed raw outputs")

    retention_rows = read_jsonl(retention_path)
    retention_probes = context["retention_probes"]
    if len(retention_rows) != CONSTRAINT_FIRST_RETENTION_PROBES:
        raise RuntimeError("constraint-first retention artifact does not contain exactly 200 probes")
    categories = sorted({probe["category"] for probe in retention_probes})
    for probe, row in zip(retention_probes, retention_rows):
        if any(row.get(key) != probe.get(key) for key in ("probe_id", "category", "prompt", "expected")):
            raise RuntimeError("constraint-first retention artifact is not bound to the frozen probe set")
        normalized = str(row.get("raw_output", "")).lower().rstrip(".").strip()
        if row.get("normalized_output") != normalized or row.get("correct") != (normalized == probe["expected"]):
            raise RuntimeError("constraint-first retention result does not match its raw output")
    expected_retention = {
        "probe_count": CONSTRAINT_FIRST_RETENTION_PROBES,
        "overall_score": sum(row["correct"] for row in retention_rows) / CONSTRAINT_FIRST_RETENTION_PROBES,
        "category_scores": {
            category: sum(row["correct"] for row in retention_rows if row["category"] == category)
            / sum(row["category"] == category for row in retention_rows)
            for category in categories
        },
    }
    if summary.get("retention") != expected_retention:
        raise RuntimeError("constraint-first retention summary differs from recomputed raw probe results")

    artifact_integrity = {
        "games": {"rows": len(games), "sha256": sha256_file(games_path)},
        "diagnostics": {"rows": len(diagnostic_rows), "sha256": sha256_file(diagnostic_items_path)},
        "retention": {"rows": len(retention_rows), "sha256": sha256_file(retention_path)},
    }
    if "artifact_integrity" in summary and summary["artifact_integrity"] != artifact_integrity:
        raise RuntimeError("constraint-first reused summary raw-artifact hash binding mismatch")
    recomputed_gates = constraint_first_gate_status(summary)
    if summary.get("development_gates") is not None and summary["development_gates"] != recomputed_gates:
        raise RuntimeError("constraint-first reused summary development gates drift")
    return {
        "status": "passed",
        "run_id": run_dir.name,
        "checkpoint": checkpoint,
        "evaluation_input_contract": context["binding"],
        "artifact_integrity": artifact_integrity,
        "locked_test_access": False,
    }


def prepare_run(spec: dict[str, Any]) -> Path:
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"constraint-first-s{spec['seed']}-{digest}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing run at {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", read_json(DEFAULT_OUTPUT / "manifest.json"))
    return run_dir


def _unit_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing or invalid constraint-first metric: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid constraint-first metric: {label}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"constraint-first metric {label} must be finite and in [0, 1]")
    return number


def _constraint_dose_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("experiment_id") != CONSTRAINT_POLICY_ID:
        raise ValueError("constraint-first dose summary has the wrong experiment id")
    if summary.get("split") != "dev":
        raise ValueError("constraint-first dose selection requires development-only summaries")
    if summary.get("locked_test_access") is not False:
        raise ValueError("constraint-first dose summary must explicitly keep the locked test closed")
    gameplay = summary.get("gameplay")
    diagnostics = summary.get("diagnostics")
    retention = summary.get("retention")
    if not isinstance(gameplay, dict) or not isinstance(diagnostics, dict) or not isinstance(retention, dict):
        raise ValueError("constraint-first dose summary is missing gameplay, diagnostics, or retention")
    by_turn = diagnostics.get("by_turn")
    if not isinstance(by_turn, dict):
        raise ValueError("constraint-first dose summary is missing diagnostic turn metrics")
    turn_two = by_turn.get("2", by_turn.get(2))
    if not isinstance(turn_two, dict):
        raise ValueError("constraint-first dose summary is missing turn-2 diagnostics")
    wins = gameplay.get("wins")
    games = gameplay.get("n_games")
    if isinstance(wins, bool) or not isinstance(wins, int) or wins < 0:
        raise ValueError("constraint-first gameplay wins must be a nonnegative integer")
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0 or wins > games:
        raise ValueError("constraint-first gameplay n_games must be a positive integer no smaller than wins")
    metrics = {
        "terminal_marker_compliance": _unit_metric(
            gameplay.get("terminal_marker_compliance"), "gameplay.terminal_marker_compliance"
        ),
        "invalid_guess_rate": _unit_metric(gameplay.get("invalid_guess_rate"), "gameplay.invalid_guess_rate"),
        "repeat_guess_rate": _unit_metric(gameplay.get("repeat_guess_rate"), "gameplay.repeat_guess_rate"),
        "gameplay_constraint_violation_rate": _unit_metric(
            gameplay.get("constraint_violation_rate"), "gameplay.constraint_violation_rate"
        ),
        "posterior_constraint_violation_rate": _unit_metric(
            diagnostics.get("posterior_constraint_violation_rate"),
            "diagnostics.posterior_constraint_violation_rate",
        ),
        "turn_2_posterior_constraint_violation_rate": _unit_metric(
            turn_two.get("posterior_constraint_violation_rate"),
            "diagnostics.by_turn.2.posterior_constraint_violation_rate",
        ),
        "singleton_answer_accuracy": _unit_metric(
            diagnostics.get("singleton_answer_accuracy"), "diagnostics.singleton_answer_accuracy"
        ),
        "retention_overall_score": _unit_metric(retention.get("overall_score"), "retention.overall_score"),
        "win_rate": _unit_metric(gameplay.get("win_rate"), "gameplay.win_rate"),
        "wins": wins,
        "n_games": games,
    }
    for optional in ("action_target_accuracy", "posterior_consistency"):
        if diagnostics.get(optional) is not None:
            metrics[optional] = _unit_metric(diagnostics[optional], f"diagnostics.{optional}")
    return metrics


def constraint_first_gate_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply every explicitly post-hoc single-dose development gate."""
    metrics = _constraint_dose_metrics(summary)
    checks: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[bool]] = defaultdict(list)
    failures: list[str] = []
    for metric, rule in CONSTRAINT_FIRST_GATE_THRESHOLDS.items():
        observed = metrics[metric]
        if rule["op"] == ">=":
            passed = observed >= rule["value"]
        elif rule["op"] == "<=":
            passed = observed <= rule["value"]
        elif rule["op"] == "<":
            passed = observed < rule["value"]
        else:  # pragma: no cover - frozen constants make this unreachable
            raise ValueError(f"unsupported constraint-first gate operator: {rule['op']}")
        checks[metric] = {"observed": observed, **rule, "passed": passed}
        grouped[rule["group"]].append(passed)
        if not passed:
            failures.append(f"threshold_failed:{metric}")
    return {
        "schema_version": "constraint-first-development-gates-v1",
        "registration_status": "post_hoc_after_training_started",
        "preregistered": False,
        "passed": not failures,
        "checks": checks,
        "groups": {group: all(group_checks) for group, group_checks in sorted(grouped.items())},
        "failures": failures,
        "metrics": metrics,
        "locked_test_access": False,
    }


def aggregate_constraint_first_doses(
    summaries: Sequence[dict[str, Any]],
    *,
    checkpoints: Sequence[str] = CONSTRAINT_FIRST_CHECKPOINTS,
) -> dict[str, Any]:
    """Select one dose deterministically, or record that none is promotable."""
    expected = tuple(checkpoints)
    if expected != CONSTRAINT_FIRST_CHECKPOINTS:
        raise ValueError(f"constraint-first dose set must be exactly {list(CONSTRAINT_FIRST_CHECKPOINTS)}")
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        checkpoint = summary.get("checkpoint")
        if checkpoint not in expected:
            raise ValueError(f"unexpected constraint-first checkpoint summary: {checkpoint}")
        if checkpoint in by_checkpoint:
            raise ValueError(f"duplicate constraint-first checkpoint summary: {checkpoint}")
        by_checkpoint[str(checkpoint)] = summary
    missing = [checkpoint for checkpoint in expected if checkpoint not in by_checkpoint]
    if missing:
        raise ValueError(f"missing constraint-first checkpoint summaries: {missing}")

    doses: list[dict[str, Any]] = []
    for checkpoint in expected:
        summary = by_checkpoint[checkpoint]
        gates = constraint_first_gate_status(summary)
        if summary.get("development_gates") is not None and summary["development_gates"] != gates:
            raise ValueError(f"stored constraint-first gates drift for {checkpoint}")
        doses.append({"checkpoint": checkpoint, "development_gates": gates})

    passing = [dose for dose in doses if dose["development_gates"]["passed"]]

    def selection_key(dose: dict[str, Any]) -> tuple[Any, ...]:
        metrics = dose["development_gates"]["metrics"]
        return (
            metrics["posterior_constraint_violation_rate"],
            metrics["turn_2_posterior_constraint_violation_rate"],
            -metrics["singleton_answer_accuracy"],
            -metrics["terminal_marker_compliance"],
            -metrics["retention_overall_score"],
            -metrics["win_rate"],
            metrics["gameplay_constraint_violation_rate"],
            metrics["repeat_guess_rate"],
            expected.index(dose["checkpoint"]),
        )

    ranked = sorted(passing, key=selection_key)
    selected = ranked[0]["checkpoint"] if ranked else None
    promotable = [dose["checkpoint"] for dose in ranked]
    return {
        "schema_version": "constraint-first-dose-evaluation-v1",
        "status": "evaluation_complete",
        "experiment_id": CONSTRAINT_POLICY_ID,
        "evaluation_policy": constraint_first_evaluation_policy(),
        "split": "dev",
        "single_seed": True,
        "checkpoints": list(expected),
        "thresholds": CONSTRAINT_FIRST_GATE_THRESHOLDS,
        "selection_contract": {
            "registration_status": "post_hoc_after_training_started",
            "preregistered": False,
            "singleton_correctness_mandatory": True,
            "rank_order": [
                "posterior_constraint_violation_rate ascending",
                "turn_2_posterior_constraint_violation_rate ascending",
                "singleton_answer_accuracy descending",
                "terminal_marker_compliance descending",
                "retention_overall_score descending",
                "win_rate descending",
                "gameplay_constraint_violation_rate ascending",
                "repeat_guess_rate ascending",
                "earlier checkpoint dose",
            ],
            "scope": "post-hoc single-seed development selection for replication only",
        },
        "doses": doses,
        "promotable_checkpoints": promotable,
        "selected_checkpoint": selected,
        "replication_allowed": selected is not None,
        "decision": (
            "development_gates_passed_replication_candidate_locked_test_closed"
            if selected is not None
            else "development_gates_failed_no_promotable_checkpoint_locked_test_closed"
        ),
        "locked_test_access": False,
        "locked_test_authorized": False,
    }


def evaluate_constraint_checkpoint(
    run_dir: Path,
    checkpoint: str = "step-000600",
    *,
    source_dir: Path = DEFAULT_SOURCE,
    dev_games: int = 32,
) -> dict[str, Any]:
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    if checkpoint not in CONSTRAINT_FIRST_CHECKPOINTS:
        raise RuntimeError(f"unexpected constraint-first checkpoint: {checkpoint}")
    context = _constraint_evaluation_context(run_dir, source_dir, dev_games)
    spec = context["spec"]
    checkpoint_dir = run_dir / "checkpoints" / checkpoint
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_adapter(checkpoint_dir)
    universe = context["universe"]
    dev_answers = context["dev_answers"]
    allowed = context["allowed"]
    previous_messages = generation.inference_messages
    previous_config = dict(generation.GENERATION_CONFIG)
    try:
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update({"do_sample": False, "max_new_tokens": 128, "use_cache": True})
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            context["probes"],
            context["training_records"],
            allowed,
            universe,
            run_dir / f"eval-{checkpoint}",
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, context["retention_probes"])
        games_path = write_jsonl(run_dir / f"eval-{checkpoint}-games.jsonl", games)
        retention_path = write_jsonl(run_dir / f"eval-{checkpoint}-retention.jsonl", retention_rows)
        diagnostic_items_path = Path(diagnostics_dir) / "items.jsonl"
        summary = {
            "status": "dev_evaluated",
            "experiment_id": CONSTRAINT_POLICY_ID,
            "run_id": run_dir.name,
            "checkpoint": checkpoint,
            "split": "dev",
            "locked_test_access": False,
            "protocol_id": spec["protocol_id"],
            "protocol_sha256": spec["protocol_sha256"],
            "protocol_lock_file_sha256": spec["protocol_lock_file_sha256"],
            "evaluation_contract": spec["evaluation_contract"],
            "evaluation_input_contract": context["binding"],
            "evaluation_policy": constraint_first_evaluation_policy(),
            "selection_metric": "posterior consistency with singleton correctness mandatory",
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
            "artifact_integrity": {
                "games": {"rows": len(games), "sha256": sha256_file(games_path)},
                "diagnostics": {
                    "rows": diagnostics["items"],
                    "sha256": sha256_file(diagnostic_items_path),
                },
                "retention": {"rows": len(retention_rows), "sha256": sha256_file(retention_path)},
            },
        }
        summary["development_gates"] = constraint_first_gate_status(summary)
        write_json(run_dir / f"eval-{checkpoint}-summary.json", summary)
        validate_reused_constraint_summary(
            run_dir,
            checkpoint,
            summary,
            source_dir=source_dir,
            dev_games=dev_games,
            evaluation_context=context,
        )
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_config)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_constraint_doses(
    run_dir: Path,
    *,
    source_dir: Path = DEFAULT_SOURCE,
    dev_games: int = 32,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Evaluate all frozen doses and write one deterministic development decision."""
    run_dir = Path(run_dir)
    source_dir = _require_frozen_constraint_evaluation_request(source_dir, dev_games)
    summaries: list[dict[str, Any]] = []
    reused_integrity: list[dict[str, Any]] = []
    context: dict[str, Any] | None = None
    for checkpoint in CONSTRAINT_FIRST_CHECKPOINTS:
        summary_path = run_dir / f"eval-{checkpoint}-summary.json"
        if reuse_existing and summary_path.is_file():
            summary = read_json(summary_path)
            if context is None:
                context = _constraint_evaluation_context(run_dir, source_dir, dev_games)
            reused_integrity.append(
                validate_reused_constraint_summary(
                    run_dir,
                    checkpoint,
                    summary,
                    source_dir=source_dir,
                    dev_games=dev_games,
                    evaluation_context=context,
                )
            )
        else:
            summary = evaluate_constraint_checkpoint(
                run_dir,
                checkpoint,
                source_dir=source_dir,
                dev_games=dev_games,
            )
        summaries.append(summary)
    aggregate = aggregate_constraint_first_doses(summaries)
    aggregate["run_dir"] = str(run_dir.resolve())
    aggregate["reused_summary_integrity"] = reused_integrity
    write_json(run_dir / "evaluation_summary.json", aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or evaluate the legality-first Wordle policy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--force", action="store_true")
    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("--steps", type=int, default=600)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", default="step-000600", choices=CONSTRAINT_FIRST_CHECKPOINTS)
    eval_parser.add_argument("--dev-games", type=int, default=32)
    eval_all_parser = subparsers.add_parser("evaluate-all")
    eval_all_parser.add_argument("--run-dir", type=Path, required=True)
    eval_all_parser.add_argument("--dev-games", type=int, default=32)
    eval_all_parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_constraint_first_bundle(args.source_dir, args.output_dir, force=args.force)
    elif args.command == "dry-run":
        build_constraint_first_bundle()
        result = {
            "status": "dry_run_passed",
            "spec": constraint_policy_spec(steps=args.steps),
            "evaluation_policy": constraint_first_evaluation_policy(),
        }
    elif args.command == "evaluate":
        result = evaluate_constraint_checkpoint(args.run_dir, args.checkpoint, dev_games=args.dev_games)
    else:
        result = evaluate_constraint_doses(
            args.run_dir,
            dev_games=args.dev_games,
            reuse_existing=not args.reevaluate,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
