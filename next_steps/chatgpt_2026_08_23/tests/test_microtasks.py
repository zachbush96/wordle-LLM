from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.microtasks import (
    INVALID_REASONS,
    CurriculumRecord,
    TrainingContext,
    TrainingState,
    assess_candidate,
    audit_curriculum_records,
    build_balanced_candidate_validity_records,
    build_constraint_merge_record,
    build_feedback_decode_record,
    build_full_policy_record,
    build_singleton_record,
    decode_feedback,
    evaluate_curriculum_predictions,
    merge_constraints,
)
from wordle_lab.common import sha256_file
from wordle_lab.protocol.env import score_wordle


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DATA = ROOT / "data" / "gemma-270m-unsloth-alpaca-v2" / "u160-train120-n2000"


@pytest.fixture(scope="module")
def training_words() -> tuple[str, ...]:
    return tuple(json.loads((SOURCE_DATA / "train_secrets.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def context(training_words: tuple[str, ...]) -> TrainingContext:
    return TrainingContext(
        source_dataset_id="GEMMA-270M-UNSLOTH-ALPACA-002",
        source_manifest_sha256=sha256_file(SOURCE_DATA / "manifest.json"),
        training_secret_source="data/gemma-270m-unsloth-alpaca-v2/u160-train120-n2000/train_secrets.json",
        training_secrets=training_words,
        # Microtasks use only the training-answer universe. Development and
        # locked-test answer files are deliberately never loaded here.
        answer_universe=training_words,
    )


def validation_states() -> list[TrainingState]:
    return [
        TrainingState("validity-root", "ACRID", ()),
        TrainingState(
            "validity-adapt",
            "ACRID",
            (("ADAPT", score_wordle("ACRID", "ADAPT")),),
        ),
        TrainingState(
            "validity-adopt",
            "ACRID",
            (("ADOPT", score_wordle("ACRID", "ADOPT")),),
        ),
    ]


def singleton_state() -> TrainingState:
    return TrainingState(
        "singleton-cable",
        "CABLE",
        (
            ("ALLEY", score_wordle("CABLE", "ALLEY")),
            ("CABIN", score_wordle("CABLE", "CABIN")),
        ),
    )


def test_one_row_feedback_decoding_handles_duplicate_letters() -> None:
    # CABLE vs ALLEY is YYBYB: one L is yellow and the second L is gray.
    # That means exactly one L, not zero L and not two Ls.
    assert score_wordle("CABLE", "ALLEY") == "YYBYB"
    decoded = decode_feedback("ALLEY", "YYBYB").to_dict()
    assert decoded == {
        "fixed": [None, None, None, None, None],
        "forbidden": {"A": [1], "E": [4], "L": [2, 3], "Y": [5]},
        "min_counts": {"A": 1, "E": 1, "L": 1},
        "max_counts": {"L": 1, "Y": 0},
        "excluded": ["Y"],
        "position_evidence": {
            "yellow": {"A": [1], "E": [4], "L": [2]},
            "gray": {"L": [3], "Y": [5]},
        },
    }


def test_multi_turn_merge_is_deterministic_and_solves_combined_constraints() -> None:
    first = decode_feedback("ALLEY", score_wordle("CABLE", "ALLEY"))
    second = decode_feedback("CABIN", score_wordle("CABLE", "CABIN"))
    forward = merge_constraints([first, second])
    reverse = merge_constraints([second, first])
    assert forward == reverse
    assert forward.fixed == ("C", "A", "B", None, None)
    assert forward.minimums == {"A": 1, "B": 1, "C": 1, "E": 1, "L": 1}
    assert forward.maximums == {"I": 0, "L": 1, "N": 0, "Y": 0}
    assert assess_candidate("CABLE", singleton_state().history, forward).valid is True


def test_candidate_reason_taxonomy_uses_training_words_and_frozen_scorer(
    training_words: tuple[str, ...],
) -> None:
    assert {"ACRID", "ADAPT", "ADOPT", "AHEAD", "ALIEN", "BADLY", "AVOID"} <= set(training_words)
    adapt_history = (("ADAPT", score_wordle("ACRID", "ADAPT")),)
    adopt_history = (("ADOPT", score_wordle("ACRID", "ADOPT")),)
    assert assess_candidate("BADLY", adapt_history).reason == "green"
    assert assess_candidate("ADOPT", adapt_history).reason == "yellow"
    assert assess_candidate("ALIEN", adapt_history).reason == "missing-required"
    assert assess_candidate("AVOID", adopt_history).reason == "gray"
    assert assess_candidate("AHEAD", adapt_history).reason == "duplicate-count"
    assert assess_candidate("ADAPT", adapt_history).reason == "repeated-guess"
    assert assess_candidate("ACRID", adapt_history).valid is True


def test_balanced_candidate_builder_is_exact_and_deterministic(
    context: TrainingContext,
    training_words: tuple[str, ...],
) -> None:
    states = validation_states()
    rows = build_balanced_candidate_validity_records(
        states,
        context,
        training_words,
        per_invalid_reason=1,
        seed=2026,
    )
    repeated = build_balanced_candidate_validity_records(
        list(reversed(states)),
        context,
        list(reversed(training_words)),
        per_invalid_reason=1,
        seed=2026,
    )
    assert [row.to_dict() for row in rows] == [row.to_dict() for row in repeated]
    assert len(rows) == 12
    assert sum(row.target["valid"] for row in rows) == 6
    reasons = Counter(row.target["reason"] for row in rows if not row.target["valid"])
    assert reasons == Counter({reason: 1 for reason in INVALID_REASONS})
    assert {row.input["candidate"] for row in rows} <= set(training_words)


def test_singleton_target_is_derived_from_public_history(
    context: TrainingContext,
) -> None:
    record = build_singleton_record(singleton_state(), context)
    assert record.target == {"word": "CABLE"}
    assert record.metadata == {"posterior_size": 1, "target_derivation": "visible_history_only"}
    rendered = json.dumps(record.to_dict(), sort_keys=True)
    assert "secret_answer" not in rendered
    assert "posterior_candidates" not in rendered


def test_mixed_curriculum_accepts_microtasks_and_full_policy_rows(
    context: TrainingContext,
    training_words: tuple[str, ...],
) -> None:
    validity_states = validation_states()
    singleton = singleton_state()
    candidate_rows = build_balanced_candidate_validity_records(
        validity_states,
        context,
        training_words,
        per_invalid_reason=1,
    )
    rows = [
        *candidate_rows,
        build_feedback_decode_record(singleton, context, 0),
        build_constraint_merge_record(singleton, context),
        build_singleton_record(singleton, context),
        build_full_policy_record(singleton, context, "CABLE", policy_id="training-oracle-v1"),
    ]
    manifest = audit_curriculum_records(rows, [*validity_states, singleton], context)
    assert manifest["status"] == "passed"
    assert manifest["locked_test_access"] is False
    assert manifest["records"] == 16
    assert manifest["task_distribution"] == {
        "candidate_validity": 12,
        "constraint_merge": 1,
        "feedback_decode": 1,
        "full_policy": 1,
        "singleton_solve": 1,
    }
    assert manifest["candidate_balance"] == {
        "valid": 6,
        "invalid": 6,
        "invalid_reasons": {reason: 1 for reason in INVALID_REASONS},
    }
    assert all(len(value) == 64 for value in manifest["hashes"].values())


def test_audit_rejects_label_tampering_and_nontraining_sources(
    context: TrainingContext,
) -> None:
    state = singleton_state()
    record = build_singleton_record(state, context)
    tampered = replace(record, target={"word": "BOOBY"}, record_id="pending")
    tampered = replace(tampered, record_id=tampered.expected_record_id)
    with pytest.raises(AssertionError, match="target mismatch"):
        audit_curriculum_records([tampered], [state], context)

    outside = TrainingState("outside-training-split", "ZZZZZ", ())
    with pytest.raises(ValueError, match="declared training secret"):
        build_constraint_merge_record(outside, context)


def test_full_policy_row_rejects_feedback_inconsistent_target(
    context: TrainingContext,
) -> None:
    with pytest.raises(ValueError, match="violates visible feedback"):
        build_full_policy_record(singleton_state(), context, "BOOBY", policy_id="bad-policy")


def test_prediction_evaluator_reports_task_and_invalid_reason_accuracy(
    context: TrainingContext,
    training_words: tuple[str, ...],
) -> None:
    rows = build_balanced_candidate_validity_records(
        validation_states(),
        context,
        training_words,
        per_invalid_reason=1,
    )
    singleton = build_singleton_record(singleton_state(), context)
    rows.append(singleton)
    perfect = {row.record_id: row.target for row in rows}
    metrics = evaluate_curriculum_predictions(rows, perfect)
    assert metrics["coverage"] == metrics["accuracy"] == 1.0
    assert metrics["by_task"]["singleton_solve"]["accuracy"] == 1.0
    assert all(
        metrics["candidate_invalid_reason_accuracy"][reason]["accuracy"] == 1.0
        for reason in INVALID_REASONS
    )

    invalid = next(row for row in rows if row.task_type == "candidate_validity" and not row.target["valid"])
    incomplete = dict(perfect)
    incomplete[invalid.record_id] = False  # correct validity, but no causal reason
    degraded = evaluate_curriculum_predictions(rows, incomplete)
    assert degraded["correct"] == len(rows) - 1
    assert degraded["accuracy"] < 1.0


def test_record_type_rejects_unknown_task_during_audit(context: TrainingContext) -> None:
    state = singleton_state()
    record = build_singleton_record(state, context)
    unknown = CurriculumRecord(**{**record.__dict__, "task_type": "unknown"})
    with pytest.raises(AssertionError, match="unsupported schema/task"):
        audit_curriculum_records([unknown], [state], context)
