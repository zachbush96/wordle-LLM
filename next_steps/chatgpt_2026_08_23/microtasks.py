from __future__ import annotations

"""Structured Wordle microtask data, labels, audits, and evaluation.

The hidden answer is used only to audit that a source state was generated from
an explicitly supplied training split. Labels are derived from visible
guess/feedback history with the frozen protocol scorer. No candidate list or
secret is rendered into a curriculum record.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wordle_lab.common import canonical_json, sha256_text
from wordle_lab.protocol.env import (
    ALL_GREEN,
    is_five_ascii_letters,
    normalize_word,
    posterior_candidates,
    score_wordle,
)
from wordle_lab.protocol.lock import PROTOCOL_ID

BUILDER_VERSION = "WORDLE-STRUCTURED-MICROTASKS-001"
SCHEMA_VERSION = "wordle-structured-curriculum-v1"
TASK_TYPES = (
    "feedback_decode",
    "constraint_merge",
    "candidate_validity",
    "singleton_solve",
    "full_policy",
)
INVALID_REASONS = (
    "green",
    "yellow",
    "missing-required",
    "gray",
    "duplicate-count",
    "repeated-guess",
)
# Repetition is a policy failure even when a repeated proposal has additional
# clue violations. The remaining order follows the declared task taxonomy.
_PRIMARY_REASON_ORDER = ("repeated-guess",) + INVALID_REASONS[:-1]
_HEX = frozenset("0123456789abcdef")


def _hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value.lower()) <= _HEX


def _normalize_feedback(feedback: str) -> str:
    normalized = feedback.strip().upper()
    if len(normalized) != 5 or set(normalized) - {"G", "Y", "B"}:
        raise ValueError("feedback must contain exactly five G/Y/B symbols")
    return normalized


def _normalize_history(history: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for guess, feedback in history:
        word = normalize_word(guess)
        if not is_five_ascii_letters(word):
            raise ValueError("history guesses must be five ASCII letters")
        rows.append((word, _normalize_feedback(feedback)))
    return tuple(rows)


def _history_payload(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"guess": guess, "feedback": feedback} for guess, feedback in history]


def _pairs(mapping: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(mapping.items()))


def _position_pairs(mapping: Mapping[str, Sequence[int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((letter, tuple(sorted(set(positions)))) for letter, positions in sorted(mapping.items()))


@dataclass(frozen=True)
class FeedbackConstraints:
    """Canonical constraints inferred only from visible Wordle feedback.

    Positions are one-based in serialized labels. ``yellow_forbidden`` and
    ``gray_forbidden`` retain evidence needed for deterministic error labels;
    ``forbidden`` is their union and is the compact constraint used by models.
    """

    fixed: tuple[str | None, ...]
    forbidden: tuple[tuple[str, tuple[int, ...]], ...]
    min_counts: tuple[tuple[str, int], ...]
    max_counts: tuple[tuple[str, int], ...]
    excluded: tuple[str, ...]
    yellow_forbidden: tuple[tuple[str, tuple[int, ...]], ...] = ()
    gray_forbidden: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if len(self.fixed) != 5:
            raise ValueError("fixed must contain five positions")
        if any(letter is not None and (len(letter) != 1 or not letter.isascii() or not letter.isalpha()) for letter in self.fixed):
            raise ValueError("fixed positions must contain uppercase ASCII letters or null")
        for _, positions in self.forbidden + self.yellow_forbidden + self.gray_forbidden:
            if any(position not in range(1, 6) for position in positions):
                raise ValueError("forbidden positions must be one-based values from 1 through 5")

    @property
    def forbidden_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.forbidden)

    @property
    def yellow_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.yellow_forbidden)

    @property
    def gray_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.gray_forbidden)

    @property
    def minimums(self) -> dict[str, int]:
        return dict(self.min_counts)

    @property
    def maximums(self) -> dict[str, int]:
        return dict(self.max_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed": list(self.fixed),
            "forbidden": {letter: list(positions) for letter, positions in self.forbidden},
            "min_counts": dict(self.min_counts),
            "max_counts": dict(self.max_counts),
            "excluded": list(self.excluded),
            "position_evidence": {
                "yellow": {letter: list(positions) for letter, positions in self.yellow_forbidden},
                "gray": {letter: list(positions) for letter, positions in self.gray_forbidden},
            },
        }


def _constraints(
    *,
    fixed: Sequence[str | None],
    forbidden: Mapping[str, Sequence[int]],
    min_counts: Mapping[str, int],
    max_counts: Mapping[str, int],
    yellow_forbidden: Mapping[str, Sequence[int]],
    gray_forbidden: Mapping[str, Sequence[int]],
) -> FeedbackConstraints:
    excluded = tuple(sorted(letter for letter, count in max_counts.items() if count == 0))
    return FeedbackConstraints(
        fixed=tuple(fixed),
        forbidden=_position_pairs(forbidden),
        min_counts=tuple((letter, int(count)) for letter, count in sorted(min_counts.items()) if count > 0),
        max_counts=tuple((letter, int(count)) for letter, count in sorted(max_counts.items())),
        excluded=excluded,
        yellow_forbidden=_position_pairs(yellow_forbidden),
        gray_forbidden=_position_pairs(gray_forbidden),
    )


def decode_feedback(guess: str, feedback: str) -> FeedbackConstraints:
    """Decode one visible row, including exact duplicate-letter bounds."""

    word = normalize_word(guess)
    marks = _normalize_feedback(feedback)
    if not is_five_ascii_letters(word):
        raise ValueError("guess must be five ASCII letters")

    fixed: list[str | None] = [None] * 5
    forbidden: dict[str, set[int]] = defaultdict(set)
    yellow: dict[str, set[int]] = defaultdict(set)
    gray: dict[str, set[int]] = defaultdict(set)
    positive_counts: Counter[str] = Counter()
    gray_counts: Counter[str] = Counter()

    for position, (letter, mark) in enumerate(zip(word, marks), start=1):
        if mark == "G":
            fixed[position - 1] = letter
            positive_counts[letter] += 1
        else:
            forbidden[letter].add(position)
            if mark == "Y":
                yellow[letter].add(position)
                positive_counts[letter] += 1
            else:
                gray[letter].add(position)
                gray_counts[letter] += 1

    min_counts = {letter: count for letter, count in positive_counts.items() if count}
    # A gray copy after a green/yellow copy is an exact upper bound, not proof
    # that the letter is absent. If every copy is gray the upper bound is zero.
    max_counts = {
        letter: positive_counts[letter]
        for letter in sorted(gray_counts)
    }
    return _constraints(
        fixed=fixed,
        forbidden=forbidden,
        min_counts=min_counts,
        max_counts=max_counts,
        yellow_forbidden=yellow,
        gray_forbidden=gray,
    )


def merge_constraints(rows: Sequence[FeedbackConstraints]) -> FeedbackConstraints:
    """Merge row constraints, rejecting contradictory histories."""

    fixed: list[str | None] = [None] * 5
    forbidden: dict[str, set[int]] = defaultdict(set)
    yellow: dict[str, set[int]] = defaultdict(set)
    gray: dict[str, set[int]] = defaultdict(set)
    minimums: dict[str, int] = {}
    maximums: dict[str, int] = {}

    for row in rows:
        for index, letter in enumerate(row.fixed):
            if letter is None:
                continue
            if fixed[index] not in {None, letter}:
                raise ValueError(f"conflicting green constraints at position {index + 1}")
            fixed[index] = letter
        for letter, positions in row.forbidden:
            forbidden[letter].update(positions)
        for letter, positions in row.yellow_forbidden:
            yellow[letter].update(positions)
        for letter, positions in row.gray_forbidden:
            gray[letter].update(positions)
        for letter, count in row.min_counts:
            minimums[letter] = max(minimums.get(letter, 0), count)
        for letter, count in row.max_counts:
            maximums[letter] = min(maximums.get(letter, 5), count)

    # Separate rows can reveal distinct green copies of the same letter.
    for letter, count in Counter(letter for letter in fixed if letter is not None).items():
        minimums[letter] = max(minimums.get(letter, 0), count)

    if sum(minimums.values()) > 5:
        raise ValueError("merged minimum letter counts exceed the five available positions")
    for index, letter in enumerate(fixed, start=1):
        if letter is not None and index in forbidden.get(letter, set()):
            raise ValueError(f"position {index} is both fixed and forbidden for {letter}")
    for letter in set(minimums) | set(maximums):
        if minimums.get(letter, 0) > maximums.get(letter, 5):
            raise ValueError(f"minimum count exceeds maximum count for {letter}")

    return _constraints(
        fixed=fixed,
        forbidden=forbidden,
        min_counts=minimums,
        max_counts=maximums,
        yellow_forbidden=yellow,
        gray_forbidden=gray,
    )


def constraints_from_history(history: Sequence[tuple[str, str]]) -> FeedbackConstraints:
    normalized = _normalize_history(history)
    return merge_constraints([decode_feedback(guess, feedback) for guess, feedback in normalized])


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: str
    valid: bool
    reason: str | None
    violations: tuple[str, ...]
    protocol_consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "valid": self.valid,
            "reason": self.reason,
            "violations": list(self.violations),
            "protocol_consistent": self.protocol_consistent,
        }


def assess_candidate(
    candidate: str,
    history: Sequence[tuple[str, str]],
    constraints: FeedbackConstraints | None = None,
) -> CandidateAssessment:
    """Classify a proposed answer candidate against visible history.

    The frozen scorer independently verifies that the compact constraints are
    sufficient, including duplicate-letter cases.
    """

    word = normalize_word(candidate)
    if not is_five_ascii_letters(word):
        raise ValueError("candidate must be five ASCII letters")
    normalized_history = _normalize_history(history)
    merged = constraints or constraints_from_history(normalized_history)
    counts = Counter(word)
    violations: set[str] = set()

    if word in {guess for guess, _ in normalized_history}:
        violations.add("repeated-guess")
    if any(letter is not None and word[index] != letter for index, letter in enumerate(merged.fixed)):
        violations.add("green")
    if any(word[position - 1] == letter for letter, positions in merged.yellow_forbidden for position in positions):
        violations.add("yellow")
    if any(counts[letter] < minimum for letter, minimum in merged.min_counts):
        violations.add("missing-required")
    if any(counts[letter] for letter in merged.excluded):
        violations.add("gray")
    if any(counts[letter] > maximum for letter, maximum in merged.max_counts if maximum > 0):
        violations.add("duplicate-count")
    if any(
        maximum > 0 and any(word[position - 1] == letter for position in merged.gray_map.get(letter, ()))
        for letter, maximum in merged.max_counts
    ):
        violations.add("duplicate-count")

    protocol_consistent = all(score_wordle(word, guess) == feedback for guess, feedback in normalized_history)
    semantic_violations = violations - {"repeated-guess"}
    if protocol_consistent != (not semantic_violations):
        raise AssertionError("structured constraints disagree with the frozen Wordle scorer")
    valid = protocol_consistent and "repeated-guess" not in violations
    ordered = tuple(reason for reason in _PRIMARY_REASON_ORDER if reason in violations)
    return CandidateAssessment(
        candidate=word,
        valid=valid,
        reason=None if valid else ordered[0],
        violations=ordered,
        protocol_consistent=protocol_consistent,
    )


@dataclass(frozen=True)
class TrainingState:
    state_id: str
    secret_answer: str
    history: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.state_id.strip():
            raise ValueError("state_id must be non-empty")
        secret = normalize_word(self.secret_answer)
        if not is_five_ascii_letters(secret):
            raise ValueError("secret_answer must be five ASCII letters")
        object.__setattr__(self, "secret_answer", secret)
        object.__setattr__(self, "history", _normalize_history(self.history))

    def private_payload(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "secret_answer": self.secret_answer,
            "history": _history_payload(self.history),
        }

    def public_payload(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "history": _history_payload(self.history)}

    @property
    def sha256(self) -> str:
        return _hash(self.private_payload())


@dataclass(frozen=True)
class TrainingContext:
    source_dataset_id: str
    source_manifest_sha256: str
    training_secret_source: str
    training_secrets: tuple[str, ...]
    answer_universe: tuple[str, ...]
    protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError(f"this layer is frozen to {PROTOCOL_ID}")
        if not self.source_dataset_id.strip() or not self.training_secret_source.strip():
            raise ValueError("dataset and training-secret source identifiers are required")
        digest = self.source_manifest_sha256.lower()
        if not _is_sha256(digest):
            raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")
        training = tuple(sorted({normalize_word(word) for word in self.training_secrets}))
        universe = tuple(sorted({normalize_word(word) for word in self.answer_universe}))
        if not training or not universe:
            raise ValueError("training_secrets and answer_universe must be non-empty")
        if any(not is_five_ascii_letters(word) for word in training + universe):
            raise ValueError("secret sets must contain only five-letter ASCII words")
        if not set(training) <= set(universe):
            raise ValueError("training_secrets must be a subset of answer_universe")
        object.__setattr__(self, "source_manifest_sha256", digest)
        object.__setattr__(self, "training_secrets", training)
        object.__setattr__(self, "answer_universe", universe)

    @property
    def training_secret_set_sha256(self) -> str:
        return _hash(list(self.training_secrets))

    @property
    def answer_universe_sha256(self) -> str:
        return _hash(list(self.answer_universe))


def _validate_training_state(state: TrainingState, context: TrainingContext) -> None:
    if state.secret_answer not in context.training_secrets:
        raise ValueError(f"state {state.state_id} does not use a declared training secret")
    if any(feedback == ALL_GREEN for _, feedback in state.history):
        raise ValueError(f"state {state.state_id} continues after a solved row")
    for guess, feedback in state.history:
        if score_wordle(state.secret_answer, guess) != feedback:
            raise ValueError(f"state {state.state_id} contains incorrect feedback")
    if state.secret_answer not in posterior_candidates(state.history, context.answer_universe):
        raise ValueError(f"state {state.state_id} removes its own training secret")


def _provenance(state: TrainingState, context: TrainingContext) -> dict[str, Any]:
    _validate_training_state(state, context)
    return {
        "protocol_id": context.protocol_id,
        "builder_version": BUILDER_VERSION,
        "split": "train",
        "secret_scope": "training_only",
        "source_dataset_id": context.source_dataset_id,
        "source_manifest_sha256": context.source_manifest_sha256,
        "training_secret_source": context.training_secret_source,
        "training_secret_set_sha256": context.training_secret_set_sha256,
        "source_state_sha256": state.sha256,
        "locked_test_access": False,
    }


@dataclass(frozen=True)
class CurriculumRecord:
    record_id: str
    task_type: str
    source_state_id: str
    input: dict[str, Any]
    target: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "source_state_id": self.source_state_id,
            "input": self.input,
            "target": self.target,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, **self.content_payload()}

    @property
    def expected_record_id(self) -> str:
        return f"{self.task_type}-{_hash(self.content_payload())[:24]}"


def _record(
    task_type: str,
    state: TrainingState,
    context: TrainingContext,
    *,
    input_payload: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> CurriculumRecord:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type: {task_type}")
    record = CurriculumRecord(
        record_id="pending",
        task_type=task_type,
        source_state_id=state.state_id,
        input=input_payload,
        target=target,
        metadata=metadata or {},
        provenance=_provenance(state, context),
    )
    return CurriculumRecord(**{**record.__dict__, "record_id": record.expected_record_id})


def build_feedback_decode_record(
    state: TrainingState,
    context: TrainingContext,
    row_index: int,
) -> CurriculumRecord:
    _validate_training_state(state, context)
    if row_index not in range(len(state.history)):
        raise IndexError("row_index is outside the source history")
    guess, feedback = state.history[row_index]
    return _record(
        "feedback_decode",
        state,
        context,
        input_payload={"guess": guess, "feedback": feedback},
        target=decode_feedback(guess, feedback).to_dict(),
        metadata={"history_index": row_index, "positions_are_one_based": True},
    )


def build_constraint_merge_record(state: TrainingState, context: TrainingContext) -> CurriculumRecord:
    _validate_training_state(state, context)
    rows = [decode_feedback(guess, feedback) for guess, feedback in state.history]
    return _record(
        "constraint_merge",
        state,
        context,
        input_payload={
            "history": _history_payload(state.history),
            "row_constraints": [row.to_dict() for row in rows],
        },
        target=merge_constraints(rows).to_dict(),
        metadata={"turns_merged": len(rows), "positions_are_one_based": True},
    )


def _build_candidate_record(
    state: TrainingState,
    context: TrainingContext,
    candidate: str,
) -> CurriculumRecord:
    _validate_training_state(state, context)
    word = normalize_word(candidate)
    if word not in context.training_secrets:
        raise ValueError("candidate-validity words must come from the declared training-secret set")
    constraints = constraints_from_history(state.history)
    assessment = assess_candidate(word, state.history, constraints)
    return _record(
        "candidate_validity",
        state,
        context,
        input_payload={
            "history": _history_payload(state.history),
            "constraints": constraints.to_dict(),
            "candidate": word,
        },
        target={
            "valid": assessment.valid,
            "reason": assessment.reason,
            "violations": list(assessment.violations),
        },
        metadata={"reason_taxonomy": list(INVALID_REASONS)},
    )


def build_balanced_candidate_validity_records(
    states: Sequence[TrainingState],
    context: TrainingContext,
    candidate_words: Sequence[str],
    *,
    per_invalid_reason: int,
    seed: int = 2026,
) -> list[CurriculumRecord]:
    """Build exactly 50% valid rows and equal counts for six invalid reasons."""

    if per_invalid_reason <= 0:
        raise ValueError("per_invalid_reason must be positive")
    if not states or len({state.state_id for state in states}) != len(states):
        raise ValueError("states must be non-empty and have unique state_id values")
    candidates = tuple(sorted({normalize_word(word) for word in candidate_words}))
    if any(word not in context.training_secrets for word in candidates):
        raise ValueError("candidate_words must be drawn only from the declared training-secret set")

    valid_pool: list[tuple[TrainingState, str, tuple[str, ...]]] = []
    invalid_pools: dict[str, list[tuple[TrainingState, str, tuple[str, ...]]]] = {
        reason: [] for reason in INVALID_REASONS
    }
    for state in states:
        _validate_training_state(state, context)
        constraints = constraints_from_history(state.history)
        for candidate in candidates:
            assessment = assess_candidate(candidate, state.history, constraints)
            if assessment.valid:
                valid_pool.append((state, candidate, assessment.violations))
            else:
                invalid_pools[assessment.reason].append((state, candidate, assessment.violations))

    def ranked_pairs(
        rows: Sequence[tuple[TrainingState, str, tuple[str, ...]]], label: str
    ) -> list[tuple[TrainingState, str, tuple[str, ...]]]:
        return sorted(
            rows,
            key=lambda item: (
                # Prefer examples with one causal violation. A repeated guess is
                # allowed to violate the old clues as well because repetition is
                # itself the policy error under test.
                label != "repeated-guess" and item[2] != (label,),
                _hash({"seed": seed, "label": label, "state_id": item[0].state_id, "candidate": item[1]}),
            ),
        )

    selected: list[tuple[TrainingState, str, tuple[str, ...]]] = []
    for reason in INVALID_REASONS:
        pool = ranked_pairs(invalid_pools[reason], reason)
        if len(pool) < per_invalid_reason:
            raise ValueError(f"insufficient {reason} examples: need {per_invalid_reason}, found {len(pool)}")
        selected.extend(pool[:per_invalid_reason])
    valid_needed = per_invalid_reason * len(INVALID_REASONS)
    valid_ranked = ranked_pairs(valid_pool, "valid")
    if len(valid_ranked) < valid_needed:
        raise ValueError(f"insufficient valid examples: need {valid_needed}, found {len(valid_ranked)}")
    selected.extend(valid_ranked[:valid_needed])
    records = [_build_candidate_record(state, context, candidate) for state, candidate, _ in selected]
    return sorted(
        records,
        key=lambda row: _hash({"seed": seed, "label": "final-order", "record_id": row.record_id}),
    )


def build_singleton_record(state: TrainingState, context: TrainingContext) -> CurriculumRecord:
    _validate_training_state(state, context)
    remaining = posterior_candidates(state.history, context.answer_universe)
    if len(remaining) != 1:
        raise ValueError(f"state {state.state_id} has posterior size {len(remaining)}, not one")
    # The target is inferred from public history and the declared training-only
    # answer universe. It is never copied from secret_answer.
    target = remaining[0]
    return _record(
        "singleton_solve",
        state,
        context,
        input_payload={
            "history": _history_payload(state.history),
            "constraints": constraints_from_history(state.history).to_dict(),
        },
        target={"word": target},
        metadata={"posterior_size": 1, "target_derivation": "visible_history_only"},
    )


def build_full_policy_record(
    state: TrainingState,
    context: TrainingContext,
    target_word: str,
    *,
    policy_id: str,
) -> CurriculumRecord:
    """Wrap a natural-generation policy target in the mixed curriculum schema."""

    _validate_training_state(state, context)
    target = normalize_word(target_word)
    if target not in context.training_secrets:
        raise ValueError("full-policy targets must come from the declared training-secret set")
    assessment = assess_candidate(target, state.history)
    if not assessment.valid:
        raise ValueError(f"full-policy target violates visible feedback: {assessment.reason}")
    if not policy_id.strip():
        raise ValueError("policy_id must be non-empty")
    return _record(
        "full_policy",
        state,
        context,
        input_payload={
            "history": _history_payload(state.history),
            "constraints": constraints_from_history(state.history).to_dict(),
        },
        target={"word": target},
        metadata={"policy_id": policy_id, "generation_contract": "natural_unassisted"},
    )


def _forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    forbidden = {"secret", "secret_answer", "hidden_answer", "posterior_candidates", "candidate_words"}
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in forbidden:
                paths.append(path)
            paths.extend(_forbidden_key_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(_forbidden_key_paths(nested, f"{prefix}[{index}]"))
    return paths


def _expected_target(record: CurriculumRecord, state: TrainingState, context: TrainingContext) -> dict[str, Any]:
    history = state.history
    if record.task_type == "feedback_decode":
        index = record.metadata.get("history_index")
        if not isinstance(index, int) or index not in range(len(history)):
            raise AssertionError(f"bad history index in {record.record_id}")
        guess, feedback = history[index]
        if record.input != {"guess": guess, "feedback": feedback}:
            raise AssertionError(f"source row mismatch in {record.record_id}")
        return decode_feedback(guess, feedback).to_dict()
    if record.task_type == "constraint_merge":
        rows = [decode_feedback(guess, feedback) for guess, feedback in history]
        expected_input = {
            "history": _history_payload(history),
            "row_constraints": [row.to_dict() for row in rows],
        }
        if record.input != expected_input:
            raise AssertionError(f"merge input mismatch in {record.record_id}")
        return merge_constraints(rows).to_dict()
    if record.task_type == "candidate_validity":
        candidate = record.input.get("candidate")
        if normalize_word(str(candidate or "")) not in context.training_secrets:
            raise AssertionError(f"non-training candidate in {record.record_id}")
        if record.input.get("history") != _history_payload(history):
            raise AssertionError(f"candidate history mismatch in {record.record_id}")
        expected_constraints = constraints_from_history(history).to_dict()
        if record.input.get("constraints") != expected_constraints:
            raise AssertionError(f"candidate constraints mismatch in {record.record_id}")
        assessment = assess_candidate(candidate, history)
        return {
            "valid": assessment.valid,
            "reason": assessment.reason,
            "violations": list(assessment.violations),
        }
    if record.task_type == "singleton_solve":
        expected_input = {
            "history": _history_payload(history),
            "constraints": constraints_from_history(history).to_dict(),
        }
        if record.input != expected_input:
            raise AssertionError(f"singleton input mismatch in {record.record_id}")
        remaining = posterior_candidates(history, context.answer_universe)
        if len(remaining) != 1:
            raise AssertionError(f"non-singleton source in {record.record_id}")
        return {"word": remaining[0]}
    if record.task_type == "full_policy":
        expected_input = {
            "history": _history_payload(history),
            "constraints": constraints_from_history(history).to_dict(),
        }
        if record.input != expected_input:
            raise AssertionError(f"full-policy input mismatch in {record.record_id}")
        target = normalize_word(record.target.get("word", ""))
        if target not in context.training_secrets or not assess_candidate(target, history).valid:
            raise AssertionError(f"invalid full-policy target in {record.record_id}")
        return {"word": target}
    raise AssertionError(f"unknown task type in {record.record_id}")


def audit_curriculum_records(
    records: Sequence[CurriculumRecord],
    states: Sequence[TrainingState],
    context: TrainingContext,
) -> dict[str, Any]:
    """Recompute every label and return an explicit hash/provenance manifest."""

    if not records:
        raise ValueError("cannot audit an empty curriculum")
    state_by_id = {state.state_id: state for state in states}
    if len(state_by_id) != len(states):
        raise AssertionError("duplicate source state_id")
    for state in states:
        _validate_training_state(state, context)

    ids: set[str] = set()
    for record in records:
        if record.record_id in ids:
            raise AssertionError(f"duplicate record_id: {record.record_id}")
        ids.add(record.record_id)
        if record.schema_version != SCHEMA_VERSION or record.task_type not in TASK_TYPES:
            raise AssertionError(f"unsupported schema/task in {record.record_id}")
        if record.record_id != record.expected_record_id:
            raise AssertionError(f"record hash mismatch in {record.record_id}")
        state = state_by_id.get(record.source_state_id)
        if state is None:
            raise AssertionError(f"missing source state for {record.record_id}")
        if record.provenance != _provenance(state, context):
            raise AssertionError(f"provenance mismatch in {record.record_id}")
        leaked_keys = _forbidden_key_paths({"input": record.input, "target": record.target, "metadata": record.metadata})
        if leaked_keys:
            raise AssertionError(f"secret/candidate injection fields in {record.record_id}: {leaked_keys}")
        if record.target != _expected_target(record, state, context):
            raise AssertionError(f"target mismatch in {record.record_id}")

    candidate_rows = [record for record in records if record.task_type == "candidate_validity"]
    candidate_balance: dict[str, Any] | None = None
    if candidate_rows:
        valid = sum(record.target["valid"] for record in candidate_rows)
        invalid = len(candidate_rows) - valid
        reasons = Counter(record.target["reason"] for record in candidate_rows if not record.target["valid"])
        if valid != invalid:
            raise AssertionError("candidate-validity records are not 50/50 valid and invalid")
        if set(reasons) != set(INVALID_REASONS) or len(set(reasons.values())) != 1:
            raise AssertionError("candidate invalid reasons are not exactly balanced")
        candidate_balance = {
            "valid": valid,
            "invalid": invalid,
            "invalid_reasons": {reason: reasons[reason] for reason in INVALID_REASONS},
        }

    ordered_records = [record.to_dict() for record in sorted(records, key=lambda row: row.record_id)]
    source_hashes = {state_id: state_by_id[state_id].sha256 for state_id in sorted({r.source_state_id for r in records})}
    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "protocol_id": context.protocol_id,
        "split": "train",
        "locked_test_access": False,
        "records": len(records),
        "source_states": len(source_hashes),
        "task_distribution": dict(sorted(Counter(record.task_type for record in records).items())),
        "candidate_balance": candidate_balance,
        "hashes": {
            "records_sha256": _hash(ordered_records),
            "source_state_provenance_sha256": _hash(source_hashes),
            "source_manifest_sha256": context.source_manifest_sha256,
            "training_secret_set_sha256": context.training_secret_set_sha256,
            "answer_universe_sha256": context.answer_universe_sha256,
        },
        "checks": [
            "protocol_frozen",
            "training_secret_membership",
            "feedback_recomputed_with_frozen_scorer",
            "secret_retained_by_public_history",
            "no_post_solve_states",
            "no_secret_or_candidate_list_fields",
            "record_content_hashes",
            "source_provenance_hashes",
            "labels_recomputed",
            "candidate_50_50_balance" if candidate_rows else "candidate_balance_not_applicable",
            "invalid_reason_balance" if candidate_rows else "invalid_reason_balance_not_applicable",
            "locked_test_unread",
        ],
    }


def _prediction_word(prediction: Any) -> str | None:
    if isinstance(prediction, str):
        word = normalize_word(prediction)
    elif isinstance(prediction, Mapping):
        word = normalize_word(str(prediction.get("word", "")))
    else:
        return None
    return word if is_five_ascii_letters(word) else None


def evaluate_curriculum_predictions(
    records: Sequence[CurriculumRecord],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Score held-out microtask outputs without using any hidden answer."""

    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_reason_total: Counter[str] = Counter()
    candidate_reason_correct: Counter[str] = Counter()
    correct = 0
    predicted = 0
    for record in records:
        task = record.task_type
        by_task[task]["total"] += 1
        prediction = predictions.get(record.record_id)
        if prediction is None:
            continue
        predicted += 1
        by_task[task]["predicted"] += 1
        is_correct = False
        if task in {"feedback_decode", "constraint_merge"}:
            is_correct = isinstance(prediction, Mapping) and dict(prediction) == record.target
        elif task == "candidate_validity":
            expected_valid = record.target["valid"]
            predicted_valid = prediction if isinstance(prediction, bool) else prediction.get("valid") if isinstance(prediction, Mapping) else None
            validity_correct = predicted_valid is expected_valid
            if expected_valid:
                is_correct = validity_correct
            else:
                reason = record.target["reason"]
                candidate_reason_total[reason] += 1
                predicted_reason = prediction.get("reason") if isinstance(prediction, Mapping) else None
                reason_correct = predicted_reason == reason
                candidate_reason_correct[reason] += int(reason_correct)
                is_correct = validity_correct and reason_correct
        else:
            is_correct = _prediction_word(prediction) == record.target["word"]
        if is_correct:
            correct += 1
            by_task[task]["correct"] += 1

    task_metrics = {}
    for task, counts in sorted(by_task.items()):
        total = counts["total"]
        task_metrics[task] = {
            "total": total,
            "predicted": counts["predicted"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / total,
        }
    return {
        "records": len(records),
        "predicted": predicted,
        "coverage": predicted / len(records) if records else 0.0,
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "by_task": task_metrics,
        "candidate_invalid_reason_accuracy": {
            reason: {
                "correct": candidate_reason_correct[reason],
                "total": candidate_reason_total[reason],
                "accuracy": (
                    candidate_reason_correct[reason] / candidate_reason_total[reason]
                    if candidate_reason_total[reason]
                    else None
                ),
            }
            for reason in INVALID_REASONS
        },
    }
