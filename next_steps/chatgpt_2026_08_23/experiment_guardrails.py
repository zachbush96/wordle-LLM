from __future__ import annotations

"""Shared, fail-closed auditing and reporting for the next-step experiments.

The helpers in this module deliberately stop at a development promotion
decision.  They never load, return, or authorize the locked-test answer split.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wordle_lab.common import DATA, canonical_json, sha256_file, sha256_text
from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION
from wordle_lab.protocol.generation import GENERATION_CONFIG
from wordle_lab.protocol.lock import PROTOCOL_ID
from wordle_lab.protocol.parsing import TERMINAL_PATTERN
from wordle_lab.protocol.prompting import PROMPT_VERSION, SYSTEM_PROMPT

SUITE_ID = "CHATGPT-WORDLE-NEXT-STEPS-2026-08-23"


class GuardrailViolation(RuntimeError):
    """Raised when an experiment would no longer satisfy the frozen contract."""


PROTOCOL_COMPONENT_FILES = (
    "env.py",
    "prompting.py",
    "parsing.py",
    "generation.py",
    "evaluator.py",
    "retention.py",
)

DEFAULT_RUN_ARTIFACTS = (
    "spec.json",
    "dataset_manifest.json",
    "train_metrics.jsonl",
    "accounting.json",
    "summary.json",
)

PROMOTION_THRESHOLDS = {
    "terminal_compliance": {"op": ">=", "value": 0.99},
    "turn_2_posterior_violation_rate": {"op": "<=", "value": 0.30},
    "singleton_answer_accuracy": {"op": ">=", "value": 0.80},
}

MATCHED_PROVENANCE_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "model_id",
    "model_revision",
    "dataset_manifest_sha256",
    "recipe_id",
)

_FIVE_LETTER_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LOCKED_TEST_FIELDS = {
    "locked_test_answer",
    "locked_test_answers",
    "test_answer",
    "test_answers",
}


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise GuardrailViolation(f"missing {label}: {resolved}")
    return resolved


def recompute_protocol_lock(
    *,
    wordlist: str | Path = DATA / "wordlists" / "allowed_words.txt",
    split_manifest: str | Path = DATA / "splits" / "source_split_manifest.json",
    retention_probes: str | Path = DATA / "retention_probes_v1.jsonl",
    protocol_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute WORDLE-PROTOCOL-002 without modifying its on-disk lock."""

    source_dir = Path(protocol_dir) if protocol_dir else Path(__file__).resolve().parents[2] / "wordle_lab" / "protocol"
    wordlist_path = _require_file(wordlist, "protocol word list")
    split_path = _require_file(split_manifest, "split manifest")
    retention_path = _require_file(retention_probes, "retention probes")
    component_paths = {name: _require_file(source_dir / name, f"protocol component {name}") for name in PROTOCOL_COMPONENT_FILES}
    components = {
        "protocol_id": PROTOCOL_ID,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "terminal_pattern": TERMINAL_PATTERN,
        "generation": dict(GENERATION_CONFIG),
        "wordlist_sha256": sha256_file(wordlist_path),
        "split_manifest_sha256": sha256_file(split_path),
        "retention_probe_sha256": sha256_file(retention_path),
        "component_files": {name: sha256_file(path) for name, path in component_paths.items()},
    }
    return {**components, "protocol_sha256": sha256_text(canonical_json(components))}


def assert_protocol_lock_unchanged(
    expected_lock: str | Path | Mapping[str, Any] = DATA / "protocol_lock.json",
    **recompute_kwargs: Any,
) -> dict[str, Any]:
    """Fail if any frozen protocol value, dependency hash, or component changed."""

    lock_path: Path | None = None
    if isinstance(expected_lock, Mapping):
        expected = dict(expected_lock)
    else:
        lock_path = _require_file(expected_lock, "protocol lock")
        expected = json.loads(lock_path.read_text(encoding="utf-8"))
    observed = recompute_protocol_lock(**recompute_kwargs)
    if expected != observed:
        changed = sorted(key for key in set(expected) | set(observed) if expected.get(key) != observed.get(key))
        expected_components = expected.get("component_files", {}) if isinstance(expected.get("component_files"), Mapping) else {}
        observed_components = observed.get("component_files", {})
        changed_components = sorted(
            name
            for name in set(expected_components) | set(observed_components)
            if expected_components.get(name) != observed_components.get(name)
        )
        detail = f"; changed components: {changed_components}" if changed_components else ""
        raise GuardrailViolation(f"{PROTOCOL_ID} lock mismatch in fields: {changed}{detail}")
    return {
        "status": "passed",
        "protocol_id": observed["protocol_id"],
        "protocol_sha256": observed["protocol_sha256"],
        "component_files_verified": len(observed["component_files"]),
        "lock_file_sha256": sha256_file(lock_path) if lock_path else None,
    }


def _walk_string_values(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _walk_string_values(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for index, nested in enumerate(value):
            yield from _walk_string_values(nested, f"{path}[{index}]")


def _heldout_words(words: Iterable[str], label: str) -> set[str]:
    normalized = [str(word).strip().upper() for word in words]
    if not normalized:
        raise GuardrailViolation(f"{label} answer set is empty")
    invalid = sorted({word for word in normalized if not re.fullmatch(r"[A-Z]{5}", word)})
    if invalid:
        raise GuardrailViolation(f"{label} answer set contains malformed words")
    if len(normalized) != len(set(normalized)):
        raise GuardrailViolation(f"{label} answer set contains duplicates")
    return set(normalized)


def _word_set_sha256(words: set[str]) -> str:
    return sha256_text(canonical_json(sorted(words)))


def scan_training_row_leakage(
    rows: Iterable[Mapping[str, Any]],
    *,
    dev_answers: Iterable[str],
    locked_test_answers: Iterable[str],
) -> dict[str, Any]:
    """Conservatively scan every string value for held-out five-letter tokens.

    Findings contain only a SHA-256 of the word, never the held-out word itself.
    This makes the result suitable for persisted audit reports without copying
    locked-test contents into logs.
    """

    materialized = list(rows)
    if not materialized:
        raise GuardrailViolation("training row bundle is empty")
    dev = _heldout_words(dev_answers, "development")
    locked = _heldout_words(locked_test_answers, "locked-test")
    if dev & locked:
        raise GuardrailViolation("development and locked-test answer sets overlap")

    findings: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row_index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise GuardrailViolation(f"training row {row_index} is not a mapping")
        row_id = next(
            (str(row[key]) for key in ("example_id", "comparison_id", "state_id", "pair_id") if row.get(key) is not None),
            f"row-{row_index}",
        )
        for value_path, text in _walk_string_values(row):
            occurrences = Counter(token.upper() for token in _FIVE_LETTER_WORD.findall(text))
            for token, occurrence_count in sorted(occurrences.items()):
                split = "dev" if token in dev else "locked_test" if token in locked else None
                if split is None:
                    continue
                counts[split] += occurrence_count
                findings.append(
                    {
                        "row_index": row_index,
                        "row_id": row_id,
                        "value_path": value_path,
                        "split": split,
                        "occurrences": occurrence_count,
                        "word_sha256": sha256_text(token),
                    }
                )
    return {
        "status": "passed" if not findings else "failed",
        "rows_scanned": len(materialized),
        "five_letter_token_findings": len(findings),
        "occurrences_by_split": {"dev": counts["dev"], "locked_test": counts["locked_test"]},
        "dev_answer_set_sha256": _word_set_sha256(dev),
        "locked_test_answer_set_sha256": _word_set_sha256(locked),
        "findings": findings,
    }


def assert_no_heldout_leakage(
    rows: Iterable[Mapping[str, Any]],
    *,
    dev_answers: Iterable[str],
    locked_test_answers: Iterable[str],
) -> dict[str, Any]:
    audit = scan_training_row_leakage(rows, dev_answers=dev_answers, locked_test_answers=locked_test_answers)
    if audit["status"] != "passed":
        counts = audit["occurrences_by_split"]
        raise GuardrailViolation(
            "training bundle contains held-out tokens "
            f"(dev={counts['dev']}, locked_test={counts['locked_test']}); words are redacted"
        )
    return audit


def assert_locked_test_closed(payload: Mapping[str, Any], *, require_explicit: bool = False) -> dict[str, Any]:
    """Reject test splits, answer fields, and any truthy test-access declaration."""

    violations: list[str] = []
    explicit_flags = 0

    def visit(value: Any, path: str) -> None:
        nonlocal explicit_flags
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key).lower()
                child = f"{path}.{raw_key}"
                if key in _FORBIDDEN_LOCKED_TEST_FIELDS:
                    violations.append(child)
                elif key == "locked_test_access":
                    explicit_flags += 1
                    if nested is not False:
                        violations.append(child)
                elif key == "test_access":
                    explicit_flags += 1
                    if nested not in (False, "forbidden", "closed"):
                        violations.append(child)
                elif key in {"split", "evaluation_split", "secret_split"} and str(nested).lower() in {
                    "test",
                    "locked_test",
                    "locked-test",
                }:
                    violations.append(child)
                visit(nested, child)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "$")
    if require_explicit and explicit_flags == 0:
        violations.append("$.locked_test_access:missing")
    if violations:
        raise GuardrailViolation(f"locked-test boundary violation at: {sorted(set(violations))}")
    return {"status": "passed", "locked_test_access": False, "explicit_flags_verified": explicit_flags}


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value).lower()
    if not _HEX_64.fullmatch(normalized):
        raise GuardrailViolation(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def validate_run_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum provenance required for a durable run bundle."""

    required = {
        "experiment_id",
        "protocol_id",
        "protocol_sha256",
        "model_id",
        "model_revision",
        "seed",
        "split",
        "locked_test_access",
        "dataset_manifest_sha256",
        "source_tree_sha256",
        "git_commit",
    }
    missing = sorted(required - set(provenance))
    if missing:
        raise GuardrailViolation(f"missing provenance fields: {missing}")
    result = dict(provenance)
    if not isinstance(result["experiment_id"], str) or not result["experiment_id"].strip():
        raise GuardrailViolation("experiment_id must be a non-empty string")
    if result["protocol_id"] != PROTOCOL_ID:
        raise GuardrailViolation(f"protocol_id must remain {PROTOCOL_ID}")
    if result["model_id"] != SUPPORTED_MODEL_ID or result["model_revision"] != SUPPORTED_REVISION:
        raise GuardrailViolation("run provenance must use the pinned Gemma 3 270M model and revision")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int):
        raise GuardrailViolation("seed must be an integer")
    if result["split"] != "dev":
        raise GuardrailViolation("run provenance split must be dev")
    if not isinstance(result["git_commit"], str) or not result["git_commit"].strip():
        raise GuardrailViolation("git_commit must be a non-empty string")
    result["protocol_sha256"] = _require_sha256(result["protocol_sha256"], "protocol_sha256")
    result["dataset_manifest_sha256"] = _require_sha256(
        result["dataset_manifest_sha256"], "dataset_manifest_sha256"
    )
    result["source_tree_sha256"] = _require_sha256(result["source_tree_sha256"], "source_tree_sha256")
    assert_locked_test_closed(result, require_explicit=True)
    return result


def _artifact_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise GuardrailViolation(f"artifact path must stay inside its bundle: {relative_name}")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise GuardrailViolation(f"artifact path escapes its bundle: {relative_name}") from exc
    return candidate


def build_artifact_manifest(
    bundle_dir: str | Path,
    *,
    provenance: Mapping[str, Any],
    required_artifacts: Sequence[str] = DEFAULT_RUN_ARTIFACTS,
) -> dict[str, Any]:
    """Build a content-addressed manifest after checking required run outputs."""

    root = Path(bundle_dir)
    if not root.is_dir():
        raise GuardrailViolation(f"artifact bundle directory is missing: {root}")
    names = tuple(str(name) for name in required_artifacts)
    if not names or len(names) != len(set(names)):
        raise GuardrailViolation("required_artifacts must be a non-empty unique sequence")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in names:
        path = _artifact_path(root, name)
        if not path.is_file():
            raise GuardrailViolation(f"required artifact is missing: {name}")
        size = path.stat().st_size
        if size <= 0:
            raise GuardrailViolation(f"required artifact is empty: {name}")
        artifacts[name] = {"bytes": size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "chatgpt-next-steps-artifacts-v1",
        "suite_id": SUITE_ID,
        "provenance": validate_run_provenance(provenance),
        "required_artifacts": list(names),
        "artifacts": artifacts,
    }
    manifest["manifest_content_sha256"] = sha256_text(canonical_json(manifest))
    return manifest


def verify_artifact_manifest(
    bundle_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    required_artifacts: Sequence[str] = DEFAULT_RUN_ARTIFACTS,
) -> dict[str, Any]:
    """Recompute every required artifact digest and provenance assertion."""

    supplied = dict(manifest)
    content_digest = supplied.pop("manifest_content_sha256", None)
    if content_digest != sha256_text(canonical_json(supplied)):
        raise GuardrailViolation("artifact manifest content hash mismatch")
    if supplied.get("schema_version") != "chatgpt-next-steps-artifacts-v1" or supplied.get("suite_id") != SUITE_ID:
        raise GuardrailViolation("artifact manifest schema or suite identifier mismatch")
    validate_run_provenance(supplied.get("provenance", {}))
    declared_required = supplied.get("required_artifacts")
    expected_required = list(map(str, required_artifacts))
    if declared_required != expected_required:
        raise GuardrailViolation("artifact manifest required-file declaration mismatch")
    artifact_entries = supplied.get("artifacts")
    if not isinstance(artifact_entries, Mapping):
        raise GuardrailViolation("artifact manifest requires an artifacts mapping")
    root = Path(bundle_dir)
    failures = []
    for name in expected_required:
        entry = artifact_entries.get(name)
        if not isinstance(entry, Mapping):
            failures.append(f"missing:{name}")
            continue
        path = _artifact_path(root, name)
        if not path.is_file():
            failures.append(f"missing:{name}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size <= 0 or entry.get("bytes") != actual_size or entry.get("sha256") != actual_hash:
            failures.append(f"content_mismatch:{name}")
    if failures:
        raise GuardrailViolation(f"artifact verification failed: {failures}")
    return {
        "status": "passed",
        "artifacts_verified": len(expected_required),
        "manifest_content_sha256": content_digest,
        "locked_test_access": False,
    }


def _first_defined(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise GuardrailViolation(f"missing or invalid metric: {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GuardrailViolation(f"missing or invalid metric: {field}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise GuardrailViolation(f"metric {field} must be finite and in [0, 1]")
    return number


def normalize_gate_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten gameplay and fixed-state diagnostics into the study gate schema."""

    if not isinstance(summary, Mapping):
        raise GuardrailViolation("metrics summary must be a mapping")
    assert_locked_test_closed(summary, require_explicit=True)
    if summary.get("split") != "dev":
        raise GuardrailViolation("promotion metrics must come from the dev split")
    gameplay = summary.get("gameplay", {})
    if not isinstance(gameplay, Mapping):
        raise GuardrailViolation("gameplay metrics must be a mapping")
    diagnostics = _first_defined(summary.get("diagnostics"), summary.get("state_diagnostics"), {})
    if not isinstance(diagnostics, Mapping):
        raise GuardrailViolation("diagnostic metrics must be a mapping")
    by_turn = diagnostics.get("by_turn", {})
    if not isinstance(by_turn, Mapping):
        raise GuardrailViolation("diagnostics.by_turn must be a mapping")
    turn_two = _first_defined(by_turn.get("2"), by_turn.get(2), {})
    if not isinstance(turn_two, Mapping):
        raise GuardrailViolation("turn-2 diagnostics must be a mapping")

    normalized = {
        "schema_version": "chatgpt-next-steps-gate-metrics-v1",
        "split": "dev",
        "locked_test_access": False,
        "terminal_compliance": _unit_interval(
            _first_defined(
                summary.get("terminal_compliance"),
                gameplay.get("terminal_marker_compliance"),
                summary.get("terminal_marker_compliance"),
                gameplay.get("terminal_compliance"),
                diagnostics.get("terminal_compliance"),
            ),
            "terminal_compliance",
        ),
        "invalid_guess_rate": _unit_interval(
            _first_defined(summary.get("invalid_guess_rate"), gameplay.get("invalid_guess_rate")),
            "invalid_guess_rate",
        ),
        "repeat_guess_rate": _unit_interval(
            _first_defined(summary.get("repeat_guess_rate"), gameplay.get("repeat_guess_rate")),
            "repeat_guess_rate",
        ),
        "turn_2_posterior_violation_rate": _unit_interval(
            _first_defined(
                summary.get("turn_2_posterior_violation_rate"),
                turn_two.get("posterior_constraint_violation_rate"),
            ),
            "turn_2_posterior_violation_rate",
        ),
        "singleton_answer_accuracy": _unit_interval(
            _first_defined(summary.get("singleton_answer_accuracy"), diagnostics.get("singleton_answer_accuracy")),
            "singleton_answer_accuracy",
        ),
        "source_metrics_sha256": sha256_text(canonical_json(summary)),
    }
    optional_fields = (
        "seed",
        "run_id",
        "recipe_id",
        "protocol_id",
        "protocol_sha256",
        "model_id",
        "model_revision",
        "dataset_manifest_sha256",
    )
    for field in optional_fields:
        if summary.get(field) is not None:
            normalized[field] = summary[field]
    for field in ("win_rate", "action_target_accuracy", "train_state_coverage"):
        value = _first_defined(summary.get(field), gameplay.get(field), diagnostics.get(field))
        if value is not None:
            normalized[field] = _unit_interval(value, field)
    return normalized


def aggregate_three_seed_promotion(
    seed_summaries: Iterable[Mapping[str, Any]],
    *,
    declared_seeds: Sequence[int],
) -> dict[str, Any]:
    """Apply every reliability gate to exactly three declared development seeds."""

    declared = tuple(declared_seeds)
    if len(declared) != 3 or len(set(declared)) != 3 or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in declared):
        raise GuardrailViolation("declared_seeds must contain exactly three unique integers")
    normalized = [normalize_gate_metrics(summary) for summary in seed_summaries]
    if len(normalized) != 3:
        raise GuardrailViolation("promotion requires exactly three seed summaries")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in normalized:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise GuardrailViolation("every seed summary must declare an integer seed")
        if seed in by_seed:
            raise GuardrailViolation(f"duplicate seed summary: {seed}")
        by_seed[seed] = row
    if set(by_seed) != set(declared):
        raise GuardrailViolation(
            f"observed seeds {sorted(by_seed)} do not exactly match declared seeds {sorted(declared)}"
        )

    for field in MATCHED_PROVENANCE_FIELDS:
        values = [by_seed[seed].get(field) for seed in declared]
        present = [value is not None for value in values]
        if not all(present):
            raise GuardrailViolation(f"matched provenance field is required for every seed: {field}")
        if len({canonical_json(value) for value in values}) != 1:
            raise GuardrailViolation(f"seed summaries are not matched on provenance field: {field}")

    per_seed = []
    all_failures: list[str] = []
    for seed in declared:
        metrics = by_seed[seed]
        failures = []
        checks = {}
        for metric, rule in PROMOTION_THRESHOLDS.items():
            observed = metrics[metric]
            passed = observed >= rule["value"] if rule["op"] == ">=" else observed <= rule["value"]
            checks[metric] = {"observed": observed, **rule, "passed": passed}
            if not passed:
                failure = f"seed={seed}:threshold_failed:{metric}"
                failures.append(failure)
                all_failures.append(failure)
        per_seed.append({"seed": seed, "passed": not failures, "failures": failures, "checks": checks, "metrics": metrics})

    aggregate = {}
    for metric in PROMOTION_THRESHOLDS:
        values = [by_seed[seed][metric] for seed in declared]
        aggregate[metric] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    passed = not all_failures
    result = {
        "schema_version": "chatgpt-next-steps-three-seed-promotion-v1",
        "suite_id": SUITE_ID,
        "status": "passed" if passed else "rejected",
        "promote": passed,
        "development_gate_passed": passed,
        "promotion_scope": "development_only",
        "split": "dev",
        "declared_seeds": list(declared),
        "observed_seeds": list(declared),
        "thresholds": PROMOTION_THRESHOLDS,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "failures": all_failures,
        "locked_test_access": False,
        "locked_test_eligible": False,
        "test_access": "forbidden",
    }
    assert_locked_test_closed(result, require_explicit=True)
    return result


def render_promotion_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact, LLM-readable report without granting test access."""

    assert_locked_test_closed(result, require_explicit=True)
    if result.get("schema_version") != "chatgpt-next-steps-three-seed-promotion-v1":
        raise GuardrailViolation("unsupported promotion result schema")
    lines = [
        "# Three-seed development promotion report",
        "",
        f"Outcome: **{'PASS' if result.get('development_gate_passed') else 'REJECT'}**.",
        "Locked-test access: **false**. This report cannot authorize a test run.",
        "",
        "| Seed | Terminal compliance | Turn-2 violations | Singleton accuracy | Gate |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.get("per_seed", []):
        metrics = row["metrics"]
        lines.append(
            f"| {row['seed']} | {metrics['terminal_compliance']:.1%} | "
            f"{metrics['turn_2_posterior_violation_rate']:.1%} | "
            f"{metrics['singleton_answer_accuracy']:.1%} | {'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(["", "Thresholds: terminal >=99%, turn-2 violations <=30%, singleton accuracy >=80% for every seed."])
    if result.get("failures"):
        lines.extend(["", "Failures:"] + [f"- {failure}" for failure in result["failures"]])
    return "\n".join(lines) + "\n"
