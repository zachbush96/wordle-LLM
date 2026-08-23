from __future__ import annotations

import copy
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.experiment_guardrails import (
    DEFAULT_RUN_ARTIFACTS,
    GuardrailViolation,
    aggregate_three_seed_promotion,
    assert_locked_test_closed,
    assert_no_heldout_leakage,
    assert_protocol_lock_unchanged,
    build_artifact_manifest,
    normalize_gate_metrics,
    recompute_protocol_lock,
    render_promotion_markdown,
    scan_training_row_leakage,
    verify_artifact_manifest,
)
from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION


def test_live_protocol_lock_recomputes_without_writing():
    audit = assert_protocol_lock_unchanged()
    assert audit["status"] == "passed"
    assert audit["component_files_verified"] == 6
    assert len(audit["protocol_sha256"]) == 64


def test_protocol_lock_detects_a_component_hash_change():
    tampered = recompute_protocol_lock()
    tampered = copy.deepcopy(tampered)
    tampered["component_files"]["env.py"] = "0" * 64
    with pytest.raises(GuardrailViolation, match="changed components"):
        assert_protocol_lock_unchanged(tampered)


def test_training_bundle_leakage_scan_redacts_heldout_words():
    rows = [
        {"example_id": "safe", "prompt": "Choose a word", "completion": "Final answer: ABOUT"},
        {"example_id": "dev-leak", "prompt": "Prior guess CRANE", "completion": "Final answer: ABOUT"},
        {"example_id": "test-leak", "secret_answer": "slate", "completion": "Final answer: ABOUT"},
    ]
    audit = scan_training_row_leakage(rows, dev_answers=["CRANE"], locked_test_answers=["SLATE"])
    assert audit["status"] == "failed"
    assert audit["occurrences_by_split"] == {"dev": 1, "locked_test": 1}
    assert all("word_sha256" in finding and "word" not in finding for finding in audit["findings"])
    with pytest.raises(GuardrailViolation) as captured:
        assert_no_heldout_leakage(rows, dev_answers=["CRANE"], locked_test_answers=["SLATE"])
    assert "CRANE" not in str(captured.value)
    assert "SLATE" not in str(captured.value)


def test_training_bundle_leakage_scan_accepts_training_only_rows():
    audit = assert_no_heldout_leakage(
        [{"example_id": "train-1", "completion": "Final answer: ABOUT"}],
        dev_answers=["CRANE"],
        locked_test_answers=["SLATE"],
    )
    assert audit["status"] == "passed"
    assert audit["rows_scanned"] == 1


def test_locked_test_guard_requires_false_and_rejects_test_payloads():
    assert assert_locked_test_closed({"split": "dev", "locked_test_access": False})["status"] == "passed"
    with pytest.raises(GuardrailViolation, match="locked-test boundary"):
        assert_locked_test_closed({"split": "test", "locked_test_access": False})
    with pytest.raises(GuardrailViolation, match="locked-test boundary"):
        assert_locked_test_closed({"split": "dev", "locked_test_access": True})
    with pytest.raises(GuardrailViolation, match="missing"):
        assert_locked_test_closed({"split": "dev"}, require_explicit=True)


def _provenance() -> dict:
    return {
        "experiment_id": "fixture-run",
        "protocol_id": "WORDLE-PROTOCOL-002",
        "protocol_sha256": "a" * 64,
        "model_id": SUPPORTED_MODEL_ID,
        "model_revision": SUPPORTED_REVISION,
        "seed": 2026,
        "split": "dev",
        "locked_test_access": False,
        "dataset_manifest_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
        "git_commit": "deadbeef",
    }


def _write_required_artifacts(root: Path) -> None:
    root.mkdir()
    for index, name in enumerate(DEFAULT_RUN_ARTIFACTS):
        (root / name).write_text(f"artifact {index}\n", encoding="utf-8")


def test_artifact_manifest_requires_provenance_and_recomputes_hashes(tmp_path: Path):
    bundle = tmp_path / "run"
    _write_required_artifacts(bundle)
    manifest = build_artifact_manifest(bundle, provenance=_provenance())
    verified = verify_artifact_manifest(bundle, manifest)
    assert verified["status"] == "passed"
    assert verified["artifacts_verified"] == len(DEFAULT_RUN_ARTIFACTS)
    assert verified["locked_test_access"] is False

    (bundle / "summary.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(GuardrailViolation, match="content_mismatch:summary.json"):
        verify_artifact_manifest(bundle, manifest)


def test_artifact_manifest_fails_on_missing_file_or_test_provenance(tmp_path: Path):
    bundle = tmp_path / "run"
    _write_required_artifacts(bundle)
    (bundle / "summary.json").unlink()
    with pytest.raises(GuardrailViolation, match="required artifact is missing"):
        build_artifact_manifest(bundle, provenance=_provenance())

    (bundle / "summary.json").write_text("summary\n", encoding="utf-8")
    bad = {**_provenance(), "split": "test"}
    with pytest.raises(GuardrailViolation, match="split must be dev"):
        build_artifact_manifest(bundle, provenance=bad)


def _nested_summary(seed: int, *, terminal: float = 0.995, turn_two: float = 0.25, singleton: float = 0.82) -> dict:
    return {
        "seed": seed,
        "run_id": f"run-{seed}",
        "recipe_id": "matched-recipe-v1",
        "protocol_id": "WORDLE-PROTOCOL-002",
        "protocol_sha256": "a" * 64,
        "model_id": SUPPORTED_MODEL_ID,
        "model_revision": SUPPORTED_REVISION,
        "dataset_manifest_sha256": "b" * 64,
        "split": "dev",
        "locked_test_access": False,
        "gameplay": {
            "win_rate": 0.25,
            "terminal_marker_compliance": terminal,
            "invalid_guess_rate": 0.01,
            "repeat_guess_rate": 0.02,
        },
        "diagnostics": {
            "action_target_accuracy": 0.40,
            "singleton_answer_accuracy": singleton,
            "train_state_coverage": 0.50,
            "by_turn": {"2": {"posterior_constraint_violation_rate": turn_two}},
        },
    }


def test_nested_metrics_normalize_to_the_flat_gate_schema():
    normalized = normalize_gate_metrics(_nested_summary(2026))
    assert normalized["terminal_compliance"] == pytest.approx(0.995)
    assert normalized["turn_2_posterior_violation_rate"] == pytest.approx(0.25)
    assert normalized["singleton_answer_accuracy"] == pytest.approx(0.82)
    assert normalized["invalid_guess_rate"] == pytest.approx(0.01)
    assert normalized["locked_test_access"] is False
    assert len(normalized["source_metrics_sha256"]) == 64


def test_metric_normalization_rejects_test_or_missing_gate_metrics():
    test_summary = {**_nested_summary(2026), "split": "test"}
    with pytest.raises(GuardrailViolation, match="locked-test boundary"):
        normalize_gate_metrics(test_summary)
    missing = _nested_summary(2026)
    del missing["diagnostics"]["by_turn"]["2"]
    with pytest.raises(GuardrailViolation, match="turn_2_posterior_violation_rate"):
        normalize_gate_metrics(missing)


def test_three_seed_promotion_requires_every_seed_to_pass_and_stays_dev_only():
    result = aggregate_three_seed_promotion(
        [_nested_summary(2026), _nested_summary(2027), _nested_summary(2028)],
        declared_seeds=[2026, 2027, 2028],
    )
    assert result["promote"] is True
    assert result["development_gate_passed"] is True
    assert result["locked_test_access"] is False
    assert result["locked_test_eligible"] is False
    assert result["test_access"] == "forbidden"
    report = render_promotion_markdown(result)
    assert "Locked-test access: **false**" in report
    assert "2026" in report and "2028" in report


def test_three_seed_promotion_rejects_one_failed_seed():
    result = aggregate_three_seed_promotion(
        [_nested_summary(2026), _nested_summary(2027, singleton=0.79), _nested_summary(2028)],
        declared_seeds=[2026, 2027, 2028],
    )
    assert result["promote"] is False
    assert result["status"] == "rejected"
    assert result["failures"] == ["seed=2027:threshold_failed:singleton_answer_accuracy"]


def test_three_seed_promotion_requires_exact_declared_seed_set_and_matched_provenance():
    with pytest.raises(GuardrailViolation, match="exactly match"):
        aggregate_three_seed_promotion(
            [_nested_summary(2026), _nested_summary(2027), _nested_summary(2029)],
            declared_seeds=[2026, 2027, 2028],
        )
    mismatched = _nested_summary(2028)
    mismatched["recipe_id"] = "different"
    with pytest.raises(GuardrailViolation, match="not matched"):
        aggregate_three_seed_promotion(
            [_nested_summary(2026), _nested_summary(2027), mismatched],
            declared_seeds=[2026, 2027, 2028],
        )

    missing = _nested_summary(2028)
    del missing["dataset_manifest_sha256"]
    with pytest.raises(GuardrailViolation, match="required for every seed"):
        aggregate_three_seed_promotion(
            [_nested_summary(2026), _nested_summary(2027), missing],
            declared_seeds=[2026, 2027, 2028],
        )
