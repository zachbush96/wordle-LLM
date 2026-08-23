from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import next_steps.chatgpt_2026_08_23.q_sft_frozen as q_sft_frozen
from next_steps.chatgpt_2026_08_23.q_sft_frozen import (
    BEHAVIOR_POLICY_ID,
    BLOCKED_PREREQUISITE_STATUS,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    FORBIDDEN_EMITTED_FIELDS,
    JOINED_FIELDS,
    PARENT_DEV_EVIDENCE,
    PREREQUISITE_THRESHOLDS,
    SNAPSHOT_FIELDS,
    TARGET_ID,
    assess_parent_prerequisite_legality,
    audit_bundle,
    audit_target_contract,
    build_bundle,
    build_frozen_snapshots,
    build_parser,
    conservative_bellman_target,
    dry_run,
    evaluate_prerequisite_thresholds,
    join_snapshots_to_training_rows,
    load_config,
    validate_config,
    validate_parent_adapter,
    validate_snapshot_rows,
)
from next_steps.chatgpt_2026_08_23.experiment_guardrails import GuardrailViolation
from wordle_lab.methods.q_sft import validate_q_sft_rows


ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "artifacts" / "runs" / "sft-common-balanced-word-s2026-0649b4deeb" / "checkpoints" / "final"


def _source_row(
    example_id: str,
    posterior_size: int,
    turn: int,
    *,
    split: str = "common_train",
    state_id: str | None = None,
    action: str = "ABOUT",
) -> dict:
    source_state_id = state_id or f"common_train-{example_id}"
    return {
        "example_id": example_id,
        "state_id": source_state_id,
        "target_word": action,
        "posterior_size": posterior_size,
        "turn": turn,
        "prompt": [{"role": "user", "content": "Choose a Wordle action from the visible history."}],
        "completion": [{"role": "assistant", "content": f"Final answer: {action}"}],
        "source_state": {
            "split": split,
            "state_id": source_state_id,
            "turn": turn,
            "secret_answer": "ABOUT",
            "posterior_candidates": ["ABOUT"],
            "facts": {"oracle_action": "ABOUT"},
        },
    }


def test_conservative_target_is_bounded_and_monotone():
    by_posterior = [conservative_bellman_target(size, 2) for size in (128, 64, 8, 2, 1)]
    assert by_posterior == sorted(by_posterior)
    assert all(0.0 <= value <= 1.0 for value in by_posterior)
    assert conservative_bellman_target(1, 6) == 1.0
    assert conservative_bellman_target(2, 6) == 0.5
    assert conservative_bellman_target(8, 1) >= conservative_bellman_target(8, 6)
    with pytest.raises(ValueError, match="posterior_size"):
        conservative_bellman_target(0, 1)
    contract = audit_target_contract()
    assert contract["states_checked"] == 128 * 6
    assert contract["bounded_0_1"] is True
    assert contract["nonincreasing_with_posterior_size"] is True
    assert contract["nondecreasing_with_guesses_remaining"] is True


def test_snapshot_builder_is_deterministic_and_emits_sanitized_behavior_provenance():
    state_id = "common_train-shared-state"
    rows = [
        _source_row("b", 2, 4, state_id=state_id, action="OTHER"),
        _source_row("a", 2, 4, state_id=state_id, action="ABOUT"),
    ]
    first, audit = build_frozen_snapshots(rows)
    second, repeated_audit = build_frozen_snapshots(reversed(rows))
    assert first == second
    assert audit["snapshot_rows_sha256"] == repeated_audit["snapshot_rows_sha256"]
    assert audit["target_id"] == TARGET_ID
    assert [row["comparison_id"] for row in first] == ["a", "b"]
    assert all(set(row) == SNAPSHOT_FIELDS for row in first)
    assert {row["source_state_id"] for row in first} == {state_id}
    assert {row["behavior_action"] for row in first} == {"ABOUT", "OTHER"}
    assert {row["behavior_probability"] for row in first} == {0.5}
    assert {row["behavior_support_size"] for row in first} == {2}
    assert len({row["behavior_support_sha256"] for row in first}) == 1
    assert {row["behavior_policy_id"] for row in first} == {BEHAVIOR_POLICY_ID}
    assert all(row["posterior_size"] == 2 and row["turn"] == 4 for row in first)
    assert audit["behavior_provenance"]["source_states"] == 1
    assert audit["behavior_provenance"]["behavior_actions"] == 2
    assert all(not (set(row) & FORBIDDEN_EMITTED_FIELDS) for row in first)
    assert audit["locked_test_access"] is False


def test_snapshot_join_drops_private_source_state_and_passes_core_validation():
    source = [_source_row("row-1", 3, 3), _source_row("row-2", 1, 5)]
    snapshots, _ = build_frozen_snapshots(source)
    joined = join_snapshots_to_training_rows(source, snapshots)
    assert all(set(row) == JOINED_FIELDS for row in joined)
    assert all("source_state" not in row and "secret_answer" not in row for row in joined)
    assert [row["source_state_id"] for row in joined] == ["common_train-row-1", "common_train-row-2"]
    assert all(row["behavior_action"] == "ABOUT" for row in joined)
    assert all(row["behavior_probability"] == 1.0 for row in joined)
    targets = validate_q_sft_rows(joined, discount=0.99)
    assert targets == [row["bellman_target"] for row in joined]


def test_builder_rejects_nontraining_rows_mismatched_joins_and_extra_snapshot_fields():
    with pytest.raises(GuardrailViolation, match="not training-only"):
        build_frozen_snapshots([_source_row("dev", 2, 2, split="dev")])
    source = [_source_row("row-1", 2, 2)]
    snapshots, _ = build_frozen_snapshots(source)
    with pytest.raises(GuardrailViolation, match="do not match exactly"):
        join_snapshots_to_training_rows([_source_row("different", 2, 2)], snapshots)
    bad = [{**snapshots[0], "secret_answer": "ABOUT"}]
    with pytest.raises(GuardrailViolation, match="exactly"):
        validate_snapshot_rows(bad)


def test_behavior_support_and_probability_validation_fail_closed():
    state_id = "common_train-shared-state"
    source = [
        _source_row("a", 3, 2, state_id=state_id, action="ABOUT"),
        _source_row("b", 3, 2, state_id=state_id, action="OTHER"),
    ]
    snapshots, _ = build_frozen_snapshots(source)

    changed = copy.deepcopy(snapshots)
    changed[0]["behavior_probability"] = 0.6
    with pytest.raises(GuardrailViolation, match="probability is not uniform"):
        validate_snapshot_rows(changed)

    changed = copy.deepcopy(snapshots)
    changed[0]["behavior_support_sha256"] = "0" * 64
    with pytest.raises(GuardrailViolation, match="support hash mismatch"):
        validate_snapshot_rows(changed)

    changed = copy.deepcopy(snapshots)
    changed[0]["posterior_size"] = 2
    with pytest.raises(GuardrailViolation, match="bellman_target does not match"):
        validate_snapshot_rows(changed)


def test_source_state_action_and_uniform_support_bindings_fail_closed():
    inconsistent_state = _source_row("state", 2, 2)
    inconsistent_state["source_state"]["state_id"] = "common_train-different"
    with pytest.raises(GuardrailViolation, match="unstable source_state_id"):
        build_frozen_snapshots([inconsistent_state])

    mismatched_action = _source_row("action", 2, 2)
    mismatched_action["completion"][0]["content"] = "Final answer: OTHER"
    with pytest.raises(GuardrailViolation, match="action/completion mismatch"):
        build_frozen_snapshots([mismatched_action])

    duplicate_samples = [
        _source_row("a", 2, 2, state_id="common_train-same", action="ABOUT"),
        _source_row("b", 2, 2, state_id="common_train-same", action="ABOUT"),
    ]
    snapshots, audit = build_frozen_snapshots(duplicate_samples)
    assert {row["behavior_support_size"] for row in snapshots} == {1}
    assert {row["behavior_probability"] for row in snapshots} == {1.0}
    assert audit["behavior_provenance"]["behavior_actions"] == 1
    assert audit["behavior_provenance"]["snapshot_rows"] == 2


def test_frozen_config_requires_explicit_warm_start_and_closed_test():
    config = load_config()
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["warm_start"]["required"] is True
    assert config["behavior_provenance"]["policy_id"] == BEHAVIOR_POLICY_ID
    assert config["behavior_provenance"]["uniform"] is True
    assert config["locked_test_access"] is False
    changed = copy.deepcopy(config)
    changed["locked_test_access"] = True
    with pytest.raises(GuardrailViolation, match="configuration drifted"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["warm_start"]["expected_parent_run_id"] = "implicit-base"
    with pytest.raises(GuardrailViolation, match="configuration drifted"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["behavior_provenance"]["behavior_probability"] = "unknown"
    with pytest.raises(GuardrailViolation, match="configuration drifted"):
        validate_config(changed)


def test_exact_parent_adapter_and_hash_provenance_validate():
    config = load_config()
    result = validate_parent_adapter(PARENT, config["warm_start"])
    assert result["status"] == "passed"
    assert result["parent_run_id"] == config["warm_start"]["expected_parent_run_id"]
    assert result["parent_adapter_tree_sha256"] == config["warm_start"]["expected_parent_adapter_tree_sha256"]
    assert result["locked_test_access"] is False
    changed = copy.deepcopy(config["warm_start"])
    changed["expected_parent_adapter_tree_sha256"] = "f" * 64
    with pytest.raises(GuardrailViolation, match="tree hash mismatch"):
        validate_parent_adapter(PARENT, changed)


def test_prerequisite_threshold_boundaries_are_exact():
    passing = evaluate_prerequisite_thresholds(0.99, 0.299999, 0.80)
    assert passing["passed"] is True
    assert passing["failed_checks"] == []
    assert passing["checks"]["terminal_compliance"]["comparator"] == ">="
    assert passing["checks"]["turn_2_posterior_violation_rate"]["comparator"] == "<"
    assert passing["checks"]["singleton_answer_accuracy"]["comparator"] == ">="

    terminal_failure = evaluate_prerequisite_thresholds(0.989999, 0.0, 1.0)
    turn_two_failure = evaluate_prerequisite_thresholds(1.0, 0.30, 1.0)
    singleton_failure = evaluate_prerequisite_thresholds(1.0, 0.0, 0.799999)
    assert terminal_failure["failed_checks"] == ["terminal_compliance"]
    assert turn_two_failure["failed_checks"] == ["turn_2_posterior_violation_rate"]
    assert singleton_failure["failed_checks"] == ["singleton_answer_accuracy"]


def test_fixed_parent_fails_every_gate_from_hash_pinned_development_evidence():
    config = load_config()
    result = assess_parent_prerequisite_legality(PARENT, config["warm_start"])
    assert result["status"] == BLOCKED_PREREQUISITE_STATUS
    assert result["passed"] is False
    assert result["reason"] == "parent_does_not_meet_declared_development_thresholds"
    assert result["evidence_available"] is True
    assert set(result["evidence"]) == set(PARENT_DEV_EVIDENCE)
    assert all(item["matched"] is True for item in result["evidence"].values())
    assert result["thresholds"] == PREREQUISITE_THRESHOLDS
    assert result["checks"]["terminal_compliance"]["observed"] == pytest.approx(0.8920454545454546)
    assert result["checks"]["turn_2_posterior_violation_rate"]["observed"] == pytest.approx(
        0.7931034482758621
    )
    assert result["checks"]["singleton_answer_accuracy"]["observed"] == pytest.approx(
        0.05405405405405406
    )
    assert set(result["failed_checks"]) == {
        "terminal_compliance",
        "turn_2_posterior_violation_rate",
        "singleton_answer_accuracy",
    }
    assert result["diagnostic_coverage"] == {
        "turn_2_items": 58,
        "singleton_items": 74,
        "total_items": 128,
    }
    assert result["locked_test_access"] is False


def test_parent_gate_fails_closed_for_missing_or_changed_evidence(monkeypatch):
    config = load_config()

    missing = copy.deepcopy(PARENT_DEV_EVIDENCE)
    missing["summary"]["path"] = "development-evidence-missing.json"
    monkeypatch.setattr(q_sft_frozen, "PARENT_DEV_EVIDENCE", missing)
    missing_result = assess_parent_prerequisite_legality(PARENT, config["warm_start"])
    assert missing_result["status"] == BLOCKED_PREREQUISITE_STATUS
    assert missing_result["reason"] == "frozen_parent_development_evidence_missing"
    assert missing_result["evidence_available"] is False

    changed = copy.deepcopy(PARENT_DEV_EVIDENCE)
    changed["summary"]["sha256"] = "0" * 64
    monkeypatch.setattr(q_sft_frozen, "PARENT_DEV_EVIDENCE", changed)
    changed_result = assess_parent_prerequisite_legality(PARENT, config["warm_start"])
    assert changed_result["status"] == BLOCKED_PREREQUISITE_STATUS
    assert changed_result["reason"] == "frozen_parent_development_evidence_hash_mismatch"
    assert changed_result["evidence_available"] is False
    assert changed_result["evidence"]["summary"]["matched"] is False


def test_parent_gate_rejects_locked_test_evidence_declaration_before_read(monkeypatch):
    config = load_config()
    parent = validate_parent_adapter(PARENT, config["warm_start"])
    forbidden = copy.deepcopy(PARENT_DEV_EVIDENCE)
    forbidden["summary"]["path"] = "locked_test/summary.json"
    monkeypatch.setattr(q_sft_frozen, "PARENT_DEV_EVIDENCE", forbidden)
    with pytest.raises(GuardrailViolation, match="locked-test boundary"):
        assess_parent_prerequisite_legality(
            PARENT,
            config["warm_start"],
            validated_parent=parent,
        )


def test_live_dry_run_blocks_fixed_parent_without_writing(tmp_path: Path):
    config = load_config()
    before = set(tmp_path.iterdir())
    result = dry_run(config, parent_adapter=PARENT, config_path=DEFAULT_CONFIG)
    assert result["status"] == BLOCKED_PREREQUISITE_STATUS
    assert result["training_started"] is False
    assert result["run_directory_created"] is False
    assert result["prerequisite_legality_gate"]["passed"] is False
    assert result["prerequisite_legality_gate"]["failed_checks"]
    assert result["locked_test_access"] is False
    assert set(tmp_path.iterdir()) == before


def test_bundle_build_and_reaudit_are_content_addressed(tmp_path: Path):
    config = load_config()
    bundle = tmp_path / "bundle"
    built = build_bundle(config, bundle, config_path=DEFAULT_CONFIG)
    assert built["status"] == "passed"
    assert built["rows"] == 512
    repeated = audit_bundle(config, bundle, config_path=DEFAULT_CONFIG)
    assert repeated["snapshot_sha256"] == built["snapshot_sha256"]
    assert repeated["behavior_provenance"]["policy_id"] == BEHAVIOR_POLICY_ID
    assert repeated["behavior_provenance"]["source_states"] == 378
    assert repeated["behavior_provenance"]["behavior_actions"] == 378
    assert repeated["behavior_provenance"]["snapshot_rows"] == 512
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "q-sft-frozen-bundle-v2"
    assert manifest["behavior_contract"] == config["behavior_provenance"]
    assert manifest["behavior_provenance"] == repeated["behavior_provenance"]
    snapshots = [json.loads(line) for line in (bundle / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == SNAPSHOT_FIELDS for row in snapshots)
    (bundle / "snapshots.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(GuardrailViolation, match="deterministic rebuild"):
        audit_bundle(config, bundle, config_path=DEFAULT_CONFIG)


def test_cli_requires_parent_for_every_warm_started_action():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["dry-run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--run-dir", "run"])
    parsed = parser.parse_args(["dry-run", "--parent-adapter", str(PARENT)])
    assert parsed.parent_adapter == PARENT


def test_train_refuses_fixed_parent_before_core_or_run_directory(monkeypatch, tmp_path: Path):
    config = load_config()

    def forbidden_train(*args, **kwargs):
        raise AssertionError("core trainer must not run when the prerequisite gate fails")

    monkeypatch.setattr(q_sft_frozen, "train_q_sft", forbidden_train)
    result = q_sft_frozen.train(
        config,
        parent_adapter=PARENT,
        output_root=tmp_path,
        config_path=DEFAULT_CONFIG,
    )
    assert result["status"] == BLOCKED_PREREQUISITE_STATUS
    assert result["training_started"] is False
    assert result["run_directory_created"] is False
    assert result["prerequisite_legality_gate"]["evidence_available"] is True
    assert list(tmp_path.iterdir()) == []


def test_train_and_dev_evaluate_delegate_when_parent_is_eligible(monkeypatch, tmp_path: Path):
    config = load_config()
    calls: dict[str, object] = {}

    eligible = {
        "status": "prerequisite_legality_gate_passed",
        "passed": True,
        "reason": None,
        "parent_run_id": config["warm_start"]["expected_parent_run_id"],
        "evidence_available": True,
        "evidence": {},
        "thresholds": PREREQUISITE_THRESHOLDS,
        "checks": evaluate_prerequisite_thresholds(1.0, 0.0, 1.0)["checks"],
        "failed_checks": [],
        "locked_test_access": False,
    }
    monkeypatch.setattr(
        q_sft_frozen,
        "assess_parent_prerequisite_legality",
        lambda *args, **kwargs: copy.deepcopy(eligible),
    )

    def fake_train_q_sft(rows, parent_adapter, run_dir, spec):
        calls["train_rows"] = rows
        calls["parent_adapter"] = parent_adapter
        calls["train_spec"] = spec
        (run_dir / "checkpoints" / "final").mkdir(parents=True)
        (run_dir / "train_metrics.jsonl").write_text('{"optimizer_step":1,"train_loss":1.0}\n', encoding="utf-8")
        (run_dir / "accounting.json").write_text(
            json.dumps({"method": "q_sft", "optimizer_steps": 1}) + "\n",
            encoding="utf-8",
        )
        return object(), {"method": "q_sft", "optimizer_steps": 1}

    monkeypatch.setattr(q_sft_frozen, "train_q_sft", fake_train_q_sft)
    trained = q_sft_frozen.train(
        config,
        parent_adapter=PARENT,
        output_root=tmp_path,
        config_path=DEFAULT_CONFIG,
    )
    run_dir = Path(trained["run_dir"])
    assert len(calls["train_rows"]) == 512
    assert all(set(row) == JOINED_FIELDS for row in calls["train_rows"])
    assert calls["parent_adapter"] == PARENT.resolve()
    assert calls["train_spec"]["method"] == "q_sft"
    assert calls["train_spec"]["locked_test_access"] is False
    assert calls["train_spec"]["prerequisite_legality_gate"]["passed"] is True
    assert calls["train_spec"]["data"]["behavior_policy_id"] == BEHAVIOR_POLICY_ID
    assert len(calls["train_spec"]["data"]["behavior_state_supports_sha256"]) == 64

    def fake_evaluate(model, tokenizer, answers, allowed_words, answer_vocabulary):
        calls["dev_answers"] = list(answers)
        return (
            [{"game_id": index, "won": index % 2 == 0} for index, _ in enumerate(answers)],
            {
                "n_games": len(answers),
                "win_rate": 0.5,
                "terminal_marker_compliance": 1.0,
                "invalid_guess_rate": 0.0,
                "repeat_guess_rate": 0.0,
            },
        )

    def fake_diagnostics(model, tokenizer, dev_records, training_records, allowed_words, universe, output_parent):
        calls["diagnostic_records"] = list(dev_records)
        diagnostics_dir = output_parent / "diagnostics" / "mocked"
        diagnostics_dir.mkdir(parents=True)
        return diagnostics_dir, {
            "singleton_answer_accuracy": 0.9,
            "action_target_accuracy": 0.6,
            "train_state_coverage": 0.4,
            "by_turn": {"2": {"posterior_constraint_violation_rate": 0.2}},
        }

    def fake_retention(model, tokenizer, probes):
        calls["retention_probes"] = list(probes)
        return [{"probe_id": "mock", "correct": True}], {"probe_count": 1, "overall_score": 1.0}

    monkeypatch.setattr(q_sft_frozen, "load_tokenizer", lambda checkpoint: object())
    monkeypatch.setattr(q_sft_frozen, "load_adapter", lambda checkpoint: object())
    monkeypatch.setattr(q_sft_frozen, "evaluate", fake_evaluate)
    monkeypatch.setattr(q_sft_frozen, "run_state_diagnostics", fake_diagnostics)
    monkeypatch.setattr(q_sft_frozen, "evaluate_retention", fake_retention)

    evaluated = q_sft_frozen.evaluate_run(
        config,
        parent_adapter=PARENT,
        run_dir=run_dir,
        config_path=DEFAULT_CONFIG,
    )
    assert evaluated["status"] == "dev_evaluated"
    assert evaluated["split"] == "dev"
    assert evaluated["locked_test_access"] is False
    assert len(calls["dev_answers"]) == config["evaluation"]["dev_games"]
    assert len(calls["diagnostic_records"]) == config["evaluation"]["diagnostic_items"]
    assert calls["retention_probes"]
    assert evaluated["gate_metrics"]["turn_2_posterior_violation_rate"] == 0.2
    assert evaluated["gate_metrics"]["singleton_answer_accuracy"] == 0.9
    assert evaluated["artifact_manifest"]["provenance"]["split"] == "dev"
    assert not any(path.name.startswith("test_") for path in run_dir.iterdir())
