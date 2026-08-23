from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

import next_steps.chatgpt_2026_08_23.constraint_first_policy as constraint_policy
from next_steps.chatgpt_2026_08_23.constraint_first_policy import (
    CONSTRAINT_FIRST_CHECKPOINTS,
    DEFAULT_SOURCE,
    SAMPLED_MULTI_LABEL_OBJECTIVE,
    aggregate_constraint_first_doses,
    build_constraint_first_bundle,
    constraint_first_evaluation_policy,
    constraint_policy_spec,
)
from wordle_lab.analysis.state_diagnostics import score_probe_outputs
from wordle_lab.common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from wordle_lab.protocol.env import posterior_candidates, score_wordle


def test_constraint_first_targets_are_legal_nonrepeats_and_singletons_are_mandatory(tmp_path: Path):
    manifest = build_constraint_first_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    rows = read_jsonl(tmp_path / "train.jsonl")
    universe = read_json(DEFAULT_SOURCE / "universe.json")
    assert manifest["locked_test_access"] is False
    assert manifest["singleton_rows"] == 2 * manifest["singleton_source_states"]
    for row in rows:
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        posterior = posterior_candidates(history, universe)
        assert row["target_word"] in posterior
        assert row["target_word"] not in {guess for guess, _ in history}
        if row["posterior_size"] == 1:
            assert row["target_word"] == posterior[0]


def test_constraint_first_repeated_states_receive_diverse_legal_labels(tmp_path: Path):
    manifest = build_constraint_first_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    targets = defaultdict(set)
    for row in read_jsonl(tmp_path / "train.jsonl"):
        targets[row["state_id"]].add(row["target_word"])
    assert sum(len(words) > 1 for words in targets.values()) == manifest["states_with_multiple_legal_labels"]
    assert manifest["states_with_multiple_legal_labels"] == manifest["non_singleton_source_states"]
    assert manifest["states_meeting_declared_label_coverage"] == manifest["unique_source_states"]
    assert manifest["legal_labels_per_state"] == 4


def test_constraint_first_action_set_metadata_and_per_state_coverage_are_recomputed(tmp_path: Path):
    import hashlib
    import json

    manifest = build_constraint_first_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    universe = read_json(DEFAULT_SOURCE / "universe.json")
    rows = read_jsonl(tmp_path / "train.jsonl")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        repeated = {guess for guess, _ in history}
        acceptable = sorted(word for word in posterior_candidates(history, universe) if word not in repeated)
        expected_hash = hashlib.sha256(
            json.dumps(acceptable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert row["acceptable_action_count"] == len(acceptable)
        assert row["acceptable_action_set_sha256"] == expected_hash
    for state_rows in grouped.values():
        posterior_size = state_rows[0]["posterior_size"]
        expected_labels = 1 if posterior_size == 1 else min(4, posterior_size)
        assert len({row["target_word"] for row in state_rows}) == expected_labels
    assert len(grouped) == manifest["states_meeting_declared_label_coverage"]


def test_constraint_first_manifest_and_training_spec_are_content_addressed(tmp_path: Path):
    manifest = build_constraint_first_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    spec = constraint_policy_spec(tmp_path / "train.jsonl")
    assert manifest["rows_sha256"] == sha256_file(tmp_path / "train.jsonl")
    assert spec["data"]["sha256"] == manifest["rows_sha256"]
    assert spec["word_token_weight"] == 8.0
    assert spec["locked_test_access"] is False
    assert spec["candidate_injection"] is False
    assert "objective" not in spec
    assert set(spec["evaluation_contract"]) == {
        "allowed_words_sha256",
        "retention_probes_sha256",
        "canonical_sha256",
        "dev_secrets_sha256",
        "universe_sha256",
    }
    policy = constraint_first_evaluation_policy()
    assert policy["training_objective"]["description"] == SAMPLED_MULTI_LABEL_OBJECTIVE
    assert policy["training_objective"]["set_normalized_loss"] is False
    assert policy["checkpoints"] == list(CONSTRAINT_FIRST_CHECKPOINTS)
    assert policy["registration_status"] == "post_hoc_after_training_started"
    assert policy["preregistered"] is False


def _dose_summary(
    checkpoint: str,
    *,
    posterior: float = 0.20,
    turn_two: float = 0.20,
    singleton: float = 0.85,
    terminal: float = 1.0,
    invalid: float = 0.0,
    repeat: float = 0.05,
    gameplay_constraint: float = 0.20,
    retention: float = 0.25,
    win_rate: float = 0.25,
) -> dict:
    return {
        "status": "dev_evaluated",
        "experiment_id": constraint_policy.CONSTRAINT_POLICY_ID,
        "checkpoint": checkpoint,
        "split": "dev",
        "locked_test_access": False,
        "gameplay": {
            "terminal_marker_compliance": terminal,
            "invalid_guess_rate": invalid,
            "repeat_guess_rate": repeat,
            "constraint_violation_rate": gameplay_constraint,
            "win_rate": win_rate,
            "wins": round(32 * win_rate),
            "n_games": 32,
        },
        "diagnostics": {
            "posterior_constraint_violation_rate": posterior,
            "singleton_answer_accuracy": singleton,
            "action_target_accuracy": 0.20,
            "posterior_consistency": 1.0 - posterior,
            "by_turn": {"2": {"posterior_constraint_violation_rate": turn_two}},
        },
        "retention": {"overall_score": retention},
    }


def test_constraint_first_aggregate_selects_best_gate_passing_dose_deterministically():
    summaries = [
        _dose_summary("step-000150", posterior=0.24),
        _dose_summary("step-000300", posterior=0.10, turn_two=0.15),
        _dose_summary("step-000450", posterior=0.18),
        _dose_summary("step-000600", posterior=0.29),
    ]
    forward = aggregate_constraint_first_doses(summaries)
    reverse = aggregate_constraint_first_doses(list(reversed(summaries)))
    assert forward == reverse
    assert forward["selected_checkpoint"] == "step-000300"
    assert forward["replication_allowed"] is True
    assert forward["locked_test_access"] is False
    assert forward["locked_test_authorized"] is False
    assert forward["evaluation_policy"]["training_objective"]["set_normalized_loss"] is False
    assert forward["evaluation_policy"]["training_objective"]["description"] == SAMPLED_MULTI_LABEL_OBJECTIVE
    assert forward["evaluation_policy"]["preregistered"] is False
    assert forward["selection_contract"]["registration_status"] == "post_hoc_after_training_started"


def test_constraint_first_aggregate_records_no_promotable_dose_and_strict_turn_two_boundary():
    summaries = [
        _dose_summary(checkpoint, turn_two=0.30, singleton=0.79, retention=0.19, win_rate=0.20)
        for checkpoint in CONSTRAINT_FIRST_CHECKPOINTS
    ]
    result = aggregate_constraint_first_doses(summaries)
    assert result["selected_checkpoint"] is None
    assert result["promotable_checkpoints"] == []
    assert result["replication_allowed"] is False
    assert result["decision"] == "development_gates_failed_no_promotable_checkpoint_locked_test_closed"
    gates = result["doses"][0]["development_gates"]
    assert gates["passed"] is False
    assert gates["checks"]["turn_2_posterior_constraint_violation_rate"]["passed"] is False
    assert gates["groups"]["posterior"] is False
    assert gates["groups"]["singleton"] is False
    assert gates["groups"]["retention"] is False
    assert gates["groups"]["gameplay"] is False


def test_constraint_first_aggregate_requires_exact_four_dev_doses_and_closed_test():
    summaries = [_dose_summary(checkpoint) for checkpoint in CONSTRAINT_FIRST_CHECKPOINTS]
    with pytest.raises(ValueError, match="missing constraint-first checkpoint"):
        aggregate_constraint_first_doses(summaries[:-1])
    summaries[0]["locked_test_access"] = True
    with pytest.raises(ValueError, match="locked test closed"):
        aggregate_constraint_first_doses(summaries)


def test_constraint_first_evaluate_all_calls_each_frozen_dose_and_writes_aggregate(tmp_path: Path, monkeypatch):
    observed: list[str] = []

    def fake_evaluate(run_dir, checkpoint, *, source_dir, dev_games):
        assert Path(run_dir) == tmp_path
        assert source_dir == DEFAULT_SOURCE
        assert dev_games == 32
        observed.append(checkpoint)
        return _dose_summary(checkpoint, posterior=0.10 if checkpoint == "step-000300" else 0.20)

    monkeypatch.setattr(constraint_policy, "evaluate_constraint_checkpoint", fake_evaluate)
    result = constraint_policy.evaluate_constraint_doses(tmp_path, reuse_existing=False)
    assert observed == list(CONSTRAINT_FIRST_CHECKPOINTS)
    assert result["selected_checkpoint"] == "step-000300"
    written = read_json(tmp_path / "evaluation_summary.json")
    assert written == result


def _write_reusable_dose(tmp_path: Path, checkpoint: str = "step-000150") -> tuple[dict, dict]:
    dev_answers = read_json(DEFAULT_SOURCE / "dev_secrets.json")
    universe = read_json(DEFAULT_SOURCE / "universe.json")
    allowed = [
        line.strip().upper()
        for line in (constraint_policy.ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    retention_probes = read_jsonl(constraint_policy.DATA / "retention_probes_v1.jsonl")
    diagnostic_inputs = []
    for index in range(constraint_policy.CONSTRAINT_FIRST_DIAGNOSTIC_ITEMS):
        secret = dev_answers[index % len(dev_answers)]
        history = [{"guess": "SHARE", "feedback": score_wordle(secret, "SHARE")}]
        posterior = posterior_candidates([(history[0]["guess"], history[0]["feedback"])], universe)
        diagnostic_inputs.append(
            {
                "item_id": f"synthetic-diagnostic-{index:03d}",
                "history": history,
                "turn": 2,
                "posterior_size": len(posterior),
                "oracle_action": secret,
                "train_state_seen": False,
                "secret_answer": secret,
            }
        )
    diagnostic_rows, diagnostic_summary = score_probe_outputs(
        diagnostic_inputs,
        [{"raw_output": f"Final answer: {item['secret_answer']}"} for item in diagnostic_inputs],
        allowed,
        universe,
    )

    checkpoint_dir = tmp_path / "checkpoints" / checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        tmp_path / "summary.json",
        {"status": "trained", "run_dir": str(tmp_path.resolve()), "locked_test_access": False},
    )
    games = [
        {
            "game_id": index,
            "answer": answer,
            "won": index < 8,
            "turns": [
                {
                    "format_valid": True,
                    "valid": True,
                    "repeat": False,
                    "constraint_violation": False,
                }
            ],
        }
        for index, answer in enumerate(dev_answers)
    ]
    write_jsonl(tmp_path / f"eval-{checkpoint}-games.jsonl", games)
    artifact_id = "synthetic001"
    diagnostics_dir = tmp_path / f"eval-{checkpoint}" / "diagnostics" / artifact_id
    diagnostic_items_path = write_jsonl(diagnostics_dir / "items.jsonl", diagnostic_rows)
    diagnostic_summary = {
        **diagnostic_summary,
        "artifact_id": artifact_id,
        "items_sha256": sha256_file(diagnostic_items_path),
    }
    write_json(diagnostics_dir / "summary.json", diagnostic_summary)
    retention_rows = [
        {
            **probe,
            "raw_output": probe["expected"],
            "normalized_output": probe["expected"],
            "correct": True,
        }
        for probe in retention_probes
    ]
    write_jsonl(tmp_path / f"eval-{checkpoint}-retention.jsonl", retention_rows)
    categories = sorted({probe["category"] for probe in retention_probes})
    context = {
        "spec": {
            "protocol_id": "WORDLE-PROTOCOL-002",
            "protocol_sha256": "a" * 64,
            "protocol_lock_file_sha256": "b" * 64,
            "evaluation_contract": {"dev_secrets_sha256": "c" * 64},
        },
        "dev_answers": dev_answers,
        "universe": universe,
        "allowed": allowed,
        "retention_probes": retention_probes,
        "diagnostic_inputs": diagnostic_inputs,
        "binding": {"schema_version": "synthetic-evaluation-inputs-v1"},
    }
    summary = {
        "status": "dev_evaluated",
        "experiment_id": constraint_policy.CONSTRAINT_POLICY_ID,
        "checkpoint": checkpoint,
        "split": "dev",
        "locked_test_access": False,
        "gameplay": constraint_policy._recomputed_gameplay_gate_metrics(games),
        "diagnostics": diagnostic_summary,
        "retention": {
            "probe_count": len(retention_rows),
            "overall_score": 1.0,
            "category_scores": {category: 1.0 for category in categories},
        },
    }
    return summary, context


def test_constraint_first_reused_summary_is_recomputed_from_all_frozen_raw_artifacts(tmp_path: Path):
    checkpoint = "step-000150"
    summary, context = _write_reusable_dose(tmp_path, checkpoint)
    result = constraint_policy.validate_reused_constraint_summary(
        tmp_path,
        checkpoint,
        summary,
        evaluation_context=context,
    )
    assert result["status"] == "passed"
    assert result["artifact_integrity"]["games"]["rows"] == 32
    assert result["artifact_integrity"]["diagnostics"]["rows"] == 128
    assert result["artifact_integrity"]["retention"]["rows"] == 200
    assert result["locked_test_access"] is False


def test_constraint_first_evaluation_context_binds_exact_frozen_inputs_and_hashes(tmp_path: Path):
    write_json(tmp_path / "spec.json", constraint_policy_spec())
    context = constraint_policy._constraint_evaluation_context(tmp_path)
    binding = context["binding"]
    assert binding["dev_games"] == 32
    assert binding["diagnostic_items"] == 128
    assert binding["retention_probes"] == 200
    assert binding["split"] == "dev"
    assert binding["evaluation_contract"] == context["spec"]["evaluation_contract"]
    assert len(binding["dev_answer_order_sha256"]) == 64
    assert len(binding["diagnostic_inputs_sha256"]) == 64
    assert binding["locked_test_access"] is False


def test_constraint_first_reused_summary_rejects_wrong_game_set_and_optional_hash_binding(tmp_path: Path):
    checkpoint = "step-000150"
    summary, context = _write_reusable_dose(tmp_path, checkpoint)
    games_path = tmp_path / f"eval-{checkpoint}-games.jsonl"
    games = read_jsonl(games_path)
    games[0]["answer"] = games[1]["answer"]
    write_jsonl(games_path, games)
    with pytest.raises(RuntimeError, match="exact frozen 32-game"):
        constraint_policy.validate_reused_constraint_summary(
            tmp_path,
            checkpoint,
            summary,
            evaluation_context=context,
        )

    summary, context = _write_reusable_dose(tmp_path, checkpoint)
    summary["protocol_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="protocol_sha256 binding mismatch"):
        constraint_policy.validate_reused_constraint_summary(
            tmp_path,
            checkpoint,
            summary,
            evaluation_context=context,
        )


def test_constraint_first_reused_summary_rejects_incomplete_diagnostics_and_retention(tmp_path: Path):
    checkpoint = "step-000150"
    summary, context = _write_reusable_dose(tmp_path, checkpoint)
    diagnostic_path = tmp_path / f"eval-{checkpoint}" / "diagnostics" / "synthetic001" / "items.jsonl"
    write_jsonl(diagnostic_path, read_jsonl(diagnostic_path)[:-1])
    with pytest.raises(RuntimeError, match="exactly 128"):
        constraint_policy.validate_reused_constraint_summary(
            tmp_path,
            checkpoint,
            summary,
            evaluation_context=context,
        )

    summary, context = _write_reusable_dose(tmp_path, checkpoint)
    retention_path = tmp_path / f"eval-{checkpoint}-retention.jsonl"
    write_jsonl(retention_path, read_jsonl(retention_path)[:-1])
    with pytest.raises(RuntimeError, match="exactly 200"):
        constraint_policy.validate_reused_constraint_summary(
            tmp_path,
            checkpoint,
            summary,
            evaluation_context=context,
        )


def test_constraint_first_evaluate_all_rejects_non_frozen_dev_size_before_evaluation(tmp_path: Path, monkeypatch):
    called = False

    def forbidden_evaluate(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluation must not start")

    monkeypatch.setattr(constraint_policy, "evaluate_constraint_checkpoint", forbidden_evaluate)
    with pytest.raises(RuntimeError, match="exactly 32 frozen development games"):
        constraint_policy.evaluate_constraint_doses(tmp_path, dev_games=31, reuse_existing=False)
    assert called is False
