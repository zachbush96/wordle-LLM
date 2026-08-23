from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from next_steps.chatgpt_2026_08_23 import gemma_1b_capacity as capacity
from next_steps.chatgpt_2026_08_23 import constraint_first_policy as constraint_policy


EXPECTED_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
EXPECTED_DATA_SHA256 = "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f"


def load_config() -> dict:
    return json.loads(capacity.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def bound_capacity_spec() -> dict:
    return {
        "experiment_id": capacity.CAPACITY_ID,
        "seed": 2026,
        "model": {"model_id": capacity.MODEL_ID, "revision": capacity.MODEL_REVISION},
        "matched_270m_comparator": {"run_id": capacity.EXPECTED_COMPARATOR["run_id"]},
        "data": {
            "rows_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "state_manifest_sha256": "3" * 64,
            "canonical_sha256": "4" * 64,
            "universe_sha256": "5" * 64,
            "train_secrets_sha256": "6" * 64,
            "dev_secrets_sha256": "7" * 64,
            "allowed_words_sha256": "8" * 64,
            "retention_probes_sha256": "9" * 64,
        },
        "protocol_id": "WORDLE-PROTOCOL-002",
        "protocol_sha256": "a" * 64,
        "protocol_lock_file_sha256": "b" * 64,
        "evaluation": capacity.capacity_evaluation_policy(),
        "locked_test_access": False,
    }


def dose_summary(
    checkpoint: str,
    spec: dict,
    *,
    posterior: float = 0.10,
    turn_two: float = 0.10,
    singleton: float = 0.90,
    run_dir: Path | None = None,
) -> dict:
    summary = {
        "status": "dev_evaluated",
        "experiment_id": capacity.CAPACITY_ID,
        "model": spec["model"],
        "matched_270m_comparator": spec["matched_270m_comparator"],
        "checkpoint": checkpoint,
        "split": capacity.CAPACITY_SPLIT,
        "decoder": "greedy",
        "prompt_variant": "explicit_feedback",
        "run_spec_sha256": capacity._capacity_spec_sha256(spec),
        "evaluation_data": capacity._capacity_evaluation_data_binding(spec),
        "evaluation_policy": capacity.capacity_evaluation_policy(),
        "locked_test_access": False,
        "locked_test_authorized": False,
        "gameplay": {
            "terminal_marker_compliance": 1.0,
            "invalid_guess_rate": 0.0,
            "repeat_guess_rate": 0.0,
            "constraint_violation_rate": 0.10,
            "win_rate": 0.50,
            "wins": 16,
            "n_games": 32,
        },
        "diagnostics": {
            "posterior_constraint_violation_rate": posterior,
            "by_turn": {"2": {"posterior_constraint_violation_rate": turn_two}},
            "singleton_answer_accuracy": singleton,
            "action_target_accuracy": 0.25,
            "posterior_consistency": 1.0 - posterior,
        },
        "retention": {"overall_score": 0.30},
    }
    if run_dir is not None:
        games_path = run_dir / f"eval-{checkpoint}-games.jsonl"
        retention_path = run_dir / f"eval-{checkpoint}-retention.jsonl"
        games_path.write_text('{"game_id":0,"split":"dev"}\n', encoding="utf-8")
        retention_path.write_text('{"probe_id":0,"correct":true}\n', encoding="utf-8")
        summary["evaluation_artifacts"] = {
            "games": {"path": games_path.name, "sha256": capacity.sha256_file(games_path)},
            "retention": {
                "path": retention_path.name,
                "sha256": capacity.sha256_file(retention_path),
            },
        }
    summary["development_gates"] = capacity.capacity_gate_status(summary)
    return summary


def test_config_is_the_exact_matched_balanced_002_recipe_and_pinned_revision() -> None:
    config = load_config()
    capacity.validate_capacity_config(config)
    assert capacity.CAPACITY_ID == "GEMMA-SAME-FAMILY-CAPACITY-001"
    assert capacity.MODEL_ID == "google/gemma-3-1b-it"
    assert capacity.MODEL_REVISION == EXPECTED_REVISION
    assert len(capacity.MODEL_REVISION) == 40
    assert config == {
        "experiment_id": "GEMMA-SAME-FAMILY-CAPACITY-001",
        "model_id": "google/gemma-3-1b-it",
        "model_revision": EXPECTED_REVISION,
        "source_model_revision_checked_at": "2026-08-23",
        "training_backend": "transformers_peft",
        "matched_270m_comparator": capacity.EXPECTED_COMPARATOR,
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "training_rows": 512,
        "training_rows_sha256": EXPECTED_DATA_SHA256,
        "seed": 2026,
        "max_steps": 600,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "max_length": 320,
        "word_token_weight": 8.0,
        "checkpoint_steps": [150, 300, 450, 600],
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "google/gemma-3-270m-it"),
        ("model_revision", "0" * 40),
        ("source_model_revision_checked_at", "1999-01-01"),
        ("training_backend", "unsloth"),
        ("curriculum_id", "COMMON-WORD-CURRICULUM-003"),
        ("training_rows", 1024),
        ("training_rows_sha256", "0" * 64),
        ("seed", 2027),
        ("max_steps", 601),
        ("learning_rate", 1e-4),
        ("batch_size", 2),
        ("gradient_accumulation_steps", 2),
        ("effective_batch_size", 8),
        ("word_token_weight", 1.0),
        ("locked_test_access", True),
        ("candidate_injection", True),
        ("vocabulary_masking", True),
        ("reranking", True),
        ("repeat_ban", True),
        ("output_repair", True),
    ],
)
def test_config_validation_fails_closed_on_recipe_or_protocol_drift(field: str, value: object) -> None:
    config = deepcopy(load_config())
    config[field] = value
    with pytest.raises(ValueError, match="capacity config drift"):
        capacity.validate_capacity_config(config)


def test_config_validation_rejects_effective_batch_arithmetic_drift() -> None:
    config = deepcopy(load_config())
    config["gradient_accumulation_steps"] = 4
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        capacity.validate_capacity_config(config)


def test_config_validation_rejects_unexpected_keys() -> None:
    config = deepcopy(load_config())
    config["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected_keys"):
        capacity.validate_capacity_config(config)


@pytest.mark.parametrize("field", ["r", "alpha", "dropout", "target_modules"])
def test_config_validation_rejects_lora_drift(field: str) -> None:
    config = deepcopy(load_config())
    config["lora"][field] = 1 if field != "target_modules" else ["q_proj"]
    with pytest.raises(ValueError, match="lora"):
        capacity.validate_capacity_config(config)


def test_capacity_spec_audits_matched_data_without_reading_locked_test(monkeypatch: pytest.MonkeyPatch) -> None:
    paths_read: list[Path] = []
    original_read_json = capacity.read_json

    def guarded_read_json(path: str | Path):
        resolved = Path(path).resolve()
        assert resolved.name != "test_answers.json"
        paths_read.append(resolved)
        return original_read_json(resolved)

    monkeypatch.setattr(capacity, "read_json", guarded_read_json)
    spec = capacity.capacity_spec(load_config())
    assert spec["experiment_id"] == "GEMMA-SAME-FAMILY-CAPACITY-001"
    assert spec["model"] == {
        "model_id": "google/gemma-3-1b-it",
        "revision": EXPECTED_REVISION,
        "local_path": str(capacity.DEFAULT_MODEL_DIR),
    }
    assert spec["seed"] == 2026
    assert spec["max_steps"] == 600
    assert spec["learning_rate"] == 5e-5
    assert spec["batch_size"] * spec["gradient_accumulation_steps"] == 4
    assert spec["batch_size"] == 4
    assert spec["gradient_accumulation_steps"] == 1
    assert spec["word_token_weight"] == 8.0
    assert spec["data"] == {
        "status": "passed",
        "rows": 512,
        "rows_sha256": EXPECTED_DATA_SHA256,
        "manifest_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "manifest.json"),
        "state_manifest_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "state_manifest.jsonl"),
        "canonical_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "canonical.jsonl"),
        "universe_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "universe.json"),
        "train_secrets_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "train_secrets.json"),
        "dev_secrets_sha256": capacity.sha256_file(capacity.DEFAULT_DATA / "dev_secrets.json"),
        "allowed_words_sha256": capacity.sha256_file(capacity.ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt"),
        "retention_probes_sha256": capacity.sha256_file(capacity.DATA / "retention_probes_v1.jsonl"),
        "train_secrets": 96,
        "dev_secrets": 32,
        "locked_test_access": False,
    }
    assert spec["protocol_id"] == "WORDLE-PROTOCOL-002"
    assert spec["locked_test_access"] is False
    assert spec["candidate_injection"] is False
    assert spec["reranking"] is False
    assert spec["output_repair"] is False
    assert spec["matched_270m_comparator"]["status"] == "passed"
    assert all(spec["matched_270m_comparator"]["recipe_checks"].values())
    assert paths_read
    assert not any("test_answers" in path.name for path in paths_read)


def test_dry_run_validates_before_preflight_and_never_dispatches_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = {
        "status": "blocked_missing_huggingface_auth_for_gated_model",
        "ready": False,
        "locked_test_access": False,
        "credentials_recorded": False,
    }
    spec = {"experiment_id": capacity.CAPACITY_ID, "locked_test_access": False}
    monkeypatch.setattr(capacity, "capacity_preflight", lambda: blocker)
    monkeypatch.setattr(capacity, "capacity_spec", lambda config: spec)
    monkeypatch.setattr(
        capacity,
        "train_capacity",
        lambda *_args, **_kwargs: pytest.fail("dry-run dispatched training"),
    )
    monkeypatch.setattr(
        capacity,
        "provision_model",
        lambda *_args, **_kwargs: pytest.fail("dry-run dispatched provisioning"),
    )
    assert capacity.main(["dry-run", "--config", str(capacity.DEFAULT_CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"preflight": blocker, "spec": spec, "status": "dry_run_passed"}


def test_dry_run_rejects_bad_config_before_any_remote_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(load_config())
    config["model_revision"] = "0" * 40
    path = tmp_path / "bad-capacity-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        capacity,
        "capacity_preflight",
        lambda: pytest.fail("invalid config reached remote preflight"),
    )
    with pytest.raises(ValueError, match="capacity config drift"):
        capacity.main(["dry-run", "--config", str(path)])


def test_preflight_reports_current_gated_no_auth_blocker_without_secrets_or_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeApi:
        def model_info(self, model_id: str, *, revision: str, token: str | None):
            calls.update(model_id=model_id, revision=revision, token=token)
            return SimpleNamespace(sha=EXPECTED_REVISION, gated="manual", private=False)

    written: dict[str, object] = {}
    monkeypatch.setattr(capacity, "get_token", lambda: None)
    monkeypatch.setattr(capacity, "HfApi", FakeApi)
    monkeypatch.setattr(
        capacity,
        "read_json",
        lambda _path: pytest.fail("missing-model preflight read local or locked-test data"),
    )
    monkeypatch.setattr(
        capacity,
        "write_json",
        lambda path, payload: written.update(path=Path(path), payload=payload),
    )
    output = tmp_path / "preflight.json"
    result = capacity.capacity_preflight(model_dir=tmp_path / "missing-model", output_path=output)
    assert calls == {
        "model_id": "google/gemma-3-1b-it",
        "revision": EXPECTED_REVISION,
        "token": None,
    }
    assert result["gated"] == "manual"
    assert result["remote_revision_matches"] is True
    assert result["huggingface_authenticated"] is False
    assert result["local"] == {"available": False, "reason": "local_snapshot_missing"}
    assert result["ready"] is False
    assert result["status"] == "blocked_missing_huggingface_auth_for_gated_model"
    assert result["locked_test_access"] is False
    assert result["credentials_recorded"] is False
    assert written == {"path": output, "payload": result}
    assert "token" not in result
    assert "credential" not in result


def test_preflight_uses_auth_for_remote_check_but_never_serializes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf_DO_NOT_SERIALIZE_THIS_VALUE"
    seen: dict[str, object] = {}

    class FakeApi:
        def model_info(self, model_id: str, *, revision: str, token: str | None):
            seen["token"] = token
            return SimpleNamespace(sha=EXPECTED_REVISION, gated="manual", private=False)

    monkeypatch.setattr(capacity, "get_token", lambda: secret)
    monkeypatch.setattr(capacity, "HfApi", FakeApi)
    result = capacity.capacity_preflight(model_dir=tmp_path / "missing-model", output_path=None)
    assert seen["token"] == secret
    assert result["huggingface_authenticated"] is True
    assert result["status"] == "provisioning_required"
    assert result["credentials_recorded"] is False
    assert secret not in json.dumps(result, sort_keys=True)


def test_evaluation_policy_matches_complete_270m_contract_and_all_doses() -> None:
    policy = capacity.capacity_evaluation_policy()
    assert policy["checkpoints"] == [
        "step-000150",
        "step-000300",
        "step-000450",
        "step-000600",
    ]
    assert policy["gate_thresholds"] == constraint_policy.CONSTRAINT_FIRST_GATE_THRESHOLDS
    assert policy["split"] == "balanced_002_dev_32"
    assert policy["dev_games"] == 32
    assert policy["diagnostic_items"] == 128
    assert policy["singleton_correctness_mandatory"] is True
    assert policy["registration_status"] == "declared_before_training"
    assert policy["preregistered"] is True
    assert policy["locked_test_access"] is False
    assert policy["locked_test_authorized"] is False


def test_capacity_gate_status_requires_every_group_and_singleton_correctness() -> None:
    spec = bound_capacity_spec()
    summary = dose_summary("step-000150", spec)
    gates = capacity.capacity_gate_status(summary)
    assert gates["passed"] is True
    assert gates["groups"] == {
        "format": True,
        "gameplay": True,
        "legality": True,
        "posterior": True,
        "retention": True,
        "singleton": True,
    }
    assert set(gates["checks"]) == set(capacity.CAPACITY_GATE_THRESHOLDS)
    assert gates["singleton_correctness_mandatory"] is True

    summary["diagnostics"]["singleton_answer_accuracy"] = 0.79
    failed = capacity.capacity_gate_status(summary)
    assert failed["passed"] is False
    assert failed["groups"]["singleton"] is False
    assert "threshold_failed:singleton_answer_accuracy" in failed["failures"]


def test_aggregate_capacity_doses_selects_best_passing_dose_and_keeps_test_closed() -> None:
    spec = bound_capacity_spec()
    summaries = [
        dose_summary("step-000150", spec, posterior=0.20, turn_two=0.20),
        dose_summary("step-000300", spec, posterior=0.10, turn_two=0.20),
        dose_summary("step-000450", spec, posterior=0.10, turn_two=0.05),
        dose_summary("step-000600", spec, posterior=0.01, turn_two=0.01, singleton=0.79),
    ]
    result = capacity.aggregate_capacity_doses(summaries, spec=spec)
    assert result["status"] == "evaluation_complete"
    assert result["experiment_id"] == capacity.CAPACITY_ID
    assert result["split"] == capacity.CAPACITY_SPLIT
    assert result["checkpoints"] == list(capacity.CAPACITY_CHECKPOINTS)
    assert result["selected_checkpoint"] == "step-000450"
    assert result["promotable_checkpoints"] == [
        "step-000450",
        "step-000300",
        "step-000150",
    ]
    assert result["replication_allowed"] is True
    assert result["locked_test_access"] is False
    assert result["locked_test_authorized"] is False
    assert result["selection_contract"]["singleton_correctness_mandatory"] is True
    assert result["decision"].endswith("locked_test_closed")


def test_aggregate_capacity_doses_rejects_missing_duplicate_and_binding_drift() -> None:
    spec = bound_capacity_spec()
    summaries = [dose_summary(checkpoint, spec) for checkpoint in capacity.CAPACITY_CHECKPOINTS]
    with pytest.raises(ValueError, match="missing Gemma 1B checkpoint summaries"):
        capacity.aggregate_capacity_doses(summaries[:-1], spec=spec)
    with pytest.raises(ValueError, match="duplicate Gemma 1B checkpoint summary"):
        capacity.aggregate_capacity_doses([summaries[0], summaries[0], *summaries[1:]], spec=spec)

    drifted = deepcopy(summaries)
    drifted[1]["split"] = "anything_else"
    with pytest.raises(ValueError, match="binding drift|development split"):
        capacity.aggregate_capacity_doses(drifted, spec=spec)

    drifted = deepcopy(summaries)
    drifted[2]["development_gates"]["passed"] = False
    with pytest.raises(ValueError, match="stored Gemma 1B development gates drift"):
        capacity.aggregate_capacity_doses(drifted, spec=spec)


def test_evaluate_capacity_checkpoint_cpu_mock_writes_bound_development_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = bound_capacity_spec()
    run_dir = tmp_path / "run"
    checkpoint = "step-000150"
    (run_dir / "checkpoints" / checkpoint).mkdir(parents=True)
    capacity.write_json(run_dir / "spec.json", spec)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    capacity.write_json(data_dir / "universe.json", ["ABOUT", "OTHER"])
    capacity.write_json(data_dir / "dev_secrets.json", [f"W{index:04d}" for index in range(32)])

    gameplay = {
        "terminal_marker_compliance": 1.0,
        "invalid_guess_rate": 0.0,
        "repeat_guess_rate": 0.0,
        "constraint_violation_rate": 0.1,
        "win_rate": 0.5,
        "wins": 16,
        "n_games": 32,
    }
    diagnostics = {
        "posterior_constraint_violation_rate": 0.1,
        "by_turn": {"2": {"posterior_constraint_violation_rate": 0.1}},
        "singleton_answer_accuracy": 0.9,
    }
    calls: dict[str, object] = {}
    monkeypatch.setattr(capacity, "capacity_spec", lambda _config, _data_dir: spec)
    monkeypatch.setattr(capacity, "_load_capacity_adapter", lambda path: (object(), object()))
    monkeypatch.setattr(
        capacity,
        "evaluate",
        lambda *_args: ([{"game_id": 0, "won": True}], gameplay),
    )
    monkeypatch.setattr(capacity, "generate_canonical_states", lambda *_args, **_kwargs: [])

    def fake_diagnostics(*args, **kwargs):
        calls["diagnostics_parent"] = Path(args[-1])
        directory = tmp_path / "diagnostics"
        directory.mkdir()
        return directory, diagnostics

    monkeypatch.setattr(capacity, "run_state_diagnostics", fake_diagnostics)
    monkeypatch.setattr(
        capacity,
        "evaluate_retention",
        lambda *_args: ([{"probe_id": 0, "correct": True}], {"overall_score": 0.3}),
    )
    monkeypatch.setattr(capacity, "read_jsonl", lambda _path: [])
    monkeypatch.setattr(capacity, "set_seed", lambda seed: calls.update(seed=seed))
    monkeypatch.setattr(capacity.torch.cuda, "empty_cache", lambda: None)

    result = capacity.evaluate_capacity_checkpoint(run_dir, checkpoint, data_dir)
    assert result["experiment_id"] == capacity.CAPACITY_ID
    assert result["checkpoint"] == checkpoint
    assert result["split"] == capacity.CAPACITY_SPLIT
    assert result["run_spec_sha256"] == capacity._capacity_spec_sha256(spec)
    assert result["evaluation_data"] == capacity._capacity_evaluation_data_binding(spec)
    assert result["development_gates"]["passed"] is True
    assert result["locked_test_access"] is False
    assert result["locked_test_authorized"] is False
    assert calls == {"seed": 2026, "diagnostics_parent": run_dir / f"eval-{checkpoint}"}
    assert (run_dir / f"eval-{checkpoint}-summary.json").is_file()
    capacity._validate_reuse_artifacts(result, run_dir, checkpoint)


def test_evaluate_capacity_doses_cpu_mock_runs_all_doses_then_reuses_hash_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = bound_capacity_spec()
    run_dir = tmp_path / "run"
    for checkpoint in capacity.CAPACITY_CHECKPOINTS:
        (run_dir / "checkpoints" / checkpoint).mkdir(parents=True)
    capacity.write_json(run_dir / "spec.json", spec)
    monkeypatch.setattr(capacity, "capacity_spec", lambda _config, _data_dir: spec)
    calls: list[str] = []

    def fake_evaluate(destination: Path, checkpoint: str, _data_dir: Path):
        calls.append(checkpoint)
        summary = dose_summary(checkpoint, spec, run_dir=Path(destination))
        capacity.write_json(Path(destination) / f"eval-{checkpoint}-summary.json", summary)
        return summary

    monkeypatch.setattr(capacity, "evaluate_capacity_checkpoint", fake_evaluate)
    result = capacity.evaluate_capacity_doses(run_dir, data_dir=tmp_path / "data")
    assert calls == list(capacity.CAPACITY_CHECKPOINTS)
    assert result["selected_checkpoint"] == "step-000150"
    assert result["run_dir"] == str(run_dir.resolve())
    assert (run_dir / "evaluation_summary.json").is_file()

    monkeypatch.setattr(
        capacity,
        "evaluate_capacity_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("validated reuse dispatched model evaluation"),
    )
    reused = capacity.evaluate_capacity_doses(run_dir, data_dir=tmp_path / "data")
    assert reused == result

    games = run_dir / "eval-step-000300-games.jsonl"
    games.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="reused games hash drift"):
        capacity.evaluate_capacity_doses(run_dir, data_dir=tmp_path / "data")


def test_evaluate_capacity_doses_rejects_stale_reuse_before_any_missing_dose_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = bound_capacity_spec()
    run_dir = tmp_path / "run"
    for checkpoint in capacity.CAPACITY_CHECKPOINTS:
        (run_dir / "checkpoints" / checkpoint).mkdir(parents=True)
    capacity.write_json(run_dir / "spec.json", spec)
    stale = dose_summary("step-000150", spec, run_dir=run_dir)
    stale["split"] = "not-the-frozen-development-split"
    capacity.write_json(run_dir / "eval-step-000150-summary.json", stale)
    monkeypatch.setattr(capacity, "capacity_spec", lambda _config, _data_dir: spec)
    monkeypatch.setattr(
        capacity,
        "evaluate_capacity_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("stale reuse reached an unevaluated dose"),
    )
    with pytest.raises(ValueError, match="binding drift|development split"):
        capacity.evaluate_capacity_doses(run_dir, data_dir=tmp_path / "data")


def test_evaluate_all_cli_dispatches_four_dose_orchestration_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "status": "evaluation_complete",
        "experiment_id": capacity.CAPACITY_ID,
        "locked_test_access": False,
    }
    calls: dict[str, object] = {}

    def fake_all(run_dir: Path, *, reuse_existing: bool):
        calls.update(run_dir=Path(run_dir), reuse_existing=reuse_existing)
        return expected

    monkeypatch.setattr(capacity, "evaluate_capacity_doses", fake_all)
    assert capacity.main(["evaluate-all", "--run-dir", str(tmp_path), "--no-reuse-existing"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert calls == {"run_dir": tmp_path, "reuse_existing": False}
