from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.collect_results import (
    CollectionViolation,
    OUTPUT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    collect_results,
    main,
    template_manifest,
    validate_collection_manifest,
)
from wordle_lab.common import canonical_json


def _artifact(source: str, role: str, scope: str, *, required: bool = True, max_bytes: int = 4096) -> dict:
    return {
        "source": source,
        "destination": source,
        "role": role,
        "scope": scope,
        "required": required,
        "max_bytes": max_bytes,
    }


def _manifest(experiments: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": "unit-results",
        "locked_test_access": False,
        "experiments": experiments,
    }


def _complete_experiment(run_name: str = "run-complete") -> dict:
    return {
        "experiment_id": "complete-cell",
        "run_dir": f"artifacts/runs/{run_name}",
        "status": "complete",
        "artifacts": [
            _artifact("spec.json", "spec", "metadata"),
            _artifact("accounting.json", "accounting", "train"),
            _artifact("train_metrics.jsonl", "train_metrics", "train"),
            _artifact("evaluation/summary.json", "eval_summary", "dev"),
            _artifact("comparison/summary.json", "comparison_summary", "dev"),
            _artifact("gate_metrics.json", "gate_metrics", "dev"),
            _artifact("games.jsonl", "raw_jsonl", "dev", required=False),
        ],
    }


def _write_mock_run(workspace: Path, run_name: str = "run-complete") -> Path:
    run = workspace / "artifacts" / "runs" / run_name
    (run / "evaluation").mkdir(parents=True)
    (run / "comparison").mkdir(parents=True)
    (run / "checkpoints" / "final").mkdir(parents=True)
    (run / "spec.json").write_text(
        '{"seed":2026,"locked_test_access":false,"split":"dev"}\n', encoding="utf-8"
    )
    (run / "accounting.json").write_text('{"optimizer_steps":10}\n', encoding="utf-8")
    (run / "train_metrics.jsonl").write_text(
        '{"train_loss":2.0,"optimizer_step":1}\n{"optimizer_step":10,"train_loss":0.1}\n',
        encoding="utf-8",
    )
    (run / "evaluation" / "summary.json").write_text(
        '{"split":"dev","win_rate":0.25,"locked_test_access":false}\n', encoding="utf-8"
    )
    (run / "comparison" / "summary.json").write_text(
        '{"split":"dev","deltas":{"mean_target_rank":-3}}\n', encoding="utf-8"
    )
    (run / "gate_metrics.json").write_text(
        '{"split":"dev","invalid_guess_rate":0.0,"locked_test_access":false}\n', encoding="utf-8"
    )
    (run / "games.jsonl").write_text(
        '{"game_id":0,"answer":"ABOUT","won":true}\n', encoding="utf-8"
    )
    (run / "checkpoints" / "final" / "adapter_model.safetensors").write_bytes(b"model weights")
    return run


def test_collects_only_declared_compact_evidence_with_source_and_output_hashes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run = _write_mock_run(workspace)
    output = tmp_path / "results"
    result = collect_results(_manifest([_complete_experiment()]), output, workspace_root=workspace)

    assert result["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert result["status"] == "complete"
    assert result["experiments_available"] == 1
    assert result["files_available"] == 7
    report = result["experiments"][0]
    assert report["availability"] == "available"
    assert report["metrics"] is None
    assert all(artifact["availability"] == "available" for artifact in report["artifacts"])
    assert not any("checkpoint" in str(path).lower() for path in output.rglob("*"))

    spec_output = output / "complete-cell" / "spec.json"
    assert json.loads(spec_output.read_text(encoding="utf-8"))["seed"] == 2026
    assert spec_output.read_text(encoding="utf-8").startswith("{\n  \"locked_test_access\"")
    metrics_output = output / "complete-cell" / "train_metrics.jsonl"
    assert metrics_output.read_text(encoding="utf-8").splitlines()[0] == (
        '{"optimizer_step":1,"train_loss":2.0}'
    )
    gate = json.loads((output / "complete-cell" / "gate_metrics.json").read_text(encoding="utf-8"))
    assert gate["invalid_guess_rate"] == 0.0

    spec_record = next(artifact for artifact in report["artifacts"] if artifact["role"] == "spec")
    assert spec_record["source_sha256"] == hashlib.sha256((run / "spec.json").read_bytes()).hexdigest()
    assert spec_record["collected_sha256"] == hashlib.sha256(spec_output.read_bytes()).hexdigest()
    assert spec_record["source_bytes"] == (run / "spec.json").stat().st_size
    assert spec_record["collected_bytes"] == spec_output.stat().st_size
    manifest_output = json.loads((output / "collection_manifest.json").read_text(encoding="utf-8"))
    content_hash = manifest_output.pop("manifest_content_sha256")
    assert content_hash == hashlib.sha256(canonical_json(manifest_output).encode("utf-8")).hexdigest()


def test_missing_and_blocked_experiments_are_unavailable_not_zero(tmp_path: Path):
    workspace = tmp_path / "workspace"
    experiments = [
        {
            "experiment_id": "missing-run",
            "run_dir": "artifacts/runs/not-created",
            "status": "expected",
            "artifacts": [_artifact("summary.json", "eval_summary", "dev")],
        },
        {
            "experiment_id": "blocked-run",
            "status": "blocked",
            "reason": "GPU runtime did not finish",
            "artifacts": [],
        },
    ]
    result = collect_results(_manifest(experiments), tmp_path / "results", workspace_root=workspace)
    assert result["status"] == "complete_with_unavailable_experiments"
    assert result["experiments_unavailable"] == 2
    missing, blocked = result["experiments"]
    assert missing["availability"] == "unavailable"
    assert missing["reason"] == "run_directory_missing"
    assert missing["metrics"] is None
    assert blocked["availability"] == "unavailable"
    assert blocked["reason"] == "GPU runtime did not finish"
    assert blocked["metrics"] is None


def test_optional_oversized_raw_output_is_reported_unavailable_but_summary_survives(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run = _write_mock_run(workspace)
    experiment = _complete_experiment()
    raw = experiment["artifacts"][-1]
    raw["max_bytes"] = 5
    result = collect_results(_manifest([experiment]), tmp_path / "results", workspace_root=workspace)
    report = result["experiments"][0]
    assert report["availability"] == "available"
    raw_report = report["artifacts"][-1]
    assert raw_report["availability"] == "unavailable"
    assert raw_report["reason"] == "source_artifact_exceeds_size_limit"
    assert not (tmp_path / "results" / "complete-cell" / "games.jsonl").exists()
    assert (tmp_path / "results" / "complete-cell" / "spec.json").is_file()
    assert run.is_dir()


@pytest.mark.parametrize(
    "source",
    [
        "test_summary.json",
        "locked-test/games.jsonl",
        "checkpoints/final/summary.json",
        "checkpoint-100/metrics.jsonl",
        "adapter_model.safetensors",
    ],
)
def test_manifest_rejects_locked_test_checkpoint_and_binary_paths(source: str):
    experiment = _complete_experiment()
    experiment["artifacts"] = [
        {
            "source": source,
            "destination": "evidence.jsonl" if source.endswith("jsonl") else "evidence.json",
            "role": "raw_jsonl" if source.endswith("jsonl") else "eval_summary",
            "scope": "dev",
            "required": True,
        }
    ]
    with pytest.raises(CollectionViolation, match="locked-test|checkpoint|binary"):
        validate_collection_manifest(_manifest([experiment]))


def test_locked_test_payload_fails_closed_before_any_results_are_written(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run = workspace / "artifacts" / "runs" / "unsafe-run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"split":"test","win_rate":1.0}\n', encoding="utf-8")
    experiment = {
        "experiment_id": "unsafe",
        "run_dir": "artifacts/runs/unsafe-run",
        "status": "complete",
        "artifacts": [_artifact("summary.json", "eval_summary", "dev")],
    }
    output = tmp_path / "results"
    with pytest.raises(CollectionViolation, match="locked-test split"):
        collect_results(_manifest([experiment]), output, workspace_root=workspace)
    assert not output.exists()


def test_missing_required_artifact_makes_experiment_unavailable(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_mock_run(workspace)
    experiment = _complete_experiment()
    experiment["artifacts"].append(_artifact("missing-summary.json", "eval_summary", "dev"))
    result = collect_results(_manifest([experiment]), tmp_path / "results", workspace_root=workspace)
    report = result["experiments"][0]
    assert report["availability"] == "unavailable"
    assert report["reason"] == "required_artifacts_unavailable"
    assert report["required_artifacts_unavailable"] == ["missing-summary.json"]
    assert report["metrics"] is None


def test_dry_run_hashes_inputs_but_writes_nothing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_mock_run(workspace)
    output = tmp_path / "results"
    result = collect_results(
        _manifest([_complete_experiment()]),
        output,
        workspace_root=workspace,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["files_available"] == 7
    assert not output.exists()


def test_template_and_validate_cli_do_not_read_runs(tmp_path: Path, capsys):
    assert validate_collection_manifest(template_manifest())["experiments"] == []
    declaration = _manifest(
        [
            {
                "experiment_id": "future-run",
                "run_dir": "artifacts/runs/future-run",
                "status": "expected",
                "artifacts": [_artifact("spec.json", "spec", "metadata")],
            }
        ]
    )
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    assert main(["validate", "--manifest", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "valid"
    assert output["runs_read"] is False
    assert output["locked_test_access"] is False


def test_manifest_validation_rejects_scope_drift_and_duplicate_destinations():
    experiment = _complete_experiment()
    experiment["artifacts"][0]["scope"] = "test"
    with pytest.raises(CollectionViolation, match="scope must"):
        validate_collection_manifest(_manifest([experiment]))

    experiment = _complete_experiment()
    duplicate = copy.deepcopy(experiment["artifacts"][0])
    duplicate["source"] = "other-spec.json"
    experiment["artifacts"].append(duplicate)
    with pytest.raises(CollectionViolation, match="duplicate artifact destination"):
        validate_collection_manifest(_manifest([experiment]))
