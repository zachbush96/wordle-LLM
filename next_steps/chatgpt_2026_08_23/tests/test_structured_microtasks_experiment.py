from __future__ import annotations

import json
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.structured_microtasks_experiment import (
    DEFAULT_CONFIG,
    TASKS,
    _record_from_dict,
    audit_bundle,
    audit_evaluation_source,
    audit_source,
    build_bundle,
    dry_run,
    evaluate_gates,
    evaluate_raw_outputs,
    load_config,
    parse_generated_output,
    training_spec,
)
from wordle_lab.common import canonical_json, read_json, read_jsonl


@pytest.fixture(scope="module")
def built_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("structured-microtasks") / "bundle"
    config = {
        **load_config(DEFAULT_CONFIG),
        "bundle_directory": str(directory),
    }
    built, manifest = build_bundle(config, directory)
    return config, built, manifest


def test_source_is_exact_balanced_002_and_locked_test_stays_closed():
    config = load_config(DEFAULT_CONFIG)
    audit = audit_source(config)
    assert audit["status"] == "passed"
    assert audit["dataset_id"] == "COMMON-WORD-CURRICULUM-002"
    assert audit["training_rows"] == 512
    assert audit["train_secret_count"] == 96
    assert audit["dev_secret_count"] == 32
    assert audit["train_dev_overlap"] == 0
    assert audit["locked_test_access"] is False
    evaluation = audit_evaluation_source(config)
    assert evaluation["status"] == "passed"
    assert evaluation["allowed_words_sha256"] == config["evaluation"]["allowed_words_sha256"]
    assert evaluation["locked_test_access"] is False


def test_bundle_has_preregistered_task_mix_candidate_balance_and_no_overlap(built_bundle):
    config, directory, manifest = built_bundle
    audit = audit_bundle(config, directory)
    assert audit["status"] == "passed"
    assert audit["locked_test_access"] is False
    assert audit["train_dev_record_overlap"] == 0
    assert audit["train_dev_history_overlap"] == 0
    assert audit["splits"]["train"]["task_distribution"] == {
        "candidate_validity": 288,
        "constraint_merge": 192,
        "feedback_decode": 256,
        "full_policy": 400,
        "singleton_solve": 80,
    }
    assert audit["splits"]["dev"]["task_distribution"] == {
        "candidate_validity": 96,
        "constraint_merge": 128,
        "feedback_decode": 128,
        "full_policy": 128,
        "singleton_solve": 128,
    }
    assert manifest["splits"]["train"]["candidate_balance"]["valid"] == 144
    assert manifest["splits"]["train"]["candidate_balance"]["invalid"] == 144
    assert set(manifest["splits"]["train"]["candidate_balance"]["invalid_reasons"].values()) == {24}
    assert set(manifest["splits"]["dev"]["candidate_balance"]["invalid_reasons"].values()) == {8}


def test_duplicate_heavy_examples_are_preserved_in_every_state_task(built_bundle):
    config, _, manifest = built_bundle
    minimum = config["selection"]["duplicate_heavy_min_fraction"]
    for split in ("train", "dev"):
        distribution = manifest["splits"][split]["task_distribution"]
        duplicate = manifest["splits"][split]["duplicate_heavy_distribution"]
        for task in ("feedback_decode", "constraint_merge", "singleton_solve", "full_policy"):
            assert duplicate[task] / distribution[task] >= minimum


def test_rendered_auxiliary_rows_are_machine_readable_and_policy_rows_are_natural(built_bundle):
    _, directory, _ = built_bundle
    rows = read_jsonl(directory / "train" / "mixed.jsonl")
    for row in rows:
        prompt_payload = canonical_json(row["prompt"])
        assert "secret_answer" not in prompt_payload
        assert "posterior_candidates" not in prompt_payload
        assert row["messages"] == row["prompt"] + row["completion"]
        if row["task_type"] == "full_policy":
            assert row["output_format"] == "natural_terminal_answer"
            assert row["completion"][0]["content"].startswith("Final answer: ")
        else:
            assert row["output_format"] == "strict_json_object"
            assert json.loads(row["completion"][0]["content"]) == row["target"]
        if row["task_type"] == "constraint_merge":
            user_request = json.loads(row["prompt"][-1]["content"])
            assert set(user_request["input"]) == {"history"}


def test_build_is_deterministic_across_output_directories(tmp_path: Path):
    config = load_config(DEFAULT_CONFIG)
    _, first = build_bundle(config, tmp_path / "one")
    _, second = build_bundle(config, tmp_path / "two")
    for split in ("train", "dev"):
        assert first["splits"][split]["files"] == second["splits"][split]["files"]
        assert first["splits"][split]["primitive_audit"]["hashes"] == second["splits"][split]["primitive_audit"]["hashes"]


def test_perfect_natural_outputs_score_every_task_and_reason(built_bundle):
    config, directory, _ = built_bundle
    records = [_record_from_dict(row) for row in read_jsonl(directory / "dev" / "records.jsonl")]
    raw_outputs = {
        record.record_id: (
            f"Final answer: {record.target['word']}"
            if record.task_type == "full_policy"
            else canonical_json(record.target)
        )
        for record in records
    }
    allowed = read_json(Path(config["source"]["directory"]) / "universe.json")
    parsed, metrics = evaluate_raw_outputs(records, raw_outputs, allowed)
    assert len(parsed) == len(records)
    assert metrics["coverage"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert all(item["accuracy"] == 1.0 for item in metrics["by_task"].values())
    assert all(
        item["accuracy"] == 1.0
        for item in metrics["candidate_invalid_reason_accuracy"].values()
    )
    assert evaluate_gates(metrics, config["evaluation"]["gates"])["passed"] is True


def test_parsers_are_strict_and_do_not_repair_outputs(built_bundle):
    config, directory, _ = built_bundle
    records = [_record_from_dict(row) for row in read_jsonl(directory / "dev" / "records.jsonl")]
    auxiliary = next(record for record in records if record.task_type == "candidate_validity")
    parsed, status = parse_generated_output(auxiliary, '```json\n{"valid":true,"reason":null}\n```', [])
    assert parsed is None
    assert status["parse_status"] == "invalid_json"
    parsed, status = parse_generated_output(
        auxiliary,
        canonical_json({**auxiliary.target, "extra": "not allowed"}),
        [],
    )
    assert parsed is None
    assert status["parse_status"] == "candidate_schema_error"
    parsed, status = parse_generated_output(
        auxiliary,
        '{"valid":false,"reason":"green","violations":[{}]}',
        [],
    )
    assert parsed is None
    assert status["parse_status"] == "candidate_schema_error"
    constraint = next(record for record in records if record.task_type == "feedback_decode")
    parsed, status = parse_generated_output(constraint, '{"arbitrary":"object"}', [])
    assert parsed is None
    assert status["parse_status"] == "constraint_schema_error"
    singleton = next(record for record in records if record.task_type == "singleton_solve")
    parsed, status = parse_generated_output(
        singleton,
        canonical_json({"word": singleton.target["word"], "extra": 1}),
        [],
    )
    assert parsed is None
    assert status["parse_status"] == "word_schema_error"
    policy = next(record for record in records if record.task_type == "full_policy")
    allowed = read_json(Path(config["source"]["directory"]) / "universe.json")
    parsed, status = parse_generated_output(
        policy,
        f"Final answer: {policy.target['word']}\nextra prose",
        allowed,
    )
    assert parsed is None
    assert status["format_valid"] is False


def test_completion_sft_specs_are_audited_for_both_backends(built_bundle):
    config, _, manifest = built_bundle
    ordinary = training_spec(config, "transformers", manifest)
    unsloth = training_spec(config, "unsloth", manifest)
    assert ordinary["word_token_weight"] == unsloth["word_token_weight"] == 1.0
    assert ordinary["data"] == unsloth["data"]
    assert ordinary["backend"] != unsloth["backend"]
    for spec in (ordinary, unsloth):
        assert spec["locked_test_access"] is False
        assert spec["candidate_injection"] is False
        assert spec["output_repair"] is False


def test_dry_run_never_starts_training(built_bundle):
    config, directory, _ = built_bundle
    result = dry_run(config, "unsloth")
    assert result["status"] == "dry_run_passed"
    assert result["training_started"] is False
    assert result["locked_test_access"] is False
    assert result["spec"]["data"]["development_file"] == "dev/mixed.jsonl"
