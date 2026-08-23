from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23 import full_finetune as core
from next_steps.chatgpt_2026_08_23 import full_finetune_experiment as experiment
from wordle_lab.common import canonical_json, sha256_text, write_json


@pytest.fixture(scope="module")
def primary_spec() -> dict:
    return experiment.matched_full_spec(experiment.DEFAULT_DATA)


def test_primary_spec_is_exact_native_transformers_lora_match(primary_spec: dict) -> None:
    metadata = core.validate_full_finetune_spec(primary_spec)
    assert metadata["model_id"] == "google/gemma-3-270m-it"
    assert metadata["revision"] == "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
    assert primary_spec["experiment_id"] == core.FULL_FINETUNE_EXPERIMENT_ID
    assert primary_spec["experiment_mode"] == core.PRIMARY_MODE
    assert primary_spec["matched_comparison"] is True
    assert primary_spec["max_steps"] == 600
    assert primary_spec["seed"] == 2026
    assert primary_spec["learning_rate"] == 5e-5
    assert primary_spec["batch_size"] == 4
    assert primary_spec["gradient_accumulation_steps"] == 1
    assert primary_spec["effective_batch_size"] == 4
    assert primary_spec["max_length"] == 320
    assert primary_spec["word_token_weight"] == 8.0
    assert primary_spec["optimizer"] == "torch.optim.AdamW"
    assert primary_spec["scheduler"] == "linear_warmup_5pct_cosine"
    assert primary_spec["checkpoint_steps"] == [150, 300, 450, 600]
    assert primary_spec["checkpoint_fractions"] == [0.25, 0.5, 0.75, 1.0]
    assert primary_spec["protocol_id"] == "WORDLE-PROTOCOL-002"
    assert primary_spec["protocol_sha256"] == "afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "other"),
        ("experiment_mode", core.SMOKE_MODE),
        ("seed", 99),
        ("max_steps", 7),
        ("learning_rate", 1e-3),
        ("batch_size", 1),
        ("gradient_accumulation_steps", 4),
        ("effective_batch_size", 1),
        ("max_length", 128),
        ("word_token_weight", 1.0),
        ("optimizer", "adamw"),
        ("scheduler", "constant"),
        ("checkpoint_steps", [600]),
        ("checkpoint_fractions", [1.0]),
        ("gradient_checkpointing", True),
        ("precision", "float16"),
        ("locked_test_access", True),
        ("candidate_injection", True),
        ("vocabulary_masking", True),
        ("reranking", True),
        ("repeat_ban", True),
        ("output_repair", True),
        ("harness_selected_guess", True),
    ],
)
def test_primary_contract_fails_closed_on_recipe_or_protocol_drift(
    primary_spec: dict, field: str, value: object
) -> None:
    changed = deepcopy(primary_spec)
    changed[field] = value
    with pytest.raises(ValueError, match="full fine-tune spec drift"):
        core.validate_full_finetune_spec(changed)


def test_primary_contract_rejects_nested_data_model_protocol_and_comparator_drift(primary_spec: dict) -> None:
    mutations = [
        ("data", "rendered_sha256", "0" * 64),
        ("model", "revision", "0" * 40),
        ("protocol", "protocol_sha256", "0" * 64),
        ("comparator", "accounting_sha256", "0" * 64),
        ("comparison_contract", "matched", False),
        ("evaluation", "dev_games", 31),
    ]
    for section, key, value in mutations:
        changed = deepcopy(primary_spec)
        changed[section][key] = value
        with pytest.raises(ValueError, match="full fine-tune spec drift"):
            core.validate_full_finetune_spec(changed)


def test_smoke_is_fixed_two_step_actual_benchmark_and_never_matched() -> None:
    spec = experiment.smoke_full_spec()
    core.validate_full_finetune_spec(spec)
    assert spec["experiment_id"] == core.FULL_FINETUNE_SMOKE_ID
    assert spec["experiment_mode"] == core.SMOKE_MODE
    assert spec["max_steps"] == 2
    assert spec["batch_size"] == spec["effective_batch_size"] == 1
    assert spec["checkpoint_steps"] == spec["checkpoint_fractions"] == []
    assert spec["matched_comparison"] is False
    assert spec["comparison_contract"]["matched"] is False
    changed = deepcopy(spec)
    changed["matched_comparison"] = True
    with pytest.raises(ValueError, match="full fine-tune spec drift"):
        core.validate_full_finetune_spec(changed)


def test_native_lora_comparator_is_pinned_byte_for_byte() -> None:
    audit = experiment.audit_lora_comparator()
    assert audit["run_id"] == "sft-common-balanced-word-s2026-0649b4deeb"
    assert audit["spec_sha256"] == core.EXPECTED_COMPARATOR_EVIDENCE["spec_sha256"]
    assert audit["dataset_manifest_sha256"] == core.EXPECTED_COMPARATOR_EVIDENCE["dataset_manifest_sha256"]
    assert audit["accounting_sha256"] == core.EXPECTED_COMPARATOR_EVIDENCE["accounting_sha256"]
    assert audit["final_adapter_tree_sha256"] == core.EXPECTED_COMPARATOR_EVIDENCE["final_adapter_tree_sha256"]
    assert audit["checkpoint_adapter_tree_sha256"]["step-000600"] == audit["final_adapter_tree_sha256"]
    assert set(audit["final_adapter_files"]) == {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert audit["final_adapter_files"]["adapter_model.safetensors"]["sha256"] == (
        "6404d0d4c06f2c2cc2dcd871fc7a6dd7b05201057d11d6be613e3188c76a2817"
    )


def test_balanced_data_and_protocol_are_exact_and_locked_test_free() -> None:
    audit = experiment.audit_balanced_source(experiment.DEFAULT_DATA)
    protocol = experiment.audit_protocol()
    assert audit["rows"] == 512
    assert audit["files"] == experiment.EXPECTED_DATA_HASHES
    assert audit["train_secret_count"] == 96
    assert audit["dev_secret_count"] == 32
    assert audit["locked_test_access"] is False
    assert protocol["protocol_id"] == "WORDLE-PROTOCOL-002"
    assert protocol["locked_test_access"] is False


def test_vram_preflight_is_read_only_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(core.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(core.torch.cuda, "get_device_name", lambda _: "test-gpu")
    monkeypatch.setattr(core.torch.cuda, "mem_get_info", lambda: (2_000, 4_000))
    ready = core.full_finetune_vram_preflight(parameter_count=100, margin_bytes=100)
    assert ready == {
        "status": "ready_for_full_allocation",
        "ready": True,
        "read_only": True,
        "model_loaded": False,
        "parameter_count": 100,
        "estimated_parameter_state_bytes": 1_200,
        "activation_allocator_margin_bytes": 100,
        "required_free_vram_bytes": 1_300,
        "free_vram_bytes": 2_000,
        "total_vram_bytes": 4_000,
        "gpu": "test-gpu",
        "bf16_supported": True,
    }
    monkeypatch.setattr(core.torch.cuda, "mem_get_info", lambda: (1_000, 4_000))
    blocked = core.full_finetune_vram_preflight(parameter_count=100, margin_bytes=100)
    assert blocked["ready"] is False
    assert blocked["status"] == "blocked_insufficient_free_vram"


def test_vram_preflight_blocks_without_cuda_or_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core.torch.cuda, "is_available", lambda: False)
    assert core.full_finetune_vram_preflight(parameter_count=100, margin_bytes=0)["status"] == "blocked_cuda_unavailable"
    monkeypatch.setattr(core.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core.torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(core.torch.cuda, "mem_get_info", lambda: (10_000, 10_000))
    monkeypatch.setattr(core.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(core.torch.cuda, "get_device_name", lambda _: "test-gpu")
    assert core.full_finetune_vram_preflight(parameter_count=100, margin_bytes=0)["status"] == "blocked_bf16_unsupported"


def test_prepare_run_refuses_existing_directory_before_writes(
    primary_spec: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "ARTIFACTS", tmp_path)
    run_dir = experiment.prepare_run(primary_spec)
    assert (run_dir / "spec.json").is_file()
    assert (run_dir / "dataset_manifest.json").is_file()
    assert (run_dir / "comparison_manifest.json").is_file()
    before = (run_dir / "spec.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.prepare_run(primary_spec)
    assert (run_dir / "spec.json").read_bytes() == before


def test_training_blocks_on_read_only_preflight_before_creating_run(
    primary_spec: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "full_finetune_vram_preflight",
        lambda **_: {"ready": False, "status": "blocked_insufficient_free_vram"},
    )
    prepare_called = False

    def unexpected_prepare(*args, **kwargs):
        nonlocal prepare_called
        prepare_called = True
        raise AssertionError("prepare_run must not be called after a blocked preflight")

    monkeypatch.setattr(experiment, "prepare_run", unexpected_prepare)
    with pytest.raises(RuntimeError, match="preflight blocked"):
        experiment.run_training(primary_spec)
    assert prepare_called is False


def test_evaluation_data_hashes_are_bound_to_run_spec(primary_spec: dict) -> None:
    observed = experiment.assert_evaluation_data_binding(primary_spec)
    assert observed["files"] == primary_spec["data"]["files"]
    changed = deepcopy(primary_spec)
    changed["data"]["canonical_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="evaluation data does not match"):
        experiment.assert_evaluation_data_binding(changed)


def test_paired_summary_joins_pinned_lora_and_bound_full_metrics(primary_spec: dict, tmp_path: Path) -> None:
    write_json(tmp_path / "spec.json", primary_spec)
    with pytest.raises(ValueError, match="comparable metrics only"):
        experiment.build_paired_comparison_summary(tmp_path, "step-000150")
    full_summary = {
        "status": "dev_evaluated",
        "spec_sha256": sha256_text(canonical_json(primary_spec)),
        "split": primary_spec["evaluation"]["split"],
        "checkpoint": "step-000600",
        "matched_comparison": True,
        "evaluation_data": experiment._binding_view(primary_spec["data"]),
        "protocol": primary_spec["protocol"],
        "locked_test_access": False,
        "gameplay": {
            "wins": 9,
            "win_rate": 9 / 32,
            "terminal_marker_compliance": 1.0,
            "invalid_guess_rate": 0.0,
            "repeat_guess_rate": 0.1,
        },
        "diagnostics": {
            "posterior_constraint_violation_rate": 0.5,
            "singleton_answer_accuracy": 0.25,
            "action_target_accuracy": 0.2,
            "by_turn": {"2": {"posterior_constraint_violation_rate": 0.4}},
        },
    }
    write_json(tmp_path / "eval-step-000600-summary.json", full_summary)
    result = experiment.build_paired_comparison_summary(tmp_path, "step-000600")
    assert result["status"] == "paired_development_comparison_ready"
    assert result["native_transformers_lora"]["run_id"] == "sft-common-balanced-word-s2026-0649b4deeb"
    assert result["full_parameter"]["metrics"]["wins"] == 9
    assert result["delta_full_minus_lora"]["wins"] == 1
    assert result["single_seed"] is True
    assert result["locked_test_access"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.build_paired_comparison_summary(tmp_path, "step-000600")


def test_memory_estimate_and_checkpoint_doses_are_deterministic() -> None:
    assert core.estimated_adamw_training_bytes(100, parameter_bytes=2) == 1_200
    assert core._checkpoint_steps(600, (0.25, 0.5, 0.75, 1.0)) == [150, 300, 450, 600]
    with pytest.raises(ValueError, match="fractions"):
        core._checkpoint_steps(10, (float("nan"), 1.0))
