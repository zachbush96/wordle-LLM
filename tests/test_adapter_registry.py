from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wordle_lab.methods.adapters import (
    DEFAULT_LORA_ADAPTER,
    build_adapter_config,
    normalize_adapter_config,
    technique_metadata,
    validate_technique_config,
    validate_trainable_targets,
)


def test_legacy_lora_block_preserves_historical_defaults():
    normalized = normalize_adapter_config(
        {
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
        }
    )
    assert normalized == DEFAULT_LORA_ADAPTER
    peft_config = build_adapter_config(normalized)
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.05
    assert peft_config.bias == "none"
    assert peft_config.use_rslora is False
    assert peft_config.use_dora is False


def test_normalized_adapter_supports_registered_lora_variants():
    normalized = normalize_adapter_config(
        {
            "adapter": {
                "type": "lora",
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": True,
                "use_dora": True,
            }
        }
    )
    assert normalized["r"] == 8
    assert normalized["use_rslora"] is True
    assert normalized["use_dora"] is True


@pytest.mark.parametrize(
    "config, message",
    [
        ({"adapter": {"type": "ia3"}}, "only the registered lora"),
        ({"adapter": {"r": 0}}, "positive integer"),
        ({"adapter": {"dropout": 1.0}}, "in [0, 1)"),
        ({"adapter": {"target_modules": ["q_proj", "q_proj"]}}, "must be unique"),
        ({"adapter": {"surprise": True}}, "unsupported adapter fields"),
    ],
)
def test_adapter_config_rejects_invalid_values(config, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace(")", r"\)")):
        normalize_adapter_config(config)


def test_technique_registry_describes_reference_and_warm_start_semantics():
    assert technique_metadata("sft")["warm_start_required"] is False
    assert technique_metadata("dpo")["reference_policy_required"] is True
    assert technique_metadata("orpo")["reference_policy_required"] is False
    assert technique_metadata("grpo")["objective_family"] == "on_policy_group_relative"
    assert technique_metadata("q_sft")["training_signal"] == "bellman_likelihood_targets"
    with pytest.raises(ValueError, match="unsupported technique"):
        technique_metadata("unknown")


def test_objective_config_validation_handles_grids_and_required_parameters():
    validated = validate_technique_config(
        "dpo", {"method": "dpo", "beta": [0.05, 0.1], "learning_rate": [5e-6, 1e-5], "warm_start_required": True}
    )
    assert validated["beta"] == [0.05, 0.1]
    with pytest.raises(ValueError, match="requires lambda_or"):
        validate_technique_config("orpo", {"method": "orpo"})
    with pytest.raises(ValueError, match="warm_start_required conflicts"):
        validate_technique_config("sft", {"method": "sft", "warm_start_required": True})


@pytest.mark.parametrize("technique_id", ["sft", "dpo", "orpo", "grpo", "q_sft"])
def test_repository_objective_yaml_matches_registry(technique_id):
    path = Path(__file__).parents[1] / "configs" / "methods" / f"{technique_id}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate_technique_config(technique_id, config)["method"] == technique_id


def test_trainable_target_validation_requires_every_declared_module():
    parameters = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": SimpleNamespace(requires_grad=True),
        "base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight": SimpleNamespace(requires_grad=True),
        "base_model.model.layers.0.mlp.down_proj.weight": SimpleNamespace(requires_grad=False),
    }
    model = SimpleNamespace(named_parameters=lambda: parameters.items())
    summary = validate_trainable_targets(model, ["q_proj", "v_proj"])
    assert summary["trainable_tensors_by_target"] == {"q_proj": 1, "v_proj": 1}
    with pytest.raises(RuntimeError, match="adapter targets have no trainable parameters"):
        validate_trainable_targets(model, ["q_proj", "down_proj"])
