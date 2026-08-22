from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from peft import LoraConfig, TaskType, get_peft_model


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_LORA_ADAPTER = {
    "type": "lora",
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": list(DEFAULT_LORA_TARGET_MODULES),
    "use_rslora": False,
    "use_dora": False,
}


@dataclass(frozen=True)
class TechniqueMetadata:
    technique_id: str
    objective_family: str
    training_signal: str
    warm_start_required: bool
    reference_policy_required: bool
    trainer_entrypoint: str


TECHNIQUE_REGISTRY: dict[str, TechniqueMetadata] = {
    "sft": TechniqueMetadata(
        technique_id="sft",
        objective_family="supervised_likelihood",
        training_signal="oracle_demonstrations",
        warm_start_required=False,
        reference_policy_required=False,
        trainer_entrypoint="wordle_lab.methods.sft.train_sft",
    ),
    "dpo": TechniqueMetadata(
        technique_id="dpo",
        objective_family="offline_preference",
        training_signal="chosen_rejected_pairs",
        warm_start_required=True,
        reference_policy_required=True,
        trainer_entrypoint="wordle_lab.methods.dpo.train_dpo",
    ),
    "orpo": TechniqueMetadata(
        technique_id="orpo",
        objective_family="monolithic_preference",
        training_signal="chosen_rejected_pairs_plus_chosen_likelihood",
        warm_start_required=True,
        reference_policy_required=False,
        trainer_entrypoint="wordle_lab.methods.orpo.train_orpo",
    ),
    "grpo": TechniqueMetadata(
        technique_id="grpo",
        objective_family="on_policy_group_relative",
        training_signal="verifiable_environment_rewards",
        warm_start_required=True,
        reference_policy_required=False,
        trainer_entrypoint="wordle_lab.methods.grpo.train_grpo",
    ),
    "q_sft": TechniqueMetadata(
        technique_id="q_sft",
        objective_family="offline_value_learning",
        training_signal="bellman_likelihood_targets",
        warm_start_required=True,
        reference_policy_required=False,
        trainer_entrypoint="wordle_lab.methods.q_sft.train_q_sft",
    ),
}


def technique_metadata(technique_id: str) -> dict[str, Any]:
    """Return stable, serializable metadata for a registered objective."""
    try:
        return asdict(TECHNIQUE_REGISTRY[technique_id.lower()])
    except KeyError as exc:
        supported = ", ".join(sorted(TECHNIQUE_REGISTRY))
        raise ValueError(f"unsupported technique {technique_id!r}; expected one of: {supported}") from exc


def _positive_values(value: Any, field: str) -> None:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0 for item in values):
        raise ValueError(f"{field} must contain positive numeric values")


def validate_technique_config(technique_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate objective-specific configuration without requiring a runtime parent path."""
    metadata = technique_metadata(technique_id)
    normalized = dict(config)
    configured_method = str(normalized.get("method", technique_id)).lower()
    if configured_method != metadata["technique_id"]:
        raise ValueError(
            f"technique/config mismatch: requested {metadata['technique_id']!r}, config declares {configured_method!r}"
        )
    normalized["method"] = configured_method
    if "learning_rate" in normalized:
        _positive_values(normalized["learning_rate"], "learning_rate")
    if "max_steps" in normalized:
        _positive_values(normalized["max_steps"], "max_steps")
    if configured_method == "dpo":
        if "beta" not in normalized:
            raise ValueError("dpo config requires beta")
        _positive_values(normalized["beta"], "beta")
    if configured_method == "orpo":
        if "lambda_or" not in normalized:
            raise ValueError("orpo config requires lambda_or")
        _positive_values(normalized["lambda_or"], "lambda_or")
    if configured_method == "grpo":
        if "group_size" not in normalized:
            raise ValueError("grpo config requires group_size")
        _positive_values(normalized["group_size"], "group_size")
    if configured_method == "q_sft":
        discount = normalized.get("discount")
        if isinstance(discount, bool) or not isinstance(discount, (int, float)) or not 0 <= float(discount) <= 1:
            raise ValueError("q_sft config requires discount in [0, 1]")
    if "warm_start_required" in normalized and bool(normalized["warm_start_required"]) != metadata["warm_start_required"]:
        raise ValueError(f"{configured_method} warm_start_required conflicts with the technique registry")
    return normalized


def _adapter_mapping(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if "adapter" in config:
        value = config["adapter"]
        if not isinstance(value, Mapping):
            raise ValueError("adapter must be a mapping")
        return dict(value)
    if "lora" in config:
        value = config["lora"]
        if not isinstance(value, Mapping):
            raise ValueError("lora must be a mapping")
        return {"type": "lora", **dict(value)}
    return dict(config)


def normalize_adapter_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize the legacy ``lora`` block or a new ``adapter`` block."""
    supplied = _adapter_mapping(config)
    aliases = {
        "adapter_type": "type",
        "peft_type": "type",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
    }
    for old, new in aliases.items():
        if old in supplied:
            if new in supplied:
                raise ValueError(f"adapter config supplies both {old} and {new}")
            supplied[new] = supplied.pop(old)
    allowed = set(DEFAULT_LORA_ADAPTER)
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"unsupported adapter fields: {unknown}")
    normalized = {**DEFAULT_LORA_ADAPTER, **supplied}
    normalized["type"] = str(normalized["type"]).lower()
    if normalized["type"] != "lora":
        raise ValueError("only the registered lora adapter is currently supported")
    for field in ("r", "alpha"):
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 or int(value) != value:
            raise ValueError(f"adapter {field} must be a positive integer")
        normalized[field] = int(value)
    dropout = normalized["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("adapter dropout must be in [0, 1)")
    normalized["dropout"] = float(dropout)
    if normalized["bias"] not in {"none", "all", "lora_only"}:
        raise ValueError("adapter bias must be one of: none, all, lora_only")
    targets = normalized["target_modules"]
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError("adapter target_modules must be a sequence of names")
    targets = list(targets)
    if not targets or any(not isinstance(name, str) or not name.strip() for name in targets):
        raise ValueError("adapter target_modules must contain non-empty strings")
    if len(set(targets)) != len(targets):
        raise ValueError("adapter target_modules must be unique")
    normalized["target_modules"] = targets
    for field in ("use_rslora", "use_dora"):
        if not isinstance(normalized[field], bool):
            raise ValueError(f"adapter {field} must be boolean")
    return normalized


def build_adapter_config(config: Mapping[str, Any] | None = None) -> LoraConfig:
    """Build the registered PEFT configuration while preserving historical defaults."""
    normalized = normalize_adapter_config(config)
    return LoraConfig(
        r=normalized["r"],
        lora_alpha=normalized["alpha"],
        lora_dropout=normalized["dropout"],
        bias=normalized["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=normalized["target_modules"],
        use_rslora=normalized["use_rslora"],
        use_dora=normalized["use_dora"],
    )


def validate_trainable_targets(model, expected_target_modules: Sequence[str]) -> dict[str, Any]:
    """Fail when a declared LoRA target did not produce trainable adapter parameters."""
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter attachment produced no trainable parameters")
    matched = {
        target: [name for name in trainable if f".{target}." in f".{name}."]
        for target in expected_target_modules
    }
    missing = [target for target, names in matched.items() if not names]
    if missing:
        raise RuntimeError(f"adapter targets have no trainable parameters: {missing}")
    return {
        "trainable_parameter_tensors": len(trainable),
        "validated_target_modules": list(expected_target_modules),
        "trainable_tensors_by_target": {target: len(names) for target, names in matched.items()},
    }


def attach_adapter(model, config: Mapping[str, Any] | None = None):
    """Attach a validated registered adapter and return model/config/validation metadata."""
    normalized = normalize_adapter_config(config)
    adapted = get_peft_model(model, build_adapter_config(normalized))
    validation = validate_trainable_targets(adapted, normalized["target_modules"])
    return adapted, normalized, validation
