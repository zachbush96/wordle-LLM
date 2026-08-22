"""Validated, dev-only orchestration for an SFT -> stable-GRPO study."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from wordle_lab.methods.grpo_stability import (
    validate_reward_rubric,
    validate_virtual_support_spec,
)
from wordle_lab.protocol.lock import PROTOCOL_ID


FORBIDDEN_BEHAVIOR_FIELDS = {
    "hardcoded_opening_guess",
    "opening_guess",
    "output_repair",
    "candidate_forcing",
    "forced_candidate",
    "test_answers",
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def validate_hybrid_spec(spec: Mapping[str, object]) -> dict:
    """Validate the frozen-protocol study contract without accessing data."""

    if not isinstance(spec, Mapping):
        raise ValueError("hybrid spec must be a mapping")
    forbidden = FORBIDDEN_BEHAVIOR_FIELDS.intersection(spec)
    if forbidden:
        raise ValueError(f"forbidden behavior fields: {sorted(forbidden)}")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must remain frozen at {PROTOCOL_ID}")
    if spec.get("test_access") != "forbidden":
        raise ValueError("hybrid study must forbid locked-test access")
    if spec.get("pipeline") != ["sft", "grpo"]:
        raise ValueError("pipeline must be exactly ['sft', 'grpo']")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    stages = spec.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {"sft", "grpo"}:
        raise ValueError("stages must contain exactly sft and grpo")
    sft = dict(stages["sft"])
    grpo = dict(stages["grpo"])
    if sft.get("method") != "sft" or sft.get("parent_checkpoint", "base") != "base":
        raise ValueError("SFT stage must train from the base checkpoint")
    if grpo.get("method") != "grpo" or grpo.get("parent_from") != "sft.final_checkpoint":
        raise ValueError("GRPO must warm-start from sft.final_checkpoint")
    for stage_name, stage in (("sft", sft), ("grpo", grpo)):
        _positive_int(stage.get("max_steps"), f"{stage_name}.max_steps")
        _positive_int(stage.get("batch_size"), f"{stage_name}.batch_size")
        _positive_int(stage.get("gradient_accumulation_steps"), f"{stage_name}.gradient_accumulation_steps")
        _positive_float(stage.get("learning_rate"), f"{stage_name}.learning_rate")
    _positive_int(grpo.get("group_size"), "grpo.group_size")
    if int(grpo["group_size"]) < 2:
        raise ValueError("grpo.group_size must be at least two")
    rubric = validate_reward_rubric(grpo.get("reward_rubric", {}))
    virtual = validate_virtual_support_spec(grpo.get("virtual_support"))

    promotion = spec.get("promotion_gate")
    if not isinstance(promotion, Mapping) or promotion.get("split") != "dev":
        raise ValueError("promotion_gate must use the dev split")
    if promotion.get("require_checkpoint", True) is not True:
        raise ValueError("promotion gate must require a checkpoint")
    thresholds = promotion.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("promotion_gate.thresholds must be a non-empty mapping")
    normalized_thresholds = {}
    for metric, rule in thresholds.items():
        if not isinstance(rule, Mapping) or set(rule) != {"op", "value"} or rule["op"] not in {"<=", ">="}:
            raise ValueError(f"invalid promotion rule for {metric}")
        normalized_thresholds[str(metric)] = {"op": rule["op"], "value": float(rule["value"])}

    stability = grpo.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError("grpo.stability is required")
    maximum_rate = float(stability.get("maximum_advantage_collapse_rate", -1))
    if not 0 <= maximum_rate <= 1:
        raise ValueError("maximum_advantage_collapse_rate must be between zero and one")
    entropy_guard = stability.get("entropy_guard")
    if not isinstance(entropy_guard, Mapping) or int(entropy_guard.get("patience", 0)) < 1:
        raise ValueError("an entropy guard with positive patience is required")

    return {
        **dict(spec),
        "seed": seed,
        "stages": {"sft": sft, "grpo": {**grpo, "reward_rubric": rubric, "virtual_support": virtual}},
        "promotion_gate": {**dict(promotion), "thresholds": normalized_thresholds},
    }


def promotion_decision(metrics: Mapping[str, object], gate: Mapping[str, object], checkpoint: str | Path | None) -> dict:
    """Return an auditable dev-gate decision for the SFT checkpoint."""

    failures = []
    if not checkpoint:
        failures.append("missing_checkpoint")
    if metrics.get("split") != "dev":
        failures.append("evaluation_split_is_not_dev")
    for metric, rule in gate["thresholds"].items():
        if metric not in metrics:
            failures.append(f"missing_metric:{metric}")
            continue
        observed = float(metrics[metric])
        passed = observed <= rule["value"] if rule["op"] == "<=" else observed >= rule["value"]
        if not passed:
            failures.append(f"threshold_failed:{metric}")
    return {"promote": not failures, "failures": failures, "split": metrics.get("split")}


def build_hybrid_plan(spec: Mapping[str, object]) -> list[dict]:
    validated = validate_hybrid_spec(spec)
    return [
        {"stage": "sft", "action": "train", "spec": validated["stages"]["sft"]},
        {"stage": "sft_dev_gate", "action": "evaluate", "split": "dev", "gate": validated["promotion_gate"]},
        {
            "stage": "grpo",
            "action": "train_if_promoted",
            "parent_from": "sft.final_checkpoint",
            "spec": validated["stages"]["grpo"],
        },
    ]


def run_hybrid_pipeline(
    spec: Mapping[str, object],
    *,
    train_sft_stage: Callable[[Mapping[str, object]], Mapping[str, object]],
    evaluate_dev: Callable[..., Mapping[str, object]],
    train_grpo_stage: Callable[..., Mapping[str, object]],
) -> dict:
    """Run callbacks in order, stopping before GRPO unless the dev gate passes.

    This orchestrator never exposes a test split. Callback injection keeps the
    policy separate from expensive training and makes dry runs straightforward.
    """

    validated = validate_hybrid_spec(spec)
    sft_result = dict(train_sft_stage(validated["stages"]["sft"]))
    checkpoint = sft_result.get("final_checkpoint")
    dev_metrics = dict(evaluate_dev(checkpoint=checkpoint, split="dev"))
    decision = promotion_decision(dev_metrics, validated["promotion_gate"], checkpoint)
    result = {"status": "stopped_at_sft_dev_gate", "sft": sft_result, "sft_dev": dev_metrics, "promotion": decision}
    if not decision["promote"]:
        return result
    grpo_result = dict(
        train_grpo_stage(
            validated["stages"]["grpo"],
            parent_checkpoint=checkpoint,
            protocol_id=validated["protocol_id"],
        )
    )
    result.update({"status": "grpo_completed", "grpo": grpo_result})
    return result
