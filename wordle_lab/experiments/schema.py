from __future__ import annotations

ALLOWED_STATES = ("PLANNED", "DATA_READY", "TRAINING", "TRAINED", "DEV_EVALUATED", "SELECTED", "TEST_EVALUATED", "REPORTED")


def validate_spec(spec: dict) -> dict:
    required = {"method", "representation", "seed", "max_steps", "learning_rate", "batch_size", "gradient_accumulation_steps", "max_length", "lora"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"missing spec fields: {missing}")
    if spec["method"] not in {"sft", "continued_sft", "dpo", "orpo", "grpo", "q_sft", "sft_grpo"}:
        raise ValueError("unsupported method")
    if spec["representation"] not in {"state_direct", "episode_multiturn", "state_rationale", "mixed_curriculum"}:
        raise ValueError("unsupported representation")
    if int(spec["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive")
    if float(spec["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if int(spec["batch_size"]) <= 0 or int(spec["gradient_accumulation_steps"]) <= 0:
        raise ValueError("batch sizes must be positive")
    return spec
