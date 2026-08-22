from __future__ import annotations

from dataclasses import asdict, dataclass


class ProtocolCompatibilityError(ValueError):
    """Raised when an experiment would silently change WORDLE-PROTOCOL-002."""


@dataclass(frozen=True)
class Technique:
    name: str
    kind: str
    canonical_compatible: bool
    implementation: str
    notes: str


TECHNIQUES = {
    item.name: item
    for item in (
        Technique("lora", "adapter", True, "wordle_lab.methods.adapters", "PEFT mechanism shared by training objectives."),
        Technique("rslora", "adapter", True, "wordle_lab.methods.adapters", "Rank-stabilized LoRA parameterization."),
        Technique("dora", "adapter", True, "wordle_lab.methods.adapters", "Weight-decomposed LoRA parameterization."),
        Technique("sft", "objective", True, "wordle_lab.methods.sft", "Behavioral cloning/completion loss."),
        Technique("dpo", "objective", True, "wordle_lab.methods.dpo", "Offline reference-regulated preference optimization."),
        Technique("orpo", "objective", True, "wordle_lab.methods.orpo", "Offline reference-free preference optimization."),
        Technique("grpo", "objective", True, "wordle_lab.methods.grpo", "On-policy group-relative optimization."),
        Technique("q_sft", "objective", True, "wordle_lab.methods.q_sft", "Offline Bellman-likelihood weighted SFT."),
        Technique("sft_grpo", "pipeline", True, "wordle_lab.experiments.hybrid_sft_grpo", "SFT warm start followed by GRPO."),
        Technique("acr", "diagnostic", True, "wordle_lab.methods.grpo_stability", "Advantage Collapse Rate monitoring."),
        Technique("avspo", "objective_extension", True, "wordle_lab.methods.grpo_stability", "Virtual rewards alter normalization only."),
        Technique("dynamic_state_curriculum", "data", True, "wordle_lab.data.canonical", "Training-only partial-game states."),
        Technique("multigranularity_reward", "reward", True, "wordle_lab.methods.reward_rubrics", "Auditable per-component training reward."),
        Technique(
            "structured_letter_prompt",
            "prompt",
            False,
            "not enabled",
            "Changing the frozen prompt requires a separately named protocol and matched baseline.",
        ),
        Technique(
            "hardcoded_opener",
            "harness_policy",
            False,
            "intentionally rejected",
            "Harness-selected guesses violate natural-generation and no-cheating requirements.",
        ),
    )
}


def technique_manifest() -> list[dict]:
    return [asdict(TECHNIQUES[name]) for name in sorted(TECHNIQUES)]


def validate_canonical_techniques(names: list[str]) -> list[Technique]:
    unknown = sorted(set(names) - set(TECHNIQUES))
    if unknown:
        raise ValueError(f"unknown techniques: {unknown}")
    selected = [TECHNIQUES[name] for name in names]
    incompatible = [item.name for item in selected if not item.canonical_compatible]
    if incompatible:
        raise ProtocolCompatibilityError(
            "WORDLE-PROTOCOL-002 rejects canonical execution of: " + ", ".join(incompatible)
        )
    return selected
