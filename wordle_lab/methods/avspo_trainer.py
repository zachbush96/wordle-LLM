from __future__ import annotations

import math

import torch
from trl import GRPOTrainer

from wordle_lab.methods.grpo_stability import (
    advantage_collapse_diagnostics,
    supported_advantage_estimate,
    update_adaptive_acr_threshold,
    validate_virtual_support_spec,
)


def avspo_group_advantages(
    reward_groups: list[list[float]],
    virtual_support: dict,
    *,
    adaptive_threshold: float,
) -> tuple[list[float], dict]:
    """Compute trainable real-sample advantages and an audit record."""
    diagnostics = advantage_collapse_diagnostics(
        reward_groups,
        std_threshold=float(virtual_support.get("collapse_std_threshold", 1e-6)),
    )
    acr = diagnostics["advantage_collapse_rate"]
    advantages: list[float] = []
    virtual_count = 0
    for group in reward_groups:
        estimate = supported_advantage_estimate(
            group,
            virtual_support,
            batch_acr=acr,
            adaptive_threshold=adaptive_threshold,
        )
        advantages.extend(estimate["real_advantages"])
        virtual_count += estimate["virtual_sample_count"]
    return advantages, {
        "advantage_collapse_rate": acr,
        "virtual_sample_count": virtual_count,
        "adaptive_threshold": adaptive_threshold,
        "real_sample_count": len(advantages),
    }


class AVSPOGRPOTrainer(GRPOTrainer):
    """TRL 1.10 GRPO trainer with AVSPO normalization for real samples.

    Reward functions and environment rollouts remain unchanged. Synthetic
    scalar support is used only to replace the already-computed advantage
    tensor before the policy loss; it is never appended to completions.
    """

    def __init__(self, *args, avspo_spec: dict, **kwargs):
        self.avspo_spec = {**dict(avspo_spec), **validate_virtual_support_spec(avspo_spec)}
        self._avspo_threshold = self.avspo_spec["adaptive_threshold_initial"]
        self._avspo_previous_objective: float | None = None
        self._avspo_latest_rewards_per_func: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        values = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._avspo_latest_rewards_per_func = values.detach()
        return values

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training or not self.avspo_spec["enabled"]:
            return output
        per_func = self._avspo_latest_rewards_per_func
        if per_func is None:
            raise RuntimeError("AVSPO reward capture was not populated by TRL")
        combined = (per_func * self.reward_weights.to(per_func.device).unsqueeze(0)).nansum(dim=1)
        unscorable = torch.isnan(per_func).all(dim=1)
        combined[unscorable] = torch.nan
        group_size = int(self.num_generations)
        if combined.numel() % group_size:
            raise RuntimeError("gathered rewards do not divide into GRPO groups")
        groups = combined.view(-1, group_size)
        if torch.isnan(groups).any():
            raise RuntimeError("AVSPO requires every completion in a reward group to be scorable")
        group_values = groups.detach().cpu().tolist()
        flat, audit = avspo_group_advantages(
            group_values,
            self.avspo_spec,
            adaptive_threshold=self._avspo_threshold,
        )
        all_advantages = torch.tensor(flat, dtype=output["advantages"].dtype, device=output["advantages"].device)
        local_count = len(inputs)
        start = self.accelerator.process_index * local_count
        output["advantages"] = all_advantages[start : start + local_count]

        objective = float(torch.mean(combined).item())
        delta = 0.0 if self._avspo_previous_objective is None else objective - self._avspo_previous_objective
        self._avspo_threshold = update_adaptive_acr_threshold(
            self._avspo_threshold,
            audit["advantage_collapse_rate"],
            delta,
            eta=self.avspo_spec["adaptive_eta"],
        )
        self._avspo_previous_objective = objective
        metrics = self._metrics["train"]
        metrics["avspo/acr"].append(audit["advantage_collapse_rate"])
        metrics["avspo/virtual_samples"].append(audit["virtual_sample_count"])
        metrics["avspo/adaptive_threshold"].append(self._avspo_threshold)
        metrics["avspo/mean_real_advantage"].append(
            sum(flat) / len(flat) if flat else math.nan
        )
        return output
