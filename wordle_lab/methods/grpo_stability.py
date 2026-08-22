"""Safety and stability utilities for grouped Wordle policy optimization.

The helpers in this module are deliberately independent of TRL.  They can be
used to validate a run before allocating a model and to log diagnostics during
training.  In particular, virtual samples are numerical support points for
advantage normalization only: they are not rollouts and never claim an
environment outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wordle_lab.methods.rewards import DEFAULT_WEIGHTS, REWARD_VERSION


VIRTUAL_SAMPLE_TYPE = "synthetic_virtual_reward"
VIRTUAL_SAMPLE_USAGE = "advantage_estimation_only"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_reward_rubric(rubric: Mapping[str, object]) -> dict:
    """Validate and normalize a rubric accepted by ``shaped_reward``.

    Exact component names are required so a configuration cannot silently
    omit a penalty or introduce an unimplemented reward channel.
    """

    if not isinstance(rubric, Mapping):
        raise ValueError("reward_rubric must be a mapping")
    version = rubric.get("version", REWARD_VERSION)
    if version != REWARD_VERSION:
        raise ValueError(f"unsupported reward rubric version: {version}")
    raw_weights = rubric.get("weights", DEFAULT_WEIGHTS)
    if not isinstance(raw_weights, Mapping):
        raise ValueError("reward_rubric.weights must be a mapping")
    expected = set(DEFAULT_WEIGHTS)
    actual = set(raw_weights)
    if actual != expected:
        raise ValueError(
            f"reward weight keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    weights = {name: _finite_number(raw_weights[name], f"reward weight {name}") for name in DEFAULT_WEIGHTS}
    return {"version": version, "weights": weights}


def advantage_collapse_diagnostics(
    reward_groups: Sequence[Sequence[float]], *, std_threshold: float = 1e-6
) -> dict:
    """Report the fraction of rollout groups with no useful reward contrast.

    A group is collapsed when its population reward standard deviation is
    strictly below ``std_threshold``. Empty and singleton groups are invalid rather than being
    counted as collapsed because GRPO cannot estimate their relative advantage.
    """

    threshold = _finite_number(std_threshold, "std_threshold")
    if threshold < 0:
        raise ValueError("std_threshold must be non-negative")
    if not reward_groups:
        raise ValueError("at least one reward group is required")
    rows = []
    for group_index, group in enumerate(reward_groups):
        if len(group) < 2:
            raise ValueError(f"reward group {group_index} must contain at least two samples")
        values = [_finite_number(value, f"reward_groups[{group_index}]") for value in group]
        reward_range = max(values) - min(values)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        reward_std = math.sqrt(variance)
        rows.append(
            {
                "group_index": group_index,
                "size": len(values),
                "reward_mean": mean,
                "reward_std": reward_std,
                "reward_range": reward_range,
                "collapsed": reward_std < threshold,
            }
        )
    collapsed = sum(row["collapsed"] for row in rows)
    return {
        "advantage_collapse_rate": collapsed / len(rows),
        "collapsed_groups": collapsed,
        "total_groups": len(rows),
        "std_threshold": threshold,
        "groups": rows,
    }


def validate_virtual_support_spec(spec: Mapping[str, object] | None) -> dict:
    raw = dict(spec or {})
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("virtual_support.enabled must be boolean")
    alpha = _finite_number(raw.get("alpha", 0.5), "virtual_support.alpha")
    anchor = _finite_number(raw.get("zero_reward_anchor", 0.1), "virtual_support.zero_reward_anchor")
    adaptive_threshold = _finite_number(raw.get("adaptive_threshold_initial", 0.5), "virtual_support.adaptive_threshold_initial")
    adaptive_eta = _finite_number(raw.get("adaptive_eta", 0.01), "virtual_support.adaptive_eta")
    epsilon = _finite_number(raw.get("normalization_epsilon", 1e-6), "virtual_support.normalization_epsilon")
    if alpha <= 0 or anchor <= 0 or adaptive_eta <= 0 or epsilon <= 0:
        raise ValueError("virtual support alpha, anchor, eta, and normalization_epsilon must be positive")
    if not 0 <= adaptive_threshold <= 1:
        raise ValueError("adaptive_threshold_initial must be between zero and one")
    return {
        "enabled": enabled,
        "alpha": alpha,
        "zero_reward_anchor": anchor,
        "adaptive_threshold_initial": adaptive_threshold,
        "adaptive_eta": adaptive_eta,
        "normalization_epsilon": epsilon,
        "usage": VIRTUAL_SAMPLE_USAGE,
    }


def update_adaptive_acr_threshold(current: float, acr: float, objective_delta: float, *, eta: float = 0.01) -> float:
    """Apply the AVSPO adaptive-threshold update exactly.

    ``current + eta * sign(objective_delta) * (acr - current)`` is returned
    without implicit clipping so the logged value remains faithful to the
    stated update rule.
    """

    current_value = _finite_number(current, "current threshold")
    acr_value = _finite_number(acr, "ACR")
    delta = _finite_number(objective_delta, "objective_delta")
    rate = _finite_number(eta, "eta")
    if not 0 <= current_value <= 1 or not 0 <= acr_value <= 1 or rate <= 0:
        raise ValueError("threshold and ACR must be in [0,1], and eta must be positive")
    sign = 1.0 if delta > 0 else -1.0 if delta < 0 else 0.0
    return current_value + rate * sign * (acr_value - current_value)


def virtual_sample_count(group_size: int, acr: float, *, alpha: float = 0.5) -> int:
    """Return ``max(1, min(G, ceil(G * ACR**alpha)))``."""

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    acr_value = _finite_number(acr, "ACR")
    power = _finite_number(alpha, "alpha")
    if not 0 <= acr_value <= 1 or power <= 0:
        raise ValueError("ACR must be in [0,1] and alpha must be positive")
    return max(1, min(group_size, math.ceil(group_size * acr_value**power)))


def virtual_advantage_support(
    real_rewards: Sequence[float],
    spec: Mapping[str, object] | None,
    *,
    batch_acr: float,
    adaptive_threshold: float,
) -> list[dict]:
    """Create explicitly synthetic AVSPO normalization support points.

    The values are centered on observed rewards and use only their numerical
    They must never be appended to trajectories, reward logs, or environment
    outcomes. Support is produced only when batch ACR exceeds the adaptive
    threshold and the current group is collapsed.
    """

    config = validate_virtual_support_spec(spec)
    values = [_finite_number(value, "real reward") for value in real_rewards]
    if len(values) < 2:
        raise ValueError("at least two real rewards are required")
    acr_value = _finite_number(batch_acr, "batch_acr")
    threshold_value = _finite_number(adaptive_threshold, "adaptive_threshold")
    if not 0 <= acr_value <= 1 or not 0 <= threshold_value <= 1:
        raise ValueError("batch_acr and adaptive_threshold must be between zero and one")
    group_std = math.sqrt(sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values))
    collapse_tau = _finite_number(dict(spec or {}).get("collapse_std_threshold", 1e-6), "collapse_std_threshold")
    triggered = config["enabled"] and acr_value > threshold_value and group_std < collapse_tau
    if not triggered:
        return []
    count = virtual_sample_count(len(values), acr_value, alpha=config["alpha"])
    observed_max = max(values)
    if observed_max > 0:
        rewards = [observed_max * (1 - k / (count + 1)) for k in range(1, count + 1)]
        schedule = "positive_observed_reward_decay"
    elif all(value == 0 for value in values):
        anchor = config["zero_reward_anchor"]
        rewards = [anchor * (count - k + 1) / count for k in range(1, count + 1)]
        schedule = "all_zero_anchor_decay"
    else:
        # The source rule is defined for binary/non-negative rewards. Wordle's
        # shaped rubric can yield an all-negative collapsed group; inventing an
        # extension here would be an unvalidated change to the method.
        return []
    return [
        {
            "reward": reward,
            "sample_type": VIRTUAL_SAMPLE_TYPE,
            "synthetic": True,
            "environment_outcome": False,
            "usage": VIRTUAL_SAMPLE_USAGE,
            "schedule": schedule,
        }
        for reward in rewards
    ]


def supported_advantage_estimate(
    real_rewards: Sequence[float],
    spec: Mapping[str, object] | None = None,
    *,
    batch_acr: float | None = None,
    adaptive_threshold: float | None = None,
) -> dict:
    """Normalize real rewards with optional AVSPO-style virtual support.

    Only advantages corresponding to real rewards are returned as trainable
    values. Virtual points are retained as auditable metadata.
    """

    config = validate_virtual_support_spec(spec)
    values = [_finite_number(value, "real reward") for value in real_rewards]
    if len(values) < 2:
        raise ValueError("at least two real rewards are required")
    group_acr = advantage_collapse_diagnostics([values], std_threshold=float(dict(spec or {}).get("collapse_std_threshold", 1e-6)))[
        "advantage_collapse_rate"
    ]
    effective_acr = group_acr if batch_acr is None else batch_acr
    effective_threshold = config["adaptive_threshold_initial"] if adaptive_threshold is None else adaptive_threshold
    virtual = virtual_advantage_support(
        values, config, batch_acr=effective_acr, adaptive_threshold=effective_threshold
    )
    normalization_values = values + [row["reward"] for row in virtual]
    mean = sum(normalization_values) / len(normalization_values)
    variance = sum((value - mean) ** 2 for value in normalization_values) / len(normalization_values)
    denominator = max(math.sqrt(variance), config["normalization_epsilon"])
    return {
        "real_advantages": [(value - mean) / denominator for value in values],
        "real_sample_count": len(values),
        "virtual_sample_count": len(virtual),
        "normalization_mean": mean,
        "normalization_std": math.sqrt(variance),
        "virtual_samples": virtual,
        "virtual_sample_usage": VIRTUAL_SAMPLE_USAGE,
    }


def entropy_collapse_diagnostics(entropies: Sequence[float], spec: Mapping[str, object]) -> dict:
    """Evaluate an entropy trace using an absolute and baseline-relative floor."""

    values = [_finite_number(value, "entropy") for value in entropies]
    if any(value < 0 for value in values):
        raise ValueError("entropy values must be non-negative")
    baseline_window = int(spec.get("baseline_window", 5))
    patience = int(spec.get("patience", 3))
    minimum_observations = int(spec.get("minimum_observations", baseline_window + patience))
    absolute_floor = _finite_number(spec.get("absolute_floor", 0.25), "entropy absolute_floor")
    relative_floor = _finite_number(spec.get("relative_floor", 0.35), "entropy relative_floor")
    if baseline_window < 1 or patience < 1 or minimum_observations < baseline_window:
        raise ValueError("invalid entropy guard window or patience")
    if absolute_floor < 0 or not 0 < relative_floor <= 1:
        raise ValueError("entropy floors must be absolute>=0 and 0<relative<=1")
    if not values:
        return {"stop": False, "reason": None, "observations": 0, "consecutive_below_floor": 0}
    baseline_values = values[: min(baseline_window, len(values))]
    baseline = sum(baseline_values) / len(baseline_values)
    floor = max(absolute_floor, baseline * relative_floor)
    consecutive = 0
    for value in reversed(values):
        if value < floor:
            consecutive += 1
        else:
            break
    stop = len(values) >= minimum_observations and consecutive >= patience
    return {
        "stop": stop,
        "reason": "entropy_collapse" if stop else None,
        "observations": len(values),
        "baseline_entropy": baseline,
        "effective_floor": floor,
        "latest_entropy": values[-1],
        "consecutive_below_floor": consecutive,
        "patience": patience,
    }


def stability_stop_decision(
    reward_groups: Sequence[Sequence[float]], entropies: Sequence[float], spec: Mapping[str, object]
) -> dict:
    """Combine advantage- and entropy-collapse guardrails without mutating training."""

    advantage = advantage_collapse_diagnostics(
        reward_groups, std_threshold=float(spec.get("advantage_std_threshold", 1e-6))
    )
    entropy = entropy_collapse_diagnostics(entropies, spec.get("entropy_guard", {}))
    maximum_rate = _finite_number(spec.get("maximum_advantage_collapse_rate", 0.8), "maximum collapse rate")
    if not 0 <= maximum_rate <= 1:
        raise ValueError("maximum_advantage_collapse_rate must be between zero and one")
    minimum_groups = int(spec.get("minimum_groups_before_stop", 10))
    advantage_stop = advantage["total_groups"] >= minimum_groups and advantage["advantage_collapse_rate"] > maximum_rate
    reasons = (["advantage_collapse"] if advantage_stop else []) + (["entropy_collapse"] if entropy["stop"] else [])
    return {"stop": bool(reasons), "reasons": reasons, "advantage": advantage, "entropy": entropy}
