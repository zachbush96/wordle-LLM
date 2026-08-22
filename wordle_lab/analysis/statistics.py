from __future__ import annotations

import math
import numpy as np


def paired_bootstrap_delta(base: list[bool], treatment: list[bool], seed: int = 1337, samples: int = 10_000) -> dict:
    if len(base) != len(treatment) or not base:
        raise ValueError("paired non-empty outcomes required")
    delta = np.asarray(treatment, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)])
    return {"delta": float(delta.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def mcnemar_exact(base: list[bool], treatment: list[bool]) -> dict:
    improved = sum((not left) and right for left, right in zip(base, treatment))
    regressed = sum(left and (not right) for left, right in zip(base, treatment))
    n = improved + regressed
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(improved, regressed) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"improved": improved, "regressed": regressed, "p_value": p}


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
