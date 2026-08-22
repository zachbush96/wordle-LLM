from __future__ import annotations


REWARD_VERSION = "wordle-shaped-v1"
DEFAULT_WEIGHTS = {"solve": 5.0, "information_gain": 1.0, "oracle_regret": -1.0, "repeat": -2.0, "format": -3.0}


def shaped_reward(*, solved: bool, information_gain: float, oracle_regret: float, repeated: bool, format_valid: bool, weights: dict | None = None) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    components = {
        "solve": weights["solve"] * float(solved),
        "information_gain": weights["information_gain"] * float(information_gain),
        "oracle_regret": weights["oracle_regret"] * float(oracle_regret),
        "repeat": weights["repeat"] * float(repeated),
        "format": weights["format"] * float(not format_valid),
    }
    return {"total": sum(components.values()), "components": components, "reward_version": REWARD_VERSION}
