from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def data_representation_leaderboard(table: pd.DataFrame, output: Path) -> Path:
    subset = table[table["method"] == "sft"].copy().sort_values("win_rate")
    if subset.empty:
        raise ValueError("no SFT results")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(subset["win_rate"] * 100, subset["representation"], c="#2563eb")
    axis.set(xlabel="Development win rate (%)", ylabel="Representation", title=f"SFT data screen ({len(subset)} runs)")
    axis.grid(axis="x", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)
    return output
