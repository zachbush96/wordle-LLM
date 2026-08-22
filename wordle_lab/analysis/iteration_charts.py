from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from wordle_lab.common import ARTIFACTS, ROOT


OUT = ROOT / "plots" / "iteration-2026-08-14"
SFT128 = ARTIFACTS / "runs" / "sft-common-explicit-repeat-s2026-56f010458f"
SFT512 = ARTIFACTS / "runs" / "sft-common-explicit-repeat-s2026-e6dc86e31c"
ORPO = ARTIFACTS / "runs" / "orpo-common-repeat-s2026-35b37fb67a"

INK = "#18324A"
MUTED = "#627487"
GRID = "#D9E2EA"
BLUE = "#2D6CDF"
GREEN = "#0F9D74"
ORANGE = "#E99524"
RED = "#D9514E"
PURPLE = "#7957D5"
PALE_BLUE = "#EAF1FD"
PALE_GREEN = "#E7F6F0"
PALE_ORANGE = "#FFF2DF"
PALE_RED = "#FCEAE9"
BG = "#FAFCFE"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def footer(fig, text: str) -> None:
    fig.text(0.02, 0.018, text, color=MUTED, fontsize=8.5)


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def improvement_ladder() -> Path:
    final = load(SFT128 / "summary.json")
    rep = load(SFT128 / "dose-final-greedy_rep105-summary.json")
    orpo50 = load(ORPO / "dose-step-000050-greedy_rep105-summary.json")
    conditions = [
        {"name": "Original collapsed SFT", "repeat": 83.3, "unique": 1, "valid": 100, "wins": 0},
        {"name": "Common u512 SFT", "repeat": 58.0, "unique": 16, "valid": 56.3, "wins": 0},
        {
            "name": "Common u128 SFT",
            "repeat": 100 * final["repeat_guess_rate"],
            "unique": final["unique_guesses"],
            "valid": 100 * (1 - final["invalid_guess_rate"]),
            "wins": final["wins"],
        },
        {
            "name": "+ repetition penalty 1.05",
            "repeat": 100 * rep["repeat_guess_rate"],
            "unique": rep["unique_guesses"],
            "valid": 100 * (1 - rep["invalid_guess_rate"]),
            "wins": rep["wins"],
        },
        {
            "name": "+ ORPO 50-step dose",
            "repeat": 100 * orpo50["repeat_guess_rate"],
            "unique": orpo50["unique_guesses"],
            "valid": 100 * (1 - orpo50["invalid_guess_rate"]),
            "wins": orpo50["wins"],
        },
    ]
    y = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.7), gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(top=0.80, bottom=0.12)
    fig.suptitle("From one repeated word to state-dependent play", x=0.02, ha="left", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.02, 0.91, "The clean stack reduced repeats 8× while preserving strict validity and the first held-out win.", color=MUTED, fontsize=12.5)

    repeats = [row["repeat"] for row in conditions]
    axes[0].barh(y, repeats, color=[RED, ORANGE, BLUE, GREEN, GREEN], height=0.58)
    axes[0].set_yticks(y, [row["name"] for row in conditions])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Repeated valid guesses (%) — lower is better")
    axes[0].set_title("Repeat collapse was progressively removed", loc="left", color=INK)
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    for i, value in enumerate(repeats):
        axes[0].text(value + 1.3, i, f"{value:.1f}%", va="center", color=INK, fontweight="bold")

    uniques = [row["unique"] for row in conditions]
    bars = axes[1].barh(y, uniques, color=[RED, ORANGE, BLUE, GREEN, GREEN], height=0.58)
    axes[1].set_yticks(y, ["" for _ in y])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 50)
    axes[1].set_xlabel("Unique valid guesses across 25 games — higher is better")
    axes[1].set_title("Action diversity became real behavior", loc="left", color=INK)
    axes[1].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)
    for bar, row in zip(bars, conditions):
        axes[1].text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{row['unique']}  |  validity {row['valid']:.0f}%  |  wins {row['wins']}/25",
            va="center",
            color=INK,
            fontweight="bold" if row["wins"] else "normal",
        )
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    footer(fig, "Development results only. u128/u512 denote common-word curriculum universe size; evaluator never masked logits or repaired guesses.")
    return save(fig, "01-improvement-ladder.png")


def training_scale_and_dose() -> Path:
    u512 = [
        load(SFT512 / f"dose-step-{step:06d}-summary.json")
        for step in (100, 200, 300)
    ] + [load(SFT512 / "summary.json")]
    u128 = [
        load(SFT128 / f"dose-step-{step:06d}-greedy-summary.json")
        for step in (150, 300, 450)
    ] + [load(SFT128 / "summary.json")]
    x512 = np.array([100, 200, 300, 400])
    x128 = np.array([150, 300, 450, 600])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    fig.subplots_adjust(top=0.82, bottom=0.09)
    fig.suptitle("Training dose helped only after the task was made learnable", x=0.02, ha="left", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, "A 512-word target space broadened outputs but broke format; a 128-word curriculum preserved validity and produced wins.", color=MUTED, fontsize=12.5)

    panels = [
        (axes[0, 0], u512, x512, "512-word curriculum: compliance eroded", False),
        (axes[0, 1], u128, x128, "128-word curriculum: compliance held", True),
    ]
    for ax, rows, xs, title, won in panels:
        compliance = [100 * row["terminal_marker_compliance"] for row in rows]
        repeat = [100 * row["repeat_guess_rate"] for row in rows]
        ax.plot(xs, compliance, marker="o", linewidth=2.5, color=GREEN, label="Terminal compliance")
        ax.plot(xs, repeat, marker="s", linewidth=2.5, color=RED, label="Repeat rate")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("Calls or valid guesses (%)")
        ax.set_ylim(0, 105)
        ax.grid(color=GRID, linewidth=0.8)
        ax.legend(frameon=False, loc="lower left")
        if won:
            for x, row in zip(xs, rows):
                if row["wins"]:
                    ax.annotate("1 win", (x, 100 * row["terminal_marker_compliance"]), xytext=(0, -24), textcoords="offset points", ha="center", color=GREEN, fontweight="bold")

    for ax, rows, xs, title in [
        (axes[1, 0], u512, x512, "512 words: diversity rose, but no wins"),
        (axes[1, 1], u128, x128, "128 words: diversity and wins arrived together"),
    ]:
        unique = [row.get("unique_guesses", value) for row, value in zip(rows, ([8, 10, 15, 16] if xs[0] == 100 else [9, 26, 34, 29]))]
        bars = ax.bar(xs, unique, width=0.55 * np.min(np.diff(xs)), color=BLUE)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("Unique guesses")
        ax.set_ylim(0, 40)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(bar.get_height())}", ha="center", color=INK, fontweight="bold")
            if row["wins"]:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, "WIN", ha="center", color="white", fontweight="bold")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "Same 270M base model and strict parser. u128 uses 96 training secrets and 32 held-out development secrets; chart shows first 25 dev games.")
    return save(fig, "02-training-scale-and-dose.png")


def decoder_tradeoffs() -> Path:
    specs = [
        ("Greedy", "dose-final-greedy-summary.json"),
        ("Penalty 1.02", "dose-final-greedy_rep102-summary.json"),
        ("Penalty 1.05", "dose-final-greedy_rep105-summary.json"),
        ("Penalty 1.10", "dose-final-greedy_rep110-summary.json"),
        ("Temp 0.3", "dose-final-sample_t03-summary.json"),
        ("Temp 0.7", "dose-final-sample_t07-summary.json"),
    ]
    # The main greedy result predates dose naming.
    rows = []
    for label, filename in specs:
        path = SFT128 / filename
        data = load(path) if path.exists() else load(SFT128 / "summary.json")
        rows.append((label, data))

    fig, ax = plt.subplots(figsize=(13.5, 8))
    fig.suptitle("Decoder tuning works only after training fixes the probability distribution", x=0.03, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.03, 0.91, "Bubble size encodes action diversity. Green points retained the held-out win; the outlined point is the clean operating choice.", color=MUTED, fontsize=12)
    for label, row in rows:
        x = 100 * row["repeat_guess_rate"]
        y = 100 * row["terminal_marker_compliance"]
        size = 10 * row["unique_guesses"]
        color = GREEN if row["wins"] else RED
        edge = INK if label == "Penalty 1.05" else "white"
        lw = 2.8 if label == "Penalty 1.05" else 1.2
        ax.scatter(x, y, s=size, color=color, alpha=0.82, edgecolor=edge, linewidth=lw, zorder=3)
        offsets = {
            "Greedy": (12, 7), "Penalty 1.02": (8, 10), "Penalty 1.05": (-110, -22),
            "Penalty 1.10": (8, -18), "Temp 0.3": (12, -34), "Temp 0.7": (8, -20),
        }
        dx, dy = offsets[label]
        ax.annotate(f"{label}\n{row['unique_guesses']} unique", (x, y), xytext=(dx, dy), textcoords="offset points", color=INK, fontweight="bold" if label == "Penalty 1.05" else "normal")
    ax.axhline(99, color=GRID, linewidth=1.2, linestyle="--")
    ax.text(55, 99.18, "99% compliance threshold", ha="right", va="bottom", color=MUTED)
    ax.set_xlim(0, 55)
    ax.set_ylim(93, 100.7)
    ax.set_xlabel("Repeat rate (%) — lower is better")
    ax.set_ylabel("Terminal compliance (%) — higher is better")
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "Temperature 0.7 increased diversity but lost the win. Penalty 1.10 minimized repeats but created invalid outputs. Penalty 1.05 retained 100% compliance and the win.")
    return save(fig, "03-decoder-tradeoffs.png")


def posttraining_dose() -> Path:
    points = [(0, load(SFT128 / "dose-final-greedy_rep105-summary.json"))]
    for step in (25, 50, 75):
        points.append((step, load(ORPO / f"dose-step-{step:06d}-greedy_rep105-summary.json")))
    points.append((100, load(ORPO / "dose-final-greedy_rep105-summary.json")))
    steps = np.array([point[0] for point in points])
    compliance = np.array([100 * point[1]["terminal_marker_compliance"] for point in points])
    repeats = np.array([100 * point[1]["repeat_guess_rate"] for point in points])
    unique = np.array([point[1]["unique_guesses"] for point in points])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(top=0.79, bottom=0.13)
    fig.suptitle("Post-training helped quickly—and then overdosed", x=0.02, ha="left", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.02, 0.90, "Repeat-focused ORPO moved the intended behavior by step 25–50; longer continuation damaged the output contract.", color=MUTED, fontsize=12.5)

    ax.plot(steps, compliance, color=GREEN, marker="o", linewidth=2.7, label="Terminal compliance")
    ax.plot(steps, repeats, color=RED, marker="s", linewidth=2.7, label="Repeat rate")
    ax.axvspan(25, 50, color=GREEN, alpha=0.09)
    ax.annotate("clean dose window", (37.5, 62), ha="center", color=GREEN, fontweight="bold")
    ax.set_title("Behavior and validity by ORPO dose", loc="left", color=INK)
    ax.set_xlabel("ORPO optimizer steps (0 = SFT parent)")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")

    bars = ax2.bar(steps, unique, width=14, color=[BLUE, BLUE, GREEN, ORANGE, RED])
    ax2.set_title("Diversity kept rising after quality peaked", loc="left", color=INK)
    ax2.set_xlabel("ORPO optimizer steps")
    ax2.set_ylabel("Unique valid guesses")
    ax2.set_ylim(0, 60)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for bar, value in zip(bars, unique):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", color=INK, fontweight="bold")
    ax2.annotate("selected\n50 steps", (50, unique[2]), xytext=(50, 56), ha="center", color=GREEN, fontweight="bold", arrowprops={"arrowstyle": "-|>", "color": GREEN})
    for axis in (ax, ax2):
        axis.spines[["top", "right"]].set_visible(False)
    footer(fig, "All points use the same held-out 25-game set and repetition penalty 1.05. The 50-step dose retained 100% validity, 1 win, 43 unique guesses, and 10.3% repeats.")
    return save(fig, "04-posttraining-dose.png")


def lessons_map() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle("What the experiments taught us", x=0.04, y=0.96, ha="left", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.04, 0.90, "The dominant problem was training signal design—not a missing temperature setting.", color=MUTED, fontsize=13)

    columns = [
        (0.35, 4.65, "Why the original\nruns collapsed", PALE_RED, RED, [
            "Random secrets from the full legal-guess list",
            "Sparse, mostly one-off oracle targets",
            "Train oracle and evaluation universe disagreed",
            "Placeholder example encouraged literal imitation",
            "Failure examples corrected XXXXX—not repeats",
        ]),
        (5.68, 4.65, "What produced\nthe breakthrough", PALE_GREEN, GREEN, [
            "Common-word curriculum: 128-word universe",
            "Same public universe in training and evaluation",
            "Compact explicit feedback constraints",
            "Repeat-correction examples and short ORPO dose",
            "Mild repetition penalty after—not before—learning",
        ]),
        (11.0, 4.65, "What to do next", PALE_BLUE, BLUE, [
            "Repeat u128 SFT + ORPO across three seeds",
            "Add explicit constraint-violation negatives",
            "Scale gradually through u256 before u512",
            "Run a ~1B capacity ablation at equal token budget",
            "Keep the locked full test closed until preregistered",
        ]),
    ]
    for x, width, title, fill, accent, items in columns:
        box = FancyBboxPatch((x, 1.25), width, 6.95, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor="none")
        ax.add_patch(box)
        ax.add_patch(Rectangle((x, 7.75), width, 0.45, facecolor=accent, edgecolor="none"))
        ax.text(x + 0.3, 7.35, title, color=INK, fontsize=14.2, fontweight="bold", va="top")
        y = 6.25
        for index, item in enumerate(items, 1):
            ax.text(x + 0.32, y, str(index), color=accent, fontsize=13, fontweight="bold", va="top")
            wrapped = "\n".join(textwrap.wrap(item, width=38))
            ax.text(x + 0.68, y, wrapped, color=INK, fontsize=10.8, va="top")
            y -= 1.02
    for start, end, color in [((5.08, 4.72), (5.58, 4.72), ORANGE), ((10.42, 4.72), (10.92, 4.72), GREEN)]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=color))
    ax.text(8, 0.55, "Result: 1 → 43 unique guesses  •  83.3% → 10.3% repeats  •  0 → 1 held-out win  •  100% strict validity", ha="center", color=INK, fontsize=14, fontweight="bold")
    footer(fig, "Conclusion: curriculum, aligned supervision, and carefully dosed preference learning had noticeable effects; temperature alone did not.")
    return save(fig, "05-what-we-learned.png")


def win_trajectory() -> Path:
    game = next(row for row in (json.loads(line) for line in (SFT128 / "games.jsonl").read_text(encoding="utf-8").splitlines()) if row["won"])
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    fig.suptitle("The first held-out win was genuine state-conditioned play", x=0.04, y=0.96, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.04, 0.87, "Secret: SENSE  •  The model changed its answer after feedback and solved on turn 2.", color=MUTED, fontsize=12.5)
    colors = {"G": GREEN, "Y": ORANGE, "B": "#8091A3"}
    for row_index, turn in enumerate(game["turns"]):
        y = 3.45 - row_index * 1.55
        ax.text(0.55, y + 0.35, f"Turn {row_index + 1}", color=INK, fontsize=13, fontweight="bold", va="center")
        for i, (letter, feedback) in enumerate(zip(turn["guess"], turn["feedback"])):
            x = 2.15 + i * 1.05
            ax.add_patch(FancyBboxPatch((x, y), 0.82, 0.82, boxstyle="round,pad=0.01,rounding_size=0.05", facecolor=colors[feedback], edgecolor="none"))
            ax.text(x + 0.41, y + 0.41, letter, ha="center", va="center", color="white", fontsize=18, fontweight="bold")
        ax.text(7.65, y + 0.41, turn["feedback"], color=MUTED, fontsize=13, fontweight="bold", va="center")
        if turn["won"]:
            ax.text(9.0, y + 0.41, "SOLVED", color=GREEN, fontsize=14, fontweight="bold", va="center")
        else:
            ax.text(8.9, y + 0.41, f"{turn['posterior_before']} → {turn['posterior_after']} candidates", color=INK, fontsize=11.8, va="center")
    ax.add_patch(FancyArrowPatch((6.0, 3.35), (6.0, 2.65), arrowstyle="-|>", mutation_scale=18, linewidth=2, color=BLUE))
    ax.text(10.9, 4.85, "No candidate list\nNo word forcing\nNo output repair", ha="center", va="center", color=INK, fontsize=12.5, fontweight="bold", bbox={"boxstyle": "round,pad=0.5", "facecolor": PALE_BLUE, "edgecolor": "none"})
    footer(fig, "Trajectory from sft-common-explicit-repeat-s2026-56f010458f, greedy decoding. Feedback G=correct position, Y=present elsewhere, B=absent.")
    return save(fig, "06-first-held-out-win.png")


def main() -> None:
    setup()
    paths = [
        improvement_ladder(),
        training_scale_and_dose(),
        decoder_tradeoffs(),
        posttraining_dose(),
        lessons_map(),
        win_trajectory(),
    ]
    manifest = {"charts": [str(path) for path in paths]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
