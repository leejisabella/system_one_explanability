"""E.116 figure, regenerated from the saved run summary (no API calls).

Usage:  uv run --group analysis python research_analyses/E116/make_figures.py

Reads data/summary.json (copied from results/runs/e116-v1-2026-11-23) and writes
figures/e116_no_arm_helps.png: each arm's paired change against the standard prompt (WildGuard
pooled, 95% interval) on three measures, one panel each.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,
})

ARMS = [("clarify", "Clarify"), ("grounded", "Grounded"), ("examples", "Examples"), ("clarify+grounded", "Clarify + grounded")]
PANELS = [("override_over_flagged", "Overrides Jev's over-flagged\nbenign items (n = 490)"),
          ("override_let_through", "Overrides harmful items\nJev let through (n = 178)"),
          ("cascade_accuracy", "WildGuard cascade accuracy\n(weighted to the pools)")]


def main() -> None:
    pooled = json.loads((HERE / "data" / "summary.json").read_text())["pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    ys = list(range(len(ARMS)))[::-1]
    for ax, (key, title) in zip(axes, PANELS):
        for y, (arm, _) in zip(ys, ARMS):
            v = pooled[arm]["vs_standard"][key]
            est, (lo, hi) = 100 * v["diff"], (100 * v["ci95"][0], 100 * v["ci95"][1])
            ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(est, y, "o", color=BLUE, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{est:+.1f}", (est, y), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK_2,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_xlim(-19, 13)
        ax.set_ylim(-0.6, len(ARMS) - 0.4)
        ax.set_title(title)
        ax.set_xlabel("Change vs. standard prompt, points [95%]")
    axes[0].set_yticks(ys, [label for _, label in ARMS])
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout()
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "e116_no_arm_helps.png", dpi=200)
    print(out / "e116_no_arm_helps.png")


if __name__ == "__main__":
    main()
