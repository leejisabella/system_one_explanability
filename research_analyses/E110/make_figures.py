"""E.110 figure, regenerated from the saved run summary (no API calls, no GPU).

Usage:  uv run --group analysis python research_analyses/E110/make_figures.py

Reads data/directions.json and data/summary.json (copied from results/runs/e110-v1-2026-11-22) and
writes figures/e110_found_not_causal.png: (left) held-out AUC of the give-in direction per layer at
three positions; (right) measurement 2, the paired change from steering on the test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,
})

# Measurement 2 (paired, unsteered -> steered at alpha 0.5, test set), from the run's report.md.
M2 = [("Jev's wrong label accepted\n(group A, n = 364)", -4.7, -9.9, 0.5),
      ("Jev's right label accepted\n(n = 1,033)", -4.2, -5.9, -2.3)]


def main() -> None:
    d = json.loads((HERE / "data" / "directions.json").read_text())
    auc = d["auc"]
    layers = list(range(len(auc["pre_answer"])))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    for key, label, color in [("pre_answer", "Just before the answer", BLUE),
                              ("reasoning_mean", "Mean over reasoning", ORANGE),
                              ("last_prompt", "Last prompt token (before reasoning)", MUTED)]:
        a.plot(layers, auc[key], color=color, lw=2, marker="o", ms=3)
        a.annotate(label, (layers[-1], auc[key][-1]), xytext=(6, 0), textcoords="offset points",
                   color=color if color != MUTED else INK_2, va="center", fontsize=9)
    a.axhline(0.70, color=INK_2, lw=1, ls="--")
    a.text(0, 0.715, "found threshold (0.70)", color=INK_2, fontsize=8.5)
    a.axhline(0.5, color=GRID, lw=1)
    a.set_xlim(-0.5, 33)
    a.set_xticks(range(0, 24, 4))
    a.set_ylim(0.4, 1.02)
    a.set_xlabel("Layer")
    a.set_ylabel("Held-out AUC, give-in vs. held")
    a.set_title("1. The direction is there, but only once the answer is reached")

    ys = range(len(M2))
    for y, (label, est, lo, hi) in zip(ys, M2):
        b.plot([lo, hi], [y, y], color=BLUE, lw=2.5)
        b.plot(est, y, "o", color=BLUE, ms=7)
        b.annotate(f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]", (est, y), xytext=(0, 10), textcoords="offset points",
                   ha="center", fontsize=9)
    b.axvline(0, color=INK_2, lw=1)
    b.set_yticks(list(ys), [m[0] for m in M2])
    b.set_ylim(-0.6, 1.6)
    b.set_xlim(-12, 3)
    b.set_xlabel("Paired change from steering, points [95%]")
    b.set_title("2. Steering lowers both rates")
    fig.tight_layout()
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "e110_found_not_causal.png", dpi=200)
    print(out / "e110_found_not_causal.png")


if __name__ == "__main__":
    main()
