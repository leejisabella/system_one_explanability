"""E.20 figure, drawn from data/summary.json (no API calls).

Usage:  uv run --group analysis python research_analyses/E20/make_figures.py

figures/e20_frontier.png: per pool, each reviewer variant's fix rate on Jev's mistakes against its
break rate on Jev's right labels (Jev upstream, E.102's subsample). The dashed line is break-even with
Jev alone (gain 0); points above it beat Jev. The grey curve is the ROC of gpt-oss-20b's alone vote
share as a detector of Jev's mistakes: what policies using only the reviewer's own answers can reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
GROUPS = [  # (label, colour, marker, variants)
    ("Standard prompt", "#52514e", "o", ["base_local", "e102_bedrock"]),
    ("Prompt protocols (E.103)", "#2a78d6", "s", ["e103_blind_first", "e103_warning", "e103_calibrated_warning",
                                                  "e103_consider_opposite"]),
    ("Trained reviewers (E.106–E.108)", "#d4762a", "D", ["sft_review", "sft_plain", "dpo_review", "consist_review",
                                                         "balanced_review"]),
    ("gpt-oss-120b (E.119)", "#2f9e6e", "^", ["e119_120b"]),
]
POOLS = [("civil_toxicity_pool_v1", "Civil Comments"), ("wildguard_prompt_full_v1", "WildGuard prompts"),
         ("wildguard_response_full_v1", "WildGuard responses")]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "text.parse_math": False,
})


def main() -> None:
    S = json.loads((HERE / "data" / "summary.json").read_text())
    F, roc = S["frontier"], S["roc_base_local_jev"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (pool, title) in zip(axes, POOLS):
        err = F["base_local"]["pools"][pool]["jev_err"]
        xs, ys = zip(*roc[pool])
        ax.plot([100 * x for x in xs], [100 * y for y in ys], color="#b9b7b1", lw=1.6, zorder=1,
                label="20b alone votes as a detector (ROC)")
        slope = (1 - err) / err
        ax.plot([0, 100 / slope], [0, 100], color=INK_2, lw=1, ls="--", zorder=1, label="Break-even with Jev alone")
        for label, colour, marker, names in GROUPS:
            pts = [F[n]["pools"][pool] for n in names if n in F]
            ax.scatter([100 * p["break"] for p in pts], [100 * p["fix"] for p in pts], s=46, color=colour, marker=marker,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 100)
        ax.set_title(f"{title} (Jev wrong on {100 * err:.0f}%)")
        ax.set_xlabel("Breaks Jev's right labels (%)")
    axes[0].set_ylabel("Fixes Jev's mistakes (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "e20_frontier.png", dpi=200)
    print(out / "e20_frontier.png")


if __name__ == "__main__":
    main()
