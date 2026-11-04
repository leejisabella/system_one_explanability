"""E.101 figures and data, regenerated from the saved run.

Usage:  uv run --group analysis python research_analyses/E101_pilot/make_figures.py

Reads results/runs/e101-v1-2026-11-04/ (local only) and the frozen samples' items, re-runs
the E.101 analysis, writes derived tables to data/ (committed, so the numbers survive if
results/ is lost) and the figures to figures/. No API calls.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysone.analysis.e101 import analyze_run, rate  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "e101-v1-2026-11-04"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN
SAMPLES = [("civil_toxicity_v1", "Civil Comments (toxicity)"), ("wildguard_prompt_v1", "WildGuardTest (prompt harm)")]

# Reference palette (dataviz skill, light mode), as in E0 and E1. Both sets validated
# with the skill's validate_palette.js: blue/orange categorical, and the blue ordinal ramp.
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = {"always_right": "#86b6ef", "sometimes_right": "#2a78d6", "never_right": "#104281"}
DIFFICULTY = [("always_right", "Reviewer right alone\n(3 of 3)"), ("sometimes_right", "Sometimes right\n(1–2 of 3)"),
              ("never_right", "Never right\n(0 of 3)")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,
})


def read(name: str) -> list[dict]:
    return [json.loads(line) for line in (RUN_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / "data" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name: str) -> None:
    fig.savefig(HERE / "figures" / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    summary = analyze_run(RUN_DIR)
    for name in ("summary.json", "report.md"):
        shutil.copy(RUN_DIR / name, HERE / "data" / name)
    proposals, reviews, jev_rows = read("proposals"), read("reviews"), read("jev_decisions")

    # Per sample: gold labels, the reviewer's alone answers, Jev's decision.
    gold, alone, jev = {}, defaultdict(list), {}
    for r in reviews:
        gold[(r["sample"], r["item_id"])] = r["label"]
        if r["condition"] == "alone":
            alone[(r["sample"], r["item_id"])].append(r["final_label"] == r["label"])
    for p in proposals:
        if p["condition"] == "jev_conf":
            jev[(p["sample"], p["item_id"])] = p
    difficulty = {k: "always_right" if all(v) else "never_right" if not any(v) else "sometimes_right"
                  for k, v in alone.items()}

    # 1. Where wrong proposals land, by the reviewer's own difficulty, and how often they're accepted.
    landing, accept_rows = [], []
    for sample, _ in SAMPLES:
        for cond in ("jev_label", "matched_label"):
            wrong = [r for r in reviews if r["sample"] == sample and r["condition"] == cond and not r["proposed_correct"]]
            counts = Counter(difficulty[(sample, r["item_id"])] for r in wrong)
            landing.append({"sample": sample, "condition": cond, "n_wrong": len(wrong),
                            **{d: counts.get(d, 0) for d, _ in DIFFICULTY}})
            for d, _ in DIFFICULTY:
                rr = rate([r["final_label"] == r["proposed_label"] for r in wrong
                           if difficulty[(sample, r["item_id"])] == d])
                accept_rows.append({"sample": sample, "condition": cond, "difficulty": d, "accepted": rr["k"],
                                    "n": rr["n"], "rate": rr["rate"], "ci_lo": (rr["ci95"] or [None])[0],
                                    "ci_hi": (rr["ci95"] or [None, None])[1]})
    write_csv("wrong_proposals_by_difficulty.csv", landing)
    write_csv("acceptance_by_difficulty.csv", accept_rows)

    # Error overlap: Jev's errors vs the reviewer's (alone, majority), against independence.
    overlap = []
    for sample, _ in SAMPLES:
        keys = [k for k in jev if k[0] == sample]
        maj_wrong = {k: sum(alone[k]) < len(alone[k]) / 2 for k in keys}
        jev_wrong = {k: not jev[k]["correct"] for k in keys}
        n, jw, rw = len(keys), sum(jev_wrong.values()), sum(maj_wrong.values())
        overlap.append({"sample": sample, "n_items": n, "jev_wrong": jw, "reviewer_alone_wrong": rw,
                        "both_wrong": sum(jev_wrong[k] and maj_wrong[k] for k in keys),
                        "both_wrong_if_independent": round(jw * rw / n, 1),
                        "agree_with_each_other": sum(jev_wrong[k] == maj_wrong[k] for k in keys)})
    write_csv("error_overlap.csv", overlap)

    # Trade-off: override when wrong vs accept when right, per condition.
    trade = []
    for sample, _ in SAMPLES:
        for cond, c in summary["reviewers"]["reviewer"][sample]["conditions"].items():
            if cond != "alone":
                trade.append({"sample": sample, "condition": cond,
                              "accept_when_right": c["accept_when_right"]["rate"],
                              "override_when_wrong": c["override_when_wrong"]["rate"],
                              "accuracy": c["accuracy"]["rate"], "flip_rate": c["flip_rate"]["rate"],
                              "flip_n": c["flip_rate"]["n"], "mean_output_tokens": c["mean_output_tokens"]})
            else:
                trade.append({"sample": sample, "condition": cond, "accept_when_right": None,
                              "override_when_wrong": None, "accuracy": c["accuracy"]["rate"], "flip_rate": None,
                              "flip_n": None, "mean_output_tokens": c["mean_output_tokens"]})
    write_csv("conditions.csv", trade)

    # Pipelines: accuracy and cost per item, Jev called once (its first sample's cost),
    # the reviewer alone (its first sample), and cascades that review only below a Jev confidence.
    pipes = []
    for sample, _ in SAMPLES:
        keys = sorted(k for k in jev if k[0] == sample)
        jev_cost = statistics.mean(r["cost_usd"] for r in jev_rows if r["sample"] == sample and r["sample_index"] == 0)
        jev_lat = statistics.mean(r["latency_s"] for r in jev_rows if r["sample"] == sample and r["sample_index"] == 0)
        rev = {(r["sample"], r["item_id"]): r for r in reviews if r["sample"] == sample and r["condition"] == "jev_conf"}
        alone0 = {(r["sample"], r["item_id"]): r for r in reviews
                  if r["sample"] == sample and r["condition"] == "alone" and r["sample_index"] == 0}
        n = len(keys)
        pipes.append({"sample": sample, "pipeline": "Jev alone", "threshold": None, "reviewed": 0,
                      "accuracy": round(sum(jev[k]["correct"] for k in keys) / n, 4),
                      "cost_per_item_usd": round(jev_cost, 8), "latency_per_item_s": round(jev_lat, 3)})
        pipes.append({"sample": sample, "pipeline": "Reviewer alone", "threshold": None, "reviewed": n,
                      "accuracy": round(sum(alone0[k]["final_label"] == gold[k] for k in keys) / n, 4),
                      "cost_per_item_usd": round(statistics.mean(alone0[k]["cost_usd"] for k in keys), 8),
                      "latency_per_item_s": round(statistics.mean(alone0[k]["latency_s"] for k in keys), 3)})
        for t in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
            routed = [k for k in keys if jev[k]["confidence"] < t]
            final = {k: rev[k]["final_label"] if k in routed else jev[k]["label"] for k in keys}
            pipes.append({"sample": sample, "pipeline": "Jev, then review if confidence < t",
                          "threshold": t, "reviewed": len(routed),
                          "accuracy": round(sum(final[k] == gold[k] for k in keys) / n, 4),
                          "cost_per_item_usd": round(jev_cost + sum(rev[k]["cost_usd"] for k in routed) / n, 8),
                          "latency_per_item_s": round(jev_lat + sum(rev[k]["latency_s"] for k in routed) / n, 3)})
    write_csv("pipelines.csv", pipes)

    # Jev's calibration: accuracy by its confidence.
    calib = []
    for sample, _ in SAMPLES:
        for lo, hi in ((0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            b = [jev[k]["correct"] for k in jev if k[0] == sample and lo <= jev[k]["confidence"] < hi]
            calib.append({"sample": sample, "confidence_from": lo, "confidence_to": min(hi, 1.0), **{
                k: v for k, v in rate(b).items() if k != "ci95"}})
    write_csv("jev_calibration.csv", calib)

    # Figure 1: where wrong proposals land (composition), Jev vs matched random.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, (sample, title) in zip(axes, SAMPLES):
        rows = [r for r in landing if r["sample"] == sample]
        labels = {"jev_label": "Jev's errors", "matched_label": "Matched random\nerrors"}
        for y, r in enumerate(reversed(rows)):
            left = 0.0
            for d, _ in DIFFICULTY:
                share = r[d] / r["n_wrong"] if r["n_wrong"] else 0
                ax.barh(y, share, left=left, color=RAMP[d], height=0.55, edgecolor=SURFACE, linewidth=2)
                if share >= 0.08:
                    ax.text(left + share / 2, y, f"{r[d]}", ha="center", va="center", fontsize=9,
                            color=INK if d == "always_right" else SURFACE)
                left += share
        ax.set_yticks(range(len(rows)), [labels[r["condition"]] for r in reversed(rows)])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{title}: {rows[0]['n_wrong']} wrong proposals each", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[d]) for d, _ in DIFFICULTY]
    fig.legend(handles, [lab.replace("\n", " ") for _, lab in DIFFICULTY], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.12), title="Share of wrong proposals on items where the reviewer, deciding alone, is…")
    fig.suptitle("Jev's errors fall on items the reviewer also gets wrong", x=0.01, y=1.04, ha="left", fontweight="bold")
    save(fig, "fig1_where_errors_land.png")

    # Figure 2: acceptance of wrong proposals within each difficulty bucket.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (sample, title) in zip(axes, SAMPLES):
        for j, (cond, color, name) in enumerate((("jev_label", BLUE, "Jev's errors"),
                                                 ("matched_label", ORANGE, "Matched random errors"))):
            for i, (d, _) in enumerate(DIFFICULTY):
                r = next(x for x in accept_rows if x["sample"] == sample and x["condition"] == cond and x["difficulty"] == d)
                x = i + (j - 0.5) * 0.36
                if r["n"]:
                    ax.bar(x, r["rate"], width=0.34, color=color, label=name if i == 0 else None)
                    ax.errorbar(x, r["rate"], yerr=[[r["rate"] - r["ci_lo"]], [r["ci_hi"] - r["rate"]]], color=INK_2,
                                capsize=3, linewidth=1)
                ax.text(x, (r["ci_hi"] or 0) + 0.03, f"{r['accepted']}/{r['n']}" if r["n"] else "n=0",
                        ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(3), [lab for _, lab in DIFFICULTY], fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}" if y <= 1 else "")
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Wrong proposals accepted")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Holding the reviewer's own difficulty fixed closes most of the gap (Wilson 95% intervals)",
                 x=0.01, y=1.03, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig2_acceptance_by_difficulty.png")

    # Figure 3: the trade-off, per condition.
    names = {"jev_label": "Jev", "jev_conf": "Jev + conf", "matched_label": "Matched",
             "matched_conf": "Matched + conf", "flip_p10": "Flip 10%", "flip_p20": "Flip 20%", "flip_p35": "Flip 35%"}
    # Hand-placed so labels of nearby points don't collide (offsets in points).
    LABEL_OFFSETS = {
        ("civil_toxicity_v1", "matched_label"): (-8, 10, "right"), ("civil_toxicity_v1", "matched_conf"): (4, 14, "left"),
        ("civil_toxicity_v1", "flip_p35"): (-8, -14, "right"), ("civil_toxicity_v1", "flip_p20"): (8, -2, "left"),
        ("civil_toxicity_v1", "flip_p10"): (8, -2, "left"), ("civil_toxicity_v1", "jev_label"): (-8, -14, "right"),
        ("civil_toxicity_v1", "jev_conf"): (10, 6, "left"),
        ("wildguard_prompt_v1", "matched_conf"): (-10, 6, "right"), ("wildguard_prompt_v1", "matched_label"): (-30, -6, "right"),
        ("wildguard_prompt_v1", "flip_p10"): (18, 18, "left"), ("wildguard_prompt_v1", "flip_p35"): (30, 6, "left"),
        ("wildguard_prompt_v1", "flip_p20"): (10, -12, "left"), ("wildguard_prompt_v1", "jev_label"): (-10, 8, "right"),
        ("wildguard_prompt_v1", "jev_conf"): (-10, -10, "right"),
    }
    color_of = lambda c: BLUE if c.startswith("jev") else ORANGE if c.startswith("matched") else MUTED
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (sample, title) in zip(axes, SAMPLES):
        for r in (x for x in trade if x["sample"] == sample and x["condition"] != "alone"):
            ax.scatter(r["accept_when_right"], r["override_when_wrong"], s=60, color=color_of(r["condition"]),
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            dx, dy, ha = LABEL_OFFSETS.get((sample, r["condition"]), (6, 4, "left"))
            ax.annotate(names[r["condition"]], (r["accept_when_right"], r["override_when_wrong"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8.5, color=INK_2, ha=ha,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if abs(dx) + abs(dy) > 14 else None)
        ax.set_xlim(0.6, 1.0)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.set_xlabel("Accepts the proposal when it is right")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Overrides the proposal when it is wrong")
    fig.suptitle("Against Jev the reviewer accepts more and overrides less: the same items are hard for both",
                 x=0.01, y=1.02, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig3_tradeoff.png")

    # Figure 4: reasoning tokens per condition.
    order = ["alone", "jev_label", "jev_conf", "matched_label", "matched_conf", "flip_p10", "flip_p20", "flip_p35"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, ((sample, title), color) in enumerate(zip(SAMPLES, (BLUE, ORANGE))):
        vals = [next(x["mean_output_tokens"] for x in trade if x["sample"] == sample and x["condition"] == c) for c in order]
        xs = [i + (j - 0.5) * 0.36 for i in range(len(order))]
        ax.bar(xs, vals, width=0.34, color=color, label=title)
        for x, v in zip(xs, vals):
            ax.text(x, v + 8, f"{v:.0f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(order)), ["Alone"] + [names[c] for c in order[1:]], fontsize=9)
    ax.set_ylabel("Mean output tokens per call\n(almost all reasoning)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 560)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Reviewing a proposal takes more reasoning than deciding alone")
    save(fig, "fig4_tokens.png")

    # Figure 5: accuracy vs cost per item.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (sample, title) in zip(axes, SAMPLES):
        rows = [r for r in pipes if r["sample"] == sample]
        routed = [r for r in rows if r["threshold"] is not None]
        ax.plot([r["cost_per_item_usd"] * 1e6 for r in routed], [r["accuracy"] for r in routed], color=BLUE,
                linewidth=2, marker="o", markersize=5, label="Jev, then review below a confidence")
        for r in routed:
            if r["threshold"] in (0.8, 0.95, 1.01):
                ax.annotate(f"< {min(r['threshold'], 1):.2g}" if r["threshold"] < 1 else "review all",
                            (r["cost_per_item_usd"] * 1e6, r["accuracy"]), xytext=(5, 6),
                            textcoords="offset points", fontsize=8, color=INK_2)
        for r, color in ((rows[0], INK), (rows[1], ORANGE)):
            ax.scatter(r["cost_per_item_usd"] * 1e6, r["accuracy"], s=70, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=2, label=r["pipeline"])
        ax.set_xlabel("Cost per item (millionths of a dollar)")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1%}")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("No cascade beats the better of its two models alone, at any review threshold", x=0.01, y=1.03,
                 ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig5_pipelines.png")
    print("wrote", sorted(p.name for p in (HERE / "data").iterdir()), sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
