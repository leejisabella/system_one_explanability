"""E.102 figures and data, regenerated from the saved run.

Usage:  uv run --group analysis python research_analyses/E102/make_figures.py

Reads results/runs/e102-v1-2026-11-04/ (local only), re-runs the E.102 analysis, writes derived
tables to data/ (committed, so the numbers survive if results/ is lost) and the figures to
figures/. No API calls.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysone.analysis.e101 import rate  # noqa: E402
from sysone.analysis.e102 import analyze_run  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "e102-v1-2026-11-04"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN
SAMPLES = [("civil_toxicity_pool_v1", "Civil Comments"), ("wildguard_prompt_full_v1", "WildGuard prompts"),
           ("wildguard_response_full_v1", "WildGuard responses")]

# Reference palette (dataviz skill, light mode), as in E0, E1 and E.101. Validated with the skill's
# validate_palette.js: blue/orange/aqua categorical (aqua is under 3:1 contrast, so every aqua mark
# carries a direct label), and the 4-step blue ordinal ramp.
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, ORANGE, AQUA, NAVY = "#2a78d6", "#eb6834", "#1baf7a", "#104281"
LEVELS = [("always_right", "Right alone 5 of 5", "#86b6ef"), ("mostly_right", "3–4 of 5", "#3987e5"),
          ("sometimes_right", "1–2 of 5", "#1c5cab"), ("never_right", "0 of 5", "#0d366b")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
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
    reviews = [r for r in read("reviews") if r["status"] == "ok" and r["parse_ok"]]
    S = summary["samples"]

    # Deference estimates: raw flip-rate difference and difficulty-matched, per sample and pooled.
    est = []
    for sample, label in SAMPLES:
        raw = S[sample]["contrast"]["flip_rate_diff"]
        est.append({"sample": sample, "estimate": "raw flip-rate difference", "diff": raw["diff"],
                    "ci_lo": raw["ci95"][0], "ci_hi": raw["ci95"][1], "n_jev": raw["n_a"]})
        for name in ("eligible", "strict", "all_wrong_proposals"):
            d = S[sample]["difficulty_matched"][name]
            est.append({"sample": sample, "estimate": f"difficulty-matched, {name}", "diff": d["diff"],
                        "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1], "n_jev": d["n_jev"]})
    for name in ("eligible", "strict", "all_wrong_proposals"):
        d = summary["pooled_difficulty_matched"][name]
        est.append({"sample": "pooled", "estimate": f"difficulty-matched, {name}", "diff": d["diff"],
                    "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1], "n_jev": d["n_jev"]})
    write_csv("deference_estimates.csv", est)

    # Acceptance of wrong proposals at each exact difficulty (k of 5 right alone).
    k, n = Counter(), Counter()
    for r in reviews:
        if r.get("role") == "reviewer" and r["condition"] == "alone":
            n[(r["sample"], r["item_id"])] += 1
            k[(r["sample"], r["item_id"])] += r["final_label"] == r["label"]
    by_k = []
    for sample, _ in SAMPLES:
        for cond in ("jev_label", "matched_label"):
            rows = [r for r in reviews if r["sample"] == sample and r["condition"] == cond and n[(sample, r["item_id"])] == 5]
            for kk in range(6):
                rr = rate([r["final_label"] == r["proposed_label"] for r in rows if k[(sample, r["item_id"])] == kk])
                by_k.append({"sample": sample, "arm": cond, "k_right_alone": kk, "accepted": rr["k"], "n": rr["n"],
                             "rate": rr["rate"], "ci_lo": (rr["ci95"] or [None, None])[0],
                             "ci_hi": (rr["ci95"] or [None, None])[1]})
    write_csv("acceptance_by_k.csv", by_k)

    # Where wrong proposals land, by the reviewer's difficulty.
    comp = []
    for sample, _ in SAMPLES:
        for cond in ("jev_label", "matched_label"):
            lv = S[sample]["accept_wrong_by_difficulty"][cond]
            comp.append({"sample": sample, "arm": cond, **{d: lv[d]["n"] for d, _, _ in LEVELS}})
    write_csv("wrong_proposals_by_difficulty.csv", comp)

    # Error correlation.
    phis = []
    for sample, _ in SAMPLES:
        for pair, e in S[sample]["overlap"]["pairs"].items():
            phis.append({"sample": sample, "pair": pair, "both_wrong": e["both_wrong"],
                         "both_wrong_if_independent": e["both_wrong_if_independent"], "phi": e["phi"],
                         "ci_lo": e["phi_ci95"][0], "ci_hi": e["phi_ci95"][1],
                         "phi_high_agreement": e["phi_agreement_high"], "phi_low_agreement": e["phi_agreement_low"]})
    write_csv("error_correlation.csv", phis)
    diffs = [{"sample": sample, "difference": name, "phi_diff": v["phi_diff"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
             for sample, _ in SAMPLES for name, v in S[sample]["overlap"]["pair_differences"].items()]
    write_csv("error_correlation_differences.csv", diffs)

    # Pipelines, harmful let through, Jev by slice, headline arms.
    pipes = [{"sample": sample, "pipeline": name, "accuracy": v["accuracy"]["rate"], "k": v["accuracy"]["k"],
              "n": v["accuracy"]["n"], "cost_per_item_usd": v["cost_per_item_usd"], "latency_s": v["latency_s"]}
             for sample, _ in SAMPLES for name, v in S[sample]["pipelines"].items() if isinstance(v, dict)]
    write_csv("pipelines.csv", pipes)
    harm = [{"sample": sample, "arm": arm, "proposed_unharmful": v["proposed_unharmful"], "let_through": v["let_through"]["k"],
             "rate": v["let_through"]["rate"]}
            for sample, _ in SAMPLES[1:] for arm, v in S[sample]["harmful_let_through"].items()]
    write_csv("harmful_let_through.csv", harm)
    arms = [{"sample": sample, "arm": cond, "flip_rate": a["flip_rate"]["rate"], "flip_k": a["flip_rate"]["k"],
             "flip_n": a["flip_rate"]["n"], "flip_strict": a["flip_rate_strict"]["rate"],
             "override_when_wrong": a["override_when_wrong"]["rate"],
             "accept_when_right": S[sample]["accept_when_right"][cond]["rate"],
             "mean_output_tokens": a["mean_output_tokens"]}
            for sample, _ in SAMPLES for cond, a in S[sample]["arms"].items()]
    write_csv("arms.csv", arms)
    jev = [{"sample": sample, "slice": sl, "errors": v["k"], "n": v["n"], "rate": v["rate"]}
           for sample, _ in SAMPLES for sl, v in {"all": S[sample]["jev"]["error_rate"],
                                                  **S[sample]["jev"]["error_rate_by_slice"]}.items()]
    write_csv("jev_errors.csv", jev)

    # Figure 1: the deference estimates, raw vs difficulty-matched.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rows = []
    for sample, label in SAMPLES:
        rows.append((f"{label}: raw flip-rate difference", next(e for e in est if e["sample"] == sample and e["estimate"].startswith("raw")), MUTED))
        rows.append((f"{label}: same difficulty", next(e for e in est if e["sample"] == sample and e["estimate"] == "difficulty-matched, eligible"), BLUE))
    rows.append(("All three pooled: same difficulty", next(e for e in est if e["sample"] == "pooled" and e["estimate"] == "difficulty-matched, eligible"), NAVY))
    for y, (name, e, color) in enumerate(reversed(rows)):
        ax.errorbar(e["diff"] * 100, y, xerr=[[(e["diff"] - e["ci_lo"]) * 100], [(e["ci_hi"] - e["diff"]) * 100]],
                    fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, markersize=7)
        ax.text(e["ci_hi"] * 100 + 1.2, y, f"{e['diff'] * 100:+.0f}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlabel("Extra share of wrong labels accepted when they come from Jev, in percentage points (95% interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Compared at the same difficulty, Jev's errors are still accepted more, but by much less")
    save(fig, "fig1_deference_estimates.png")

    # Figure 2: acceptance by exact difficulty.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (sample, label) in zip(axes, SAMPLES):
        for cond, color, name in (("jev_label", BLUE, "Jev's wrong labels"), ("matched_label", ORANGE, "Random wrong labels")):
            pts = [r for r in by_k if r["sample"] == sample and r["arm"] == cond and r["n"]]
            ax.plot([r["k_right_alone"] for r in pts], [r["rate"] for r in pts], color=color, linewidth=2, marker="o",
                    markersize=5, label=name)
            for r in pts:
                ax.annotate(str(r["n"]), (r["k_right_alone"], r["rate"]), xytext=(0, 7 if cond == "jev_label" else -13),
                            textcoords="offset points", ha="center", fontsize=7.5, color=color)
        ax.set_xticks(range(6))
        ax.set_ylim(-0.08, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.set_title(label)
    axes[0].set_ylabel("Wrong labels the reviewer accepted")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.supxlabel("Times the reviewer got the item right on its own (of 5): harder to the left", fontsize=10, color=INK_2, y=-0.02)
    fig.suptitle("At most difficulty levels the Jev line sits above the random line (small numbers: items per point)",
                 x=0.01, y=1.03, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig2_acceptance_by_difficulty.png")

    # Figure 3: where wrong proposals land.
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), sharey=True)
    for ax, (sample, label) in zip(axes, SAMPLES):
        for y, cond in enumerate(("matched_label", "jev_label")):
            row = next(c for c in comp if c["sample"] == sample and c["arm"] == cond)
            total, left = sum(row[d] for d, _, _ in LEVELS), 0.0
            for d, _, color in LEVELS:
                share = row[d] / total
                ax.barh(y, share, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
                if share >= 0.1:
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                            color=INK if d == "always_right" else SURFACE)
                left += share
        ax.set_yticks([0, 1], ["Random errors", "Jev's errors"])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.grid(axis="y", visible=False)
        ax.set_title(label)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LEVELS]
    fig.legend(handles, [name for _, name, _ in LEVELS], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.14),
               title="How often the reviewer gets the item right on its own")
    fig.suptitle("Jev's errors sit on items the reviewer finds hard; random errors mostly on easy ones",
                 x=0.01, y=1.06, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig3_where_errors_land.png")

    # Figure 4: error correlation between every pair of models.
    pairs = [("jev|reviewer", "Jev & gpt-oss (reviewer)", BLUE), ("jev|reviewer_qwen", "Jev & Qwen (reference)", ORANGE),
             ("reviewer|reviewer_qwen", "gpt-oss & Qwen", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, (pair, name, color) in enumerate(pairs):
        for i, (sample, _) in enumerate(SAMPLES):
            p = next(x for x in phis if x["sample"] == sample and x["pair"] == pair)
            x = i + (j - 1) * 0.27
            ax.bar(x, p["phi"], width=0.25, color=color, label=name if i == 0 else None)
            ax.errorbar(x, p["phi"], yerr=[[p["phi"] - p["ci_lo"]], [p["ci_hi"] - p["phi"]]], color=INK_2, capsize=2, linewidth=1)
            ax.text(x, p["ci_hi"] + 0.02, f"{p['phi']:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(3), [label for _, label in SAMPLES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Error correlation (phi)\n0 = independent, 1 = same errors")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
    ax.set_title("Every pair of models makes many of the same mistakes; pairs with Jev share the most")
    save(fig, "fig4_error_correlation.png")

    # Figure 5: harmful content let through.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (sample, label) in enumerate(SAMPLES[1:]):
        for j, (arm, color, name) in enumerate((("jev_label", BLUE, "Jev missed it"), ("matched_label", ORANGE, "Random error missed it"))):
            h = next(x for x in harm if x["sample"] == sample and x["arm"] == arm)
            x = i + (j - 0.5) * 0.36
            ax.bar(x, h["rate"], width=0.34, color=color, label=name if i == 0 else None)
            ax.text(x, h["rate"] + 0.02, f"{h['let_through']} of {h['proposed_unharmful']}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(2), [label for _, label in SAMPLES[1:]])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_ylabel("Harmful items the reviewer\nalso let through")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Harmful content Jev misses mostly gets past the reviewer too")
    save(fig, "fig5_harmful_let_through.png")
    print("wrote", sorted(p.name for p in (HERE / "data").iterdir()), sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
