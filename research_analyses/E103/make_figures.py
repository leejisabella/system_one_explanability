"""E.103 figures and data, regenerated from the saved runs.

Usage:  uv run --group analysis python research_analyses/E103/make_figures.py

Reads results/runs/e103-v1-2026-11-04/ and its source run e102-v1-2026-11-04/ (both local only),
re-runs the E.103 analysis, writes derived tables to data/ (committed, so the numbers survive if
results/ is lost) and the figures to figures/. No API calls.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysone.analysis.e101 import rate  # noqa: E402
from sysone.analysis.e103 import analyze_run  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN, SOURCE = "e103-v1-2026-11-04", "e102-v1-2026-11-04"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN
SOURCE_DIR = REPO_ROOT / "results" / "runs" / SOURCE
SAMPLES = [("civil_toxicity_pool_v1", "Civil Comments"), ("wildguard_prompt_full_v1", "WildGuard prompts"),
           ("wildguard_response_full_v1", "WildGuard responses")]
PROTOCOLS = [("standard", "Standard (E.102)"), ("blind_first", "Blind-first"), ("warning", "Warning"),
             ("calibrated_warning", "Calibrated warning"), ("consider_opposite", "Consider the opposite")]
NAME = dict(PROTOCOLS)
UPSTREAMS = [("explainer_open", "Llama 4 Maverick"), ("reviewer_qwen", "Qwen3-32B"), ("decision_alt", "Nova Lite")]

# Reference palette (dataviz skill, light mode), as in E0, E1, E.101 and E.102. Protocols are not
# given categorical colors: the standard prompt is gray and every alternative blue, with direct labels.
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, ORANGE, NAVY = "#2a78d6", "#eb6834", "#104281"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,
})


def read(run_dir: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (run_dir / f"{name}.jsonl").read_text().splitlines() if line.strip()]


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
    summary = analyze_run(RUN_DIR, SOURCE_DIR)
    for name in ("summary.json", "report.md"):
        shutil.copy(RUN_DIR / name, HERE / "data" / name)
    reviews = [r for r in read(RUN_DIR, "reviews") if r["status"] == "ok"]
    upstream = [r for r in read(RUN_DIR, "upstream") if r["status"] == "ok"]
    S = summary["samples"]

    # Deference at equal difficulty, per protocol, per sample and pooled.
    dfr = []
    for proto, _ in PROTOCOLS:
        for sample, _ in [*SAMPLES, ("pooled", "")]:
            src = summary["pooled"][proto] if sample == "pooled" else S[sample]["protocols"][proto]["deference"]
            for measure in ("eligible", "strict", "all_wrong_proposals"):
                d = src[measure]
                dfr.append({"protocol": proto, "sample": sample, "measure": measure, "diff": d["diff"],
                            "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1], "jev_rate": d["jev_rate"],
                            "matched_at_jev_difficulty": d["matched_at_jev_difficulty"], "n_jev": d["n_jev"]})
    write_csv("deference_by_protocol.csv", dfr)

    # Per-protocol rates, and paired changes against the standard prompt.
    cost = defaultdict(float)
    for r in reviews:
        cost[(r["sample"], r["protocol"])] += r["cost_usd"]
    prot, vs = [], []
    for sample, _ in SAMPLES:
        for proto, _ in PROTOCOLS:
            P = S[sample]["protocols"][proto]
            row = {"sample": sample, "protocol": proto}
            for key in ("override_when_wrong_jev", "override_when_wrong_matched", "accept_when_right_jev",
                        "jev_wrong_accepted_never_right", "cascade_accuracy_subsample", "harmful_let_through"):
                v = P[key] or {}
                row.update({key: v.get("rate"), f"{key}_k": v.get("k"), f"{key}_n": v.get("n")})
            row["mean_output_tokens"] = P["mean_output_tokens"]
            row["cost_per_review_usd"] = (round(cost[(sample, proto)] / P["n_reviews"], 7)
                                          if proto != "standard" else None)
            prot.append(row)
            for measure, d in P.get("vs_standard", {}).items():
                vs.append({"sample": sample, "protocol": proto, "measure": measure, "diff": d["diff"],
                           "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1], "n": d["n"]})
    write_csv("protocols.csv", prot)
    write_csv("vs_standard.csv", vs)

    # Blind-first: how often the final answer keeps the reviewer's committed first answer.
    own = []
    for sample, _ in SAMPLES:
        for cond in ("jev_label", "matched_label", "correct_label"):
            rows = [r for r in reviews if r["protocol"] == "blind_first" and r["parse_ok"]
                    and r["sample"] == sample and r["condition"] == cond]
            for first, sub in (("agrees", [r for r in rows if r["own_answer"] == r["proposed_label"]]),
                               ("disagrees", [r for r in rows if r["own_answer"] != r["proposed_label"]])):
                kept = rate([r["final_label"] == r["own_answer"] for r in sub])
                own.append({"sample": sample, "proposal": cond, "first_answer_vs_proposal": first,
                            "kept_first_answer": kept["k"], "n": kept["n"], "rate": kept["rate"]})
    write_csv("blind_first_own_answer.csv", own)

    # Pipelines needing no new calls, and the upstream models.
    pipes = [{"sample": sample, "pipeline": name, "accuracy": v["rate"], "k": v["k"], "n": v["n"],
              "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
             for sample, _ in SAMPLES for name, v in S[sample]["no_call_pipelines"].items()]
    write_csv("no_call_pipelines.csv", pipes)
    up_cost = defaultdict(float)
    for r in upstream:
        up_cost[(r["sample"], r["reviewer"])] += r.get("cost_usd") or 0.0
    ups = [{"sample": sample, "model": m, "error_rate": u["error_rate"]["rate"], "errors": u["error_rate"]["k"],
            "n": u["error_rate"]["n"], "parse_rate": u["parse_rate"]["rate"], "phi_with_jev": u["phi_with_jev"],
            "phi_with_reviewer_subsample": u["phi_with_reviewer_subsample"],
            **{f"error_rate_true_{lab}": v["rate"] for lab, v in u["error_by_true_label"].items()},
            "cost_usd": round(up_cost[(sample, m)], 4)}
           for sample, _ in SAMPLES for m, u in S[sample]["upstream"].items()]
    write_csv("upstream.csv", [{k: row.get(k) for k in dict.fromkeys(k for r in ups for k in r)} for row in ups])

    # Figure 1: pooled deference at equal difficulty, per protocol.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    measures = (("eligible", "Wrong labels on items the reviewer gets right alone", BLUE, 0.17),
                ("all_wrong_proposals", "All wrong labels", ORANGE, -0.17))
    for y, (proto, label) in enumerate(reversed(PROTOCOLS)):
        for measure, name, color, dy in measures:
            d = next(e for e in dfr if e["protocol"] == proto and e["sample"] == "pooled" and e["measure"] == measure)
            ax.errorbar(d["diff"] * 100, y + dy, xerr=[[(d["diff"] - d["ci_lo"]) * 100], [(d["ci_hi"] - d["diff"]) * 100]],
                        fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, markersize=6,
                        label=name if y == 0 else None)
            ax.text(d["ci_hi"] * 100 + 0.6, y + dy, f"{d['diff'] * 100:+.1f}", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(PROTOCOLS)), [label for _, label in reversed(PROTOCOLS)])
    ax.set_xlabel("Extra share of wrong labels accepted when they come from Jev, at equal difficulty\n"
                  "(percentage points, all three pools pooled, 95% interval)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    ax.set_title("Only blind-first shrinks the pull toward Jev; the calibrated warning increases it")
    save(fig, "fig1_deference_by_protocol.png")

    # Figure 2: where the gap goes. Jev's wrong labels accepted vs. random ones at Jev's difficulty.
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for y, (proto, label) in enumerate(reversed(PROTOCOLS)):
        d = next(e for e in dfr if e["protocol"] == proto and e["sample"] == "pooled" and e["measure"] == "all_wrong_proposals")
        j, m = d["jev_rate"], d["matched_at_jev_difficulty"]
        ax.plot([m, j], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter([j], [y], color=BLUE, s=60, zorder=2, label="Jev's wrong labels" if y == 0 else None)
        ax.scatter([m], [y], color=ORANGE, s=60, zorder=2, label="Random wrong labels, at the same difficulty" if y == 0 else None)
        ax.text(j + 0.006, y + 0.22, f"{j:.0%}", ha="center", fontsize=8.5, color=BLUE)
        ax.text(m - 0.006, y + 0.22, f"{m:.0%}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(PROTOCOLS)), [label for _, label in reversed(PROTOCOLS)])
    ax.set_xlim(0.45, 0.72)
    ax.set_ylim(-0.6, len(PROTOCOLS) - 0.3)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Wrong labels the reviewer accepted (all three pools)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    ax.set_title("Blind-first closes the gap from the random side: Jev's mistakes get through as often as before")
    save(fig, "fig2_where_the_gap_goes.png")

    # Figure 3: the trade-off, per pool. Jev's wrong labels overridden vs. Jev's right labels accepted.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (sample, label) in zip(axes, SAMPLES):
        pts = [r for r in prot if r["sample"] == sample]
        for r in pts:
            std = r["protocol"] == "standard"
            ax.scatter(r["override_when_wrong_jev"], r["accept_when_right_jev"], s=55, zorder=2,
                       color=MUTED if std else BLUE, edgecolor=SURFACE, linewidth=1.5)
            # Blind-first and consider-the-opposite sit side by side on Civil Comments.
            above = sample == "civil_toxicity_pool_v1" and r["protocol"] == "blind_first"
            ax.annotate(NAME[r["protocol"]], (r["override_when_wrong_jev"], r["accept_when_right_jev"]),
                        xytext=(0, 7) if above else (6, -3), ha="center" if above else "left",
                        textcoords="offset points", fontsize=7.5, color=INK_2)
        xs = [r["override_when_wrong_jev"] for r in pts]
        ys = [r["accept_when_right_jev"] for r in pts]
        ax.set_xlim(min(xs) - 0.03, max(xs) + 0.1)
        ax.set_ylim(min(ys) - 0.03, max(ys) + 0.02)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.set_title(label)
    axes[0].set_ylabel("Jev's right labels accepted")
    fig.supxlabel("Jev's wrong labels overridden", fontsize=10, color=INK_2, y=-0.02)
    fig.suptitle("Every protocol lands within a few points of the standard prompt (gray); "
                 "up and to the right would be better", x=0.01, y=1.03, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig3_tradeoff.png")

    # Figure 4: harmful content Jev missed that the reviewer also let through, per protocol.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (sample, label) in zip(axes, SAMPLES[1:]):
        for y, (proto, _) in enumerate(reversed(PROTOCOLS)):
            r = next(x for x in prot if x["sample"] == sample and x["protocol"] == proto)
            ax.barh(y, r["harmful_let_through"], height=0.6, color=MUTED if proto == "standard" else BLUE)
            ax.text(r["harmful_let_through"] + 0.01, y, f"{r['harmful_let_through_k']} of {r['harmful_let_through_n']}",
                    va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(PROTOCOLS)), [lab for _, lab in reversed(PROTOCOLS)])
        ax.set_xlim(0, 0.85)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.grid(axis="y", visible=False)
        ax.set_title(label)
    fig.supxlabel("Harmful items Jev called harmless that the reviewer also let through", fontsize=10, color=INK_2, y=-0.03)
    fig.suptitle("Every protocol still lets about half or more of Jev's harmful misses through",
                 x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
    save(fig, "fig4_harmful_let_through.png")
    print("wrote", sorted(p.name for p in (HERE / "data").iterdir()), sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
