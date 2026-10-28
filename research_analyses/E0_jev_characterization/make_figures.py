"""E0 (Pilot 0) figures and data, regenerated from the saved runs.

Usage:  uv run --group analysis python research_analyses/E0_jev_characterization/make_figures.py

Reads results/runs/<run>/ (local only) for the two Pilot 0 runs, writes per-variant
tables and the runs' summaries to data/ (committed, so the numbers survive if results/
is lost), and the figures to figures/. No API calls.
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
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from sysone.analysis.pilot0 import load_decisions, majority  # noqa: E402
from sysone.benchmark.records import Condition  # noqa: E402
from sysone.benchmark.refund_policy import PolicyParams, counted_days, evaluate  # noqa: E402
from sysone.benchmark.store import load_benchmark  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = {"refund_v0": "pilot0-v0-2026-10-28", "refund_v1": "pilot0-v1-2026-10-28"}

# Reference palette (dataviz skill, light mode), validated for these three slots.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def variant_table(name: str, run_id: str) -> list[dict]:
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    cases, meta = load_benchmark(REPO_ROOT / "benchmarks" / name)
    params = PolicyParams(**meta["policy"])
    samples = defaultdict(list)
    for r in load_decisions(run_dir):
        if r["parse_ok"]:
            samples[r["variant_id"]].append(r)
    rows = []
    for c in cases:
        for v in c.variants:
            rs = samples[v.variant_id]
            ps = [r["p_approve"] for r in rs]
            rec = v.record
            days = counted_days(rec) if rec.purchase_date else None
            rows.append({
                "benchmark": name, "run_id": run_id, "case_id": c.case_id, "case_type": str(c.case_type),
                "variant_id": v.variant_id, "variant_kind": str(v.kind),
                "policy_outcome": str(v.policy_outcome), "true_outcome": str(c.true_outcome),
                "variant_rule": evaluate(rec, params).rule,
                "counted_days": days,
                "window_days": params.defect_window_days if rec.condition == Condition.DEFECTIVE else params.window_days,
                "jev_majority": majority([r["decision"] for r in rs]),
                "mean_p_approve": round(sum(ps) / len(ps), 4) if ps else None,
                "p_approve_range": round(max(ps) - min(ps), 4) if ps else None,
                "n_samples": len(rs),
                "unanimous": len({r["decision"] for r in rs}) == 1 if rs else None,
            })
    return rows


def save_data(tables: dict[str, list[dict]]) -> None:
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    for name, rows in tables.items():
        with (data / f"variants_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        run_dir = REPO_ROOT / "results" / "runs" / RUNS[name]
        shutil.copy(run_dir / "summary.json", data / f"summary_{name}.json")
        shutil.copy(run_dir / "report.md", data / f"report_{name}.md")


def label_bars(ax, bars, texts, *, inside_min=0.25):
    for bar, text in zip(bars, texts):
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, text, va="center", ha="left", color=INK, fontsize=9)


def fig_window(tables, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = [r for t in tables.values() for r in t
            if r["variant_rule"].startswith("R5_") and r["counted_days"] is not None and r["window_days"] == 30
            and r["mean_p_approve"] is not None]
    for outcome, color, label in (("approve", BLUE, "policy says approve"), ("deny", ORANGE, "policy says deny")):
        pts = [r for r in rows if r["policy_outcome"] == outcome]
        ax.scatter([r["counted_days"] for r in pts], [r["mean_p_approve"] for r in pts], s=40, color=color,
                   edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    ax.axvline(30, color=INK_2, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(30.6, 1.03, "30-day window", color=INK_2, fontsize=9, va="bottom")
    ax.axhline(0.5, color=GRID, linewidth=1.2, zorder=1)
    outside = [r for r in rows if r["counted_days"] > 30]
    wrong = [r for r in outside if r["jev_majority"] == "approve"]
    ax.text(55, 0.92, f"{len(wrong)} of {len(outside)} variants past the window\nstill approved by majority\n"
            "(points above the 0.5 line)", color=INK, fontsize=9, va="top")
    ax.set_xlabel("Days from purchase to request, net of documented carrier delay")
    ax.set_ylabel("Jev's mean P(approve)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Jev approves many requests past the 30-day return window")
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, 0.01, "Both Pilot 0 runs; variants decided by the purchase window, non-defective items. "
             "Each point is one variant, mean of 3 samples.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"outside": len(outside), "outside_approved": len(wrong),
            "inside": len(rows) - len(outside),
            "inside_denied": sum(r["counted_days"] <= 30 and r["jev_majority"] == "deny" for r in rows)}


def fig_rules(summary, out):
    cells = summary["P0.2_policy_agreement_by_rule"]
    names = {
        "R1_manager_preapproval": "Manager pre-approval", "R2_final_sale": "Final sale",
        "R3_no_receipt": "No receipt", "R3_receipt_unknown": "Receipt status missing",
        "R4_condition_unknown": "Condition missing", "R4_opened": "Opened item",
        "R5_purchase_date_unknown": "Purchase date missing", "R5_within_window": "Within window",
        "R5_outside_window": "Outside window",
    }
    items = sorted(cells.items(), key=lambda kv: kv[1]["rate"], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = list(range(len(items)))[::-1]
    colors = [ORANGE if c["rate"] < 0.9 else BLUE for _, c in items]
    bars = ax.barh(y, [c["rate"] for _, c in items], height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [names.get(k, k) for k, _ in items])
    label_bars(ax, bars, [f"{c['rate']:.0%} ({c['k']}/{c['n']})" for _, c in items])
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Samples where Jev's decision matches the policy applied to the record it was shown")
    fig.suptitle("Jev's two weak rules: the purchase window and missing receipts", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "refund_v1 run, per sample (3 per variant). Orange: below 90%.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_contestable(summaries, out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(summaries)
    customer = [len(summaries[n]["P0.3_contestable_by_harm"].get("customer: wrongly denied", [])) for n in names]
    store = [len(summaries[n]["P0.3_contestable_by_harm"].get("store: wrongly approved", [])) for n in names]
    y = [1, 0]
    b1 = ax.barh(y, customer, height=0.5, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Customer wrongly denied")
    b2 = ax.barh(y, store, left=customer, height=0.5, color=ORANGE, edgecolor=SURFACE, linewidth=2,
                 label="Store wrongly approved")
    for bar, n in zip(b1, customer):
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(n), ha="center", va="center",
                color=SURFACE, fontsize=10, fontweight="bold")
    for bar, n, left in zip(b2, store, customer):
        ax.text(left + n + 0.4, bar.get_y() + bar.get_height() / 2, f"+{n}", ha="left", va="center", color=INK, fontsize=9)
    ax.axvline(20, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(20.3, 1.42, "go threshold: 20 customer-harming", color=INK_2, fontsize=9)
    ax.set_yticks(y, [f"{n}\n({summaries[n]['n_cases']} cases)" for n in names])
    ax.set_xlim(0, 32)
    ax.set_ylim(-0.5, 1.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cases where Jev's decision on the original is wrong under ground truth")
    ax.set_title("refund_v1 clears the bar for contestable cases")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_flips(summary, out):
    rows = summary["P0.5_6_flips"]
    names = {"correction": "Corrupted field corrected", "decisive_edit": "Deciding fact changed",
             "exception_removed": "Exception removed", "resolution": "Missing fact filled in",
             "cue_edit": "Name or register swapped", "discrepancy_fixed": "Irrelevant error fixed",
             "exception_salient": "Exception moved to a field"}
    groups = [("policy flips", BLUE, "Policy outcome flips: should flip"),
              ("original undetermined", AQUA, "Original undetermined: should change"),
              ("policy unchanged", ORANGE, "Policy outcome unchanged: should not flip")]
    order = [r for g, _, _ in groups for r in rows if r["expectation"] == g]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = list(range(len(order)))[::-1]
    color = {g: c for g, c, _ in groups}
    bars = ax.barh(y, [r["rate"] for r in order], height=0.55, color=[color[r["expectation"]] for r in order],
                   edgecolor=SURFACE, linewidth=2)
    for bar, r in zip(bars, order):
        ax.text(max(bar.get_width(), 0) + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r['rate']:.0%} ({r['k']}/{r['n']})", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, [names.get(r["variant_kind"], r["variant_kind"]) for r in order])
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in groups]
    ax.legend(handles, [l for _, _, l in groups], frameon=False, loc="lower right", fontsize=8.5)
    ax.set_xlabel("Variants where Jev's majority decision differs from its decision on the original")
    fig.suptitle("Jev follows decisive facts and ignores names and irrelevant errors", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "refund_v1 run. Every missed flip traces to the purchase window: the edited record is "
             "past it and Jev still approves.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_noise(table, summary, out):
    ranges = [r["p_approve_range"] for r in table if r["p_approve_range"] is not None]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = [i / 100 for i in range(0, 16)]
    ax.hist(ranges, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    p95 = summary["P0.4_probability_noise"]["p95_range_p_approve"]
    ax.axvline(p95, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(p95 + 0.002, ax.get_ylim()[1] * 0.92, f"95th percentile: {p95:.2f}\n(noise floor)", color=INK_2, fontsize=9)
    ax.set_xlabel("Range of P(approve) across 3 repeats of the same variant")
    ax.set_ylabel("Variants")
    ax.grid(axis="x", visible=False)
    st = summary["P0.4_stability"]
    ax.set_title(f"Repeats barely move: {st['unanimous_share']:.0%} of variants unanimous")
    fig.text(0.01, 0.01, "refund_v1 run, 188 variants. Repeats rotate the option order, "
             "so this is an upper bound on pure noise.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_setup(out):
    findings = [
        ("24 cases: Jev wrongly denies,\nfollowing a corrupted record", BLUE),
        ("Corrections flip all 24\n(mean shift in P(approve) 0.99)", BLUE),
        ("Noise floor: P(approve) range\n≤ 0.07 for 95% of variants", BLUE),
        ("Window errors: only 51% of\noutside-window samples denied", ORANGE),
        ("Missing receipt: 'undetermined'\nin only 58% of samples", ORANGE),
        ("No effect of names, register,\nor option order", AQUA),
        ("564 saved decisions with full\nprobabilities; $0.000028 each", AQUA),
    ]
    uses = [
        "E1 / Pilot 1: faithfulness vs.\ncontestability (reuses decisions)",
        "E2–E4: similar targets, information\nablation, CAPE under a budget",
        "E7: socially sensitive cues\n(needs Dataset C)",
        "E5: identifiability: record errors\nvs. policy misapplication",
        "E6: robustness and abstention",
        "Stretch S1–S3: harmful\njustification of wrong denials",
    ]
    links = {0: [0, 1, 5], 1: [0, 1], 2: [0, 1, 2], 3: [1, 3, 4], 4: [4], 5: [2], 6: [0, 1, 5]}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.4)
    ax.text(0.2, 7.1, "What E0 found", fontsize=11, fontweight="bold", color=INK)
    ax.text(7.0, 7.1, "What it sets up", fontsize=11, fontweight="bold", color=INK)
    fy = [6.4 - i * 0.92 for i in range(len(findings))]
    uy = [6.3 - i * 1.07 for i in range(len(uses))]
    for i, (text, color) in enumerate(findings):
        ax.add_patch(FancyBboxPatch((0.2, fy[i] - 0.36), 3.9, 0.72, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=SURFACE, edgecolor=color, linewidth=2))
        ax.text(0.38, fy[i], text, va="center", fontsize=8.8, color=INK)
    for j, text in enumerate(uses):
        ax.add_patch(FancyBboxPatch((7.0, uy[j] - 0.4), 3.8, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#f1f0ec", edgecolor=GRID, linewidth=1.2))
        ax.text(7.18, uy[j], text, va="center", fontsize=8.8, color=INK)
    for i, targets in links.items():
        for j in targets:
            ax.annotate("", xy=(7.0, uy[j]), xytext=(4.1, fy[i]),
                        arrowprops={"arrowstyle": "-|>", "color": findings[i][1], "linewidth": 1.1, "alpha": 0.75,
                                    "shrinkA": 2, "shrinkB": 2, "mutation_scale": 9})
    handles = [plt.Line2D([0], [0], color=c, linewidth=3) for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["Contestable-case material", "Jev's own weaknesses", "Controls and resources"],
              frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=3, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tables = {name: variant_table(name, run) for name, run in RUNS.items()}
    save_data(tables)
    summaries = {name: json.loads((HERE / "data" / f"summary_{name}.json").read_text()) for name in RUNS}
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    window = fig_window(tables, figs / "fig1_purchase_window.png")
    fig_rules(summaries["refund_v1"], figs / "fig2_agreement_by_rule.png")
    fig_contestable(summaries, figs / "fig3_contestable_cases.png")
    fig_flips(summaries["refund_v1"], figs / "fig4_flips_by_edit.png")
    fig_noise(tables["refund_v1"], summaries["refund_v1"], figs / "fig5_repeat_noise.png")
    fig_setup(figs / "fig6_what_e0_sets_up.png")
    print("window figure counts:", window)
    print(f"wrote {figs} and {HERE / 'data'}")


if __name__ == "__main__":
    main()
