"""E1 Pilot 1 figures and data, regenerated from the saved runs.

Usage:  uv run --group analysis python research_analyses/E1_pilot1/make_figures.py

Reads results/runs/pilot1-v1b-2026-10-28/ and the Pilot 0 run it builds on (local only),
re-runs the Pilot 1 analysis with and without the simulator-reply repair, writes derived
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
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from sysone.analysis.pilot0 import load_decisions, majority  # noqa: E402
from sysone.analysis.pilot1 import _executed, _latest_ok, _read_jsonl, analyze_run, reparse_simulations  # noqa: E402
from sysone.benchmark.refund_policy import PolicyParams, counted_days  # noqa: E402
from sysone.benchmark.store import load_benchmark  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "pilot1-v1b-2026-10-28"
PILOT0 = "pilot0-v1-2026-10-28"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN

# Reference palette (dataviz skill, light mode), as in E0.
SURFACE, INK, INK_2, GRID, PANEL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,  # labels carry prices with "$"
})

GROUPS = [  # (summary key, label); the order of every per-condition figure
    ("none/decision_probabilities", "Decision + probabilities\n(no explainer)"),
    ("explainer_gemini/free_form", "Gemini, free-form"),
    ("explainer_gemini/structured", "Gemini, structured"),
    ("explainer_open/free_form", "Llama, free-form"),
    ("explainer_open/structured", "Llama, structured"),
]
SHORT = {k: v.replace("\n(no explainer)", "") for k, v in GROUPS}
SHORT["none/decision_probabilities"] = "Decision +\nprobabilities"

# Hand coding of every explanation of an original that Jev approved past the 30-day window
# (Jev's own error, from E0). Read and coded on 2026-10-28; quotes are verbatim.
JUSTIFICATION_CODES = {
    "false_premise": "States the request was within the 30-day window",
    "invented_fact": "Invents a fact (carrier delays) to fit the policy",
    "flags_conflict": "Notes the conflict; speculates (pre-approval, hidden data, counting error)",
}
JUSTIFICATIONS = [
    ("refund_v1-042", "explainer_open", "free_form", "flags_conflict",
     "The approval could be due to the item being pre-approved in writing ... although there is no explicit "
     "evidence of this in the case file."),
    ("refund_v1-056", "explainer_open", "free_form", "false_premise",
     "since it's unopened and not defective, it still falls within the 30-day limit for a standard refund."),
    ("refund_v1-063", "explainer_open", "free_form", "flags_conflict",
     "It's possible that the system considered the request to be within the allowed timeframe or made an error "
     "in counting the days."),
    ("refund_v1-042", "explainer_gemini", "free_form", "invented_fact",
     "the system likely calculated that the shipping duration or carrier delays brought the effective date "
     "within the 30-day policy window."),
    ("refund_v1-056", "explainer_gemini", "free_form", "invented_fact",
     "the system's approval suggests it likely accounted for verified shipping transit times or carrier delays"),
    ("refund_v1-063", "explainer_gemini", "free_form", "invented_fact",
     "the system likely accounted for potential carrier delays"),
    ("refund_v1-042", "explainer_open", "structured", "false_premise",
     "the request was made within 30 days of purchase."),
    ("refund_v1-056", "explainer_open", "structured", "false_premise",
     "the refund request was made within 30 days of purchase."),
    ("refund_v1-063", "explainer_open", "structured", "false_premise",
     "the refund request was made within 30 days of purchase."),
    ("refund_v1-042", "explainer_gemini", "structured", "false_premise",
     "the request was made within the 30-day window."),
    ("refund_v1-056", "explainer_gemini", "structured", "false_premise",
     "was requested within the 30-day window allowed for non-defective items."),
    ("refund_v1-063", "explainer_gemini", "structured", "false_premise",
     "the request fell within the 30-day window allowed for standard items."),
]


def load():
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    cases, meta = load_benchmark(REPO_ROOT / "benchmarks" / "refund_v1")
    variants = {v.variant_id: (c, v) for c in cases for v in c.variants}
    sims = _latest_ok(reparse_simulations(_read_jsonl(RUN_DIR / "simulations.jsonl")), "explanation_id", "variant_id")
    raw_sims = _read_jsonl(RUN_DIR / "simulations.jsonl")
    expls = _latest_ok(_read_jsonl(RUN_DIR / "explanations.jsonl"), "explanation_id")
    ledger = _read_jsonl(RUN_DIR / "ledger.jsonl")
    jev_samples = defaultdict(list)
    for r in load_decisions(REPO_ROOT / "results" / "runs" / PILOT0):
        if r["parse_ok"]:
            jev_samples[r["variant_id"]].append(r["decision"])
    return manifest, cases, meta, variants, sims, raw_sims, expls, ledger, jev_samples


def save_data(summary_repaired, summary_raw, sims, variants, meta, ledger):
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "summary.json").write_text(json.dumps(summary_repaired, indent=2, sort_keys=True) + "\n")
    (data / "summary_no_repair.json").write_text(json.dumps(summary_raw, indent=2, sort_keys=True) + "\n")
    shutil.copy(RUN_DIR / "report.md", data / "report.md")
    params = PolicyParams(**meta["policy"])
    fields = ["variant_id", "case_id", "case_type", "variant_kind", "explainer", "condition", "jev_decision",
              "policy_outcome", "predicted_decision", "parse_ok", "repaired", "covered", "basis_ok", "counted_days"]
    with (data / "simulations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sims:
            rec = variants[r["variant_id"]][1].record
            w.writerow({**{k: r.get(k) for k in fields if k != "counted_days"},
                        "counted_days": counted_days(rec) if rec.purchase_date else None})
    with (data / "justification_coding.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "explainer", "condition", "code", "code_meaning", "quote"])
        for case_id, explainer, condition, code, quote in JUSTIFICATIONS:
            w.writerow([case_id, explainer, condition, code, JUSTIFICATION_CODES[code], quote])
    calls = Counter((e["provider"], e["model_requested"], e["status"]) for e in ledger)
    with (data / "calls_by_model.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["provider", "model", "status", "calls", "cost_usd"])
        cost = Counter()
        for e in ledger:
            cost[(e["provider"], e["model_requested"], e["status"])] += e["cost_usd"]
        for k, n in sorted(calls.items()):
            w.writerow([*k, n, round(cost[k], 6)])
    return calls


def _box(ax, x, y, w, h, title, lines, edge, *, dashed=False, fill=SURFACE, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=fill,
                                edgecolor=edge, linewidth=1.8, linestyle=(0, (4, 3)) if dashed else "-"))
    ax.text(x + 0.15, y + h - 0.2, title, fontsize=9.5, fontweight="bold", va="top", color=title_color)
    ax.text(x + 0.15, y + h - 0.55, "\n".join(lines), fontsize=8, va="top", color=INK_2, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, label=None, color=INK_2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.3, "mutation_scale": 11,
                            "shrinkA": 0, "shrinkB": 0})
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, fontsize=7.5, color=INK_2, ha="center", va="bottom")


def fig_model_stack(summary, ledger, pilot0_summary, out):
    cost = summary["cost"]["by_model_usd"]
    ok = Counter(e["model_requested"] for e in ledger if e["status"] == "ok")
    throttled = Counter(e["model_requested"] for e in ledger if e["status"] == "error")
    llama, gemini, oss = ("us.meta.llama4-maverick-17b-instruct-v1:0", "gemini-3.1-flash-lite",
                          "openai.gpt-oss-120b-1:0")
    fig, ax = plt.subplots(figsize=(12.5, 7.95))
    ax.set_axis_off()
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.95)
    ax.text(0.1, 7.85, "Pilot 1 model stack: one family per role", fontsize=13, fontweight="bold", va="top")
    ax.text(0.1, 7.45, "Explainers see Jev's decision on each original; their frozen explanations go to the "
            "simulator. Jev, the explainers\nand the simulator come from different model families, so no model "
            "scores its own family's explanations.", fontsize=9, color=INK_2, va="top")
    for x, label in ((0.1, "Input"), (2.75, "Decision model"), (5.55, "Explainers"), (9.05, "Scoring")):
        ax.text(x, 6.5, label.upper(), fontsize=8, fontweight="bold", color=INK_2, va="bottom")

    _box(ax, 0.1, 3.55, 2.3, 2.8, "Benchmark refund_v1",
         ["74 synthetic refund cases", "188 variants: original +", "hidden counterfactual edits", "Template text, seed 1",
          "sha256 5f7b5617bd0a…", "", "Frozen; hash-checked"], INK_2, fill=PANEL)
    p0 = pilot0_summary["P0.8_cost"]
    _box(ax, 2.75, 3.55, 2.45, 2.8, "Jev  (System One)",
         ["jev-1.14.0 · TypeSafe native API", "$0.042 / 1M input, output free", "",
          "Pilot 0: 3 samples per variant", f"{p0['billed_decisions']} decisions, ${p0['total_usd']:.3f}",
          "", "Pilot 1 reuses these: 0 calls"], BLUE, title_color=BLUE)
    _box(ax, 5.55, 4.95, 3.15, 1.4, "Llama 4 Maverick  (Meta)",
         ["Bedrock, us-east-2 · $0.24 / $0.97", f"{ok[llama]} calls, ${cost['bedrock/' + llama]:.3f}",
          f"+{throttled[llama]} throttled and retried, unbilled"], ORANGE, title_color=ORANGE)
    _box(ax, 5.55, 3.4, 3.15, 1.4, "Gemini 3.1 Flash-Lite  (Google)",
         ["Vertex AI, global endpoint · $0.25 / $1.50", "thinking level minimal",
          f"{ok[gemini]} calls, ${cost['gemini/' + gemini]:.3f}"], ORANGE, title_color=ORANGE)
    _box(ax, 5.55, 1.85, 3.15, 1.4, "Claude Haiku 5.5  (Anthropic)",
         ["Planned; blocked: AWS refused the", "Marketplace subscription for the account"], MUTED, dashed=True,
         title_color=MUTED)
    _box(ax, 9.05, 3.55, 3.3, 2.8, "gpt-oss-120b  (OpenAI)",
         ["Simulator · Bedrock flex tier", "reasoning effort low · $0.075 / $0.30", "",
          "Predicts Jev on each hidden variant,", "with and without the explanation", "",
          f"{ok[oss]} calls, ${cost['bedrock/' + oss]:.3f}"], AQUA, title_color="#127a55")

    _arrow(ax, 2.4, 4.95, 2.75, 4.95)
    _arrow(ax, 5.2, 5.3, 5.55, 5.65)
    _arrow(ax, 5.2, 4.6, 5.55, 4.1)
    _arrow(ax, 8.7, 5.65, 9.05, 5.2)
    _arrow(ax, 8.7, 4.1, 9.05, 4.6)
    ax.annotate("", xy=(10.7, 3.55), xytext=(10.7, 1.55),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "linewidth": 1.2, "linestyle": (0, (3, 2)),
                            "mutation_scale": 10})
    ax.plot([3.95, 3.95, 10.7], [3.55, 1.55, 1.55], color=BLUE, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(4.1, 1.62, "Jev's real decisions on the hidden variants: the answer key, never shown to the simulator",
            fontsize=7.5, color=BLUE, ha="left", va="bottom")

    ax.text(0.1, 1.05, "CONFIGURED, NOT USED IN PILOT 1", fontsize=8, fontweight="bold", color=INK_2)
    later = [("Case writers", "Qwen3 235B (Alibaba), Mistral Large 3\nBedrock · natural-language refund_v2"),
             ("LLM judge", "gpt-oss-120b (OpenAI), Bedrock flex\ncontestability scoring, Pilot 2 on"),
             ("Second decision model", "Amazon Nova Lite, Bedrock\nExperiment 6: beyond Jev")]
    for i, (title, body) in enumerate(later):
        x = 0.1 + i * 4.15
        ax.add_patch(FancyBboxPatch((x, 0.05), 3.95, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=PANEL, edgecolor=GRID, linewidth=1))
        ax.text(x + 0.12, 0.78, title, fontsize=8.5, fontweight="bold", va="top")
        ax.text(x + 0.12, 0.5, body, fontsize=7.5, color=INK_2, va="top", linespacing=1.3)
    total = summary["cost"]["total_usd"]
    ax.text(12.35, 7.85, f"Pilot 1 total: ${total:.2f}", fontsize=10, fontweight="bold", ha="right", va="top")
    ax.text(12.35, 7.45, "Prices per 1M tokens, input / output", fontsize=8, color=INK_2, ha="right", va="top")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ceiling(summary, out):
    f = summary["faithfulness"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = list(range(len(GROUPS)))[::-1]
    for yi, (key, _) in zip(y, GROUPS):
        c, a = f[key]["control_agreement_with_jev"], f[key]["a_direct_agreement_with_jev"]
        ax.plot([c, a], [yi, yi], color=GRID, linewidth=3, zorder=1)
        ax.scatter([c], [yi], s=70, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([a], [yi], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        nsg = f[key]["normalized_simulatability_gain"]
        ax.text(1.005, yi, f"{a:.0%} vs {c:.0%}   NSG {nsg:+.2f}   (n = {f[key]['n_scored']})", va="center",
                fontsize=8.8, color=INK)
    ax.set_yticks(y, [label for _, label in GROUPS])
    ax.set_xlim(0.78, 1.0)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0], ["80%", "85%", "90%", "95%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hidden variants where the simulator's prediction matches Jev's real decision")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=c) for c in (BLUE, MUTED)]
    ax.legend(handles, ["With the explanation", "Without (control, same variants)"], frameon=False,
              loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=9)
    fig.suptitle("Explanations barely change how well Jev can be predicted", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.905, "Knowing the refund policy already predicts Jev about 90% of the time, so there is "
             "little left for an explanation to add.", fontsize=8.8, color=INK_2)
    fig.tight_layout(rect=(0, 0, 0.8, 0.9))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_departures(sims, variants, out):
    deps = sorted({r["variant_id"] for r in sims if r["jev_decision"] and r["jev_decision"] != r["policy_outcome"]})
    columns = [("control", "control", "No explanation\n(control)")] + [
        (k.split("/")[0], k.split("/")[1], SHORT[k].replace(", ", "\n")) for k, _ in GROUPS]
    pred = {(r["variant_id"], r["explainer"], r["condition"]): r for r in sims}
    kinds = {"exception_removed": "exception removed", "decisive_edit": "deciding fact changed",
             "cue_edit": "name / register swap", "resolution": "missing fact filled in",
             "discrepancy_fixed": "irrelevant error fixed"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_axis_off()
    n_rows, n_cols = len(deps), len(columns)
    x0, cw, ch = 4.0, 1.05, 0.42
    ax.set_xlim(0, x0 + n_cols * cw + 0.2)
    ax.set_ylim(-1.4, n_rows * ch + 1.95)
    for j, (_, _, label) in enumerate(columns):
        ax.text(x0 + j * cw + cw / 2, n_rows * ch + 0.15, label, ha="center", va="bottom", fontsize=8, color=INK)
    hits = Counter()
    for i, vid in enumerate(deps):
        yy = (n_rows - 1 - i) * ch
        case, v = variants[vid]
        days = counted_days(v.record)
        ax.text(x0 - 0.15, yy + ch / 2, f"{vid.removeprefix('refund_v1-')}  ·  {days} days  ·  {kinds[str(v.kind)]}",
                ha="right", va="center", fontsize=8.3, color=INK)
        for j, (explainer, condition, _) in enumerate(columns):
            r = pred.get((vid, explainer, condition))
            if r is None or not r["parse_ok"]:
                color, mark = PANEL, "–"
            elif r["predicted_decision"] == r["jev_decision"]:
                color, mark = BLUE, "✓"
                hits[j] += 1
            else:
                color, mark = "#f6d5c7", "✗"
            ax.add_patch(Rectangle((x0 + j * cw + 0.04, yy + 0.03), cw - 0.08, ch - 0.06, facecolor=color,
                                   edgecolor=SURFACE, linewidth=2))
            ax.text(x0 + j * cw + cw / 2, yy + ch / 2, mark, ha="center", va="center", fontsize=10,
                    color=SURFACE if color == BLUE else ORANGE if mark == "✗" else INK_2, fontweight="bold")
    for j in range(n_cols):
        ax.text(x0 + j * cw + cw / 2, -0.25, f"{hits[j]}/{n_rows}", ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=INK)
    ax.text(x0 - 0.15, -0.25, "Matches Jev", ha="right", va="top", fontsize=9, fontweight="bold")
    ax.text(0, n_rows * ch + 1.9, "Where Jev breaks its own window, no explanation helps reliably",
            fontsize=12, fontweight="bold", va="top")
    ax.text(0, n_rows * ch + 1.55, "The 11 hidden variants where Jev's decision departs from the policy: all past "
            "the 30-day window, all approved by Jev.\n✓ = simulator predicted Jev's approval; ✗ = predicted the "
            "policy's denial.", fontsize=8.5, color=INK_2, va="top")
    ax.text(0, -0.75, "Llama-structured's 7 hits: 3 where its rules happen to omit the window for that case, 4 where "
            "the simulator overrode its own executed rules.\nThe 3 hits shared by every column (042-v1, 042-v2, "
            "063-v1) keep the original's date, which Jev had already approved.", fontsize=8, color=INK_2, va="top")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"departures": n_rows, "hits": {columns[j][2].replace("\n", " "): hits[j] for j in range(n_cols)}}


def option_b_split(expls, variants, cases):
    """Structured explanations' own rules run on each hidden variant: agreement with the policy and with Jev,
    split by whether Jev follows the policy on that variant."""
    out = {}
    jev = {}
    for r in _read_jsonl(RUN_DIR / "simulations.jsonl"):
        if r["condition"] == "control" and r["jev_decision"]:
            jev[r["variant_id"]] = r["jev_decision"]
    by_case = {c.case_id: c for c in cases}
    for explainer in ("explainer_gemini", "explainer_open"):
        cells = defaultdict(lambda: [0, 0, 0])  # [agree_policy, agree_jev, n]
        for e in expls:
            if e["explainer"] != explainer or e["condition"] != "structured" or not e["parse_ok"]:
                continue
            for v in by_case[e["case_id"]].variants[1:]:
                if v.variant_id not in jev:
                    continue
                d = _executed(e["basis"], v.record)
                if d is None:
                    continue
                subset = "Jev follows the policy" if jev[v.variant_id] == str(v.policy_outcome) else "Jev departs"
                cells[subset][0] += d == str(v.policy_outcome)
                cells[subset][1] += d == jev[v.variant_id]
                cells[subset][2] += 1
        out[explainer] = {k: {"agree_policy": a / n, "agree_jev": j / n, "n": n} for k, (a, j, n) in cells.items()}
    return out


def fig_rules_track_policy(split, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    names = {"explainer_gemini": "Gemini", "explainer_open": "Llama"}
    for ax, subset in zip(axes, ("Jev follows the policy", "Jev departs")):
        xs = list(range(len(names)))
        for k, (key, color, label) in enumerate((("agree_policy", MUTED, "matches the policy"),
                                                 ("agree_jev", BLUE, "matches Jev"))):
            vals = [split[e][subset][key] for e in names]
            bars = ax.bar([x + (k - 0.5) * 0.36 for x in xs], vals, width=0.34, color=color, edgecolor=SURFACE,
                          linewidth=2, label=label)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.0%}", ha="center", fontsize=9)
        ax.set_xticks(xs, [f"{names[e]}\n(n = {split[e][subset]['n']})" for e in names])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", visible=False)
        ax.set_title(subset + " on the variant", fontsize=10)
    axes[0].set_ylabel("Hidden variants decided by the\nexplanation's own rules")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Structured explanations restate the policy, not Jev", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Option B: each structured explanation's rules run directly against the hidden variants' "
             "records (no simulator). Variants the rules leave undecided are excluded.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_justification(out):
    rows = [("explainer_open", "free_form", "Llama, free-form"), ("explainer_gemini", "free_form", "Gemini, free-form"),
            ("explainer_open", "structured", "Llama, structured"),
            ("explainer_gemini", "structured", "Gemini, structured")]
    codes = [("false_premise", ORANGE), ("invented_fact", "#f3a27f"), ("flags_conflict", BLUE)]
    count = Counter((e, c, code) for _, e, c, code, _ in JUSTIFICATIONS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = list(range(len(rows)))[::-1]
    left = [0] * len(rows)
    for code, color in codes:
        vals = [count[(e, c, code)] for e, c, _ in rows]
        bars = ax.barh(y, vals, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2,
                       label=JUSTIFICATION_CODES[code])
        for b, val in zip(bars, vals):
            if val:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, str(val), ha="center",
                        va="center", color=SURFACE if color != "#f3a27f" else INK, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y, [label for *_, label in rows])
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Explanations of the 3 originals Jev approved 43–57 days after purchase (policy: deny)")
    fig.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.2, 0.0), ncol=1, fontsize=8.5)
    fig.suptitle("Asked to explain Jev's wrong approvals, explainers justify them", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.905, "10 of 12 justify the approval with a false or invented premise; none says plainly that "
             "it breaks the 30-day rule.\nHand-coded; quotes in data/justification_coding.csv.", fontsize=8.8,
             color=INK_2, va="top")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.76, bottom=0.36)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_disagreement(summary, out):
    d = summary["jev_explainer_disagreement"]
    names = {"explainer_gemini": "Gemini 3.1 Flash-Lite", "explainer_open": "Llama 4 Maverick"}
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = [1, 0]
    for k, (part, color, label) in enumerate((("originals", BLUE, "Original cases (n = 74)"),
                                              ("variants", AQUA, "Hidden variants (n = 114)"))):
        vals = [d[e]["disagreement_with_jev"][part] for e in names]
        bars = ax.barh([yy + (0.5 - k) * 0.34 for yy in y], [v["rate"] for v in vals], height=0.32, color=color,
                       edgecolor=SURFACE, linewidth=2, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2, f"{v['rate']:.0%} ({v['k']}/{v['n']})",
                    va="center", fontsize=9)
    ax.set_yticks(y, [f"{n}\nmatches policy {d[e]['explainer_policy_agreement']:.0%}, "
                      f"unanimous {d[e]['explainer_unanimous_share']:.0%}" for e, n in names.items()], fontsize=8.8)
    ax.set_xlim(0, 0.25)
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25], ["0%", "5%", "10%", "15%", "20%", "25%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Variants where the explainer's own majority decision differs from Jev's")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("Deciding for themselves, the explainers mostly agree with Jev", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Each explainer decided every variant 3 times, with Jev's question and rotating option "
             "order. Majority of 3 vs Jev's Pilot 0 majority of 3.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_data_quality(raw_sims, summary, out):
    groups = [("control", "control", "No explanation (control)")] + [
        (k.split("/")[0], k.split("/")[1], SHORT[k]) for k, _ in GROUPS]
    latest = _latest_ok(reparse_simulations(raw_sims), "explanation_id", "variant_id")
    counts = {g: Counter() for g in groups}
    for r in latest:
        g = next(g for g in groups if g[0] == r["explainer"] and g[1] == r["condition"])
        counts[g]["repaired" if r.get("repaired") and r["parse_ok"] else "clean" if r["parse_ok"] else "failed"] += 1
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    y = list(range(len(groups)))[::-1]
    left = [0] * len(groups)
    for key, color, label in (("clean", BLUE, "Parsed as returned"),
                              ("repaired", AQUA, "Parsed after repairing a spelled-out digit (\"0. nine\")"),
                              ("failed", ORANGE, "Unparsed")):
        vals = [counts[g][key] for g in groups]
        ax.barh(y, vals, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left = [l + v for l, v in zip(left, vals)]
    for yi, g in zip(y, groups):
        c = counts[g]
        n = sum(c.values())
        ax.text(n + 1.5, yi, f"{c['repaired']} repaired, {c['failed']} unparsed of {n}", va="center", fontsize=8.5)
    ax.set_yticks(y, [label for *_, label in groups])
    ax.set_xlim(0, 160)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Simulator replies (one per hidden variant per condition)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.24), ncol=1, fontsize=8.5)
    expl = summary["explanation_parse"]["explainer_open/structured"]
    fig.suptitle("gpt-oss garbles 13% of its replies; a narrow repair recovers almost all", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.905, f"Llama's structured condition has fewer replies: {expl['n'] - expl['k']} of its "
             f"{expl['n']} explanations had invalid rules and were not simulated.", fontsize=8.8, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {f"{g[0]}/{g[1]}": dict(counts[g]) for g in groups}


def jev_or_explainer(sims, expls, variants):
    """The slide's E1 question: on hidden variants where the explainer's own majority decision differs from
    Jev's, does the simulator (reading the explanation) predict Jev's decision or the explainer's own?"""
    samples = defaultdict(list)
    for r in _read_jsonl(RUN_DIR / "explainer_decisions.jsonl"):
        if r["parse_ok"]:
            samples[(r["explainer"], r["variant_id"])].append(r["decision"])
    own = {k: majority(v) for k, v in samples.items()}
    control = {r["variant_id"]: r for r in sims if r["condition"] == "control" and r["parse_ok"]}
    by_id = {e["explanation_id"]: e for e in expls}

    def side(pred, jev, mine):
        return "jev" if pred == jev else "explainer" if pred == mine else "neither"

    out = {}
    for explainer in ("explainer_gemini", "explainer_open"):
        for condition in ("free_form", "structured"):
            rows = [r for r in sims if r["explainer"] == explainer and r["condition"] == condition and r["parse_ok"]
                    and r["jev_decision"] and own.get((explainer, r["variant_id"])) not in (None, r["jev_decision"])]
            cell = {"n": len(rows),
                    "simulator": Counter(side(r["predicted_decision"], r["jev_decision"],
                                              own[(explainer, r["variant_id"])]) for r in rows),
                    "control": Counter(side(control[r["variant_id"]]["predicted_decision"], r["jev_decision"],
                                            own[(explainer, r["variant_id"])]) for r in rows
                                       if r["variant_id"] in control),
                    "explainer_policy_agree": sum(own[(explainer, r["variant_id"])] == r["policy_outcome"]
                                                  for r in rows)}
            if condition == "structured":
                cell["rules"] = Counter(
                    side(_executed(by_id[r["explanation_id"]]["basis"], variants[r["variant_id"]][1].record),
                         r["jev_decision"], own[(explainer, r["variant_id"])]) for r in rows)
            out[f"{explainer}/{condition}"] = {k: dict(v) if isinstance(v, Counter) else v for k, v in cell.items()}
    return out


def fig_jev_or_explainer(split, out):
    rows = []
    for key, label in (("explainer_gemini/free_form", "Gemini, free-form"),
                       ("explainer_gemini/structured", "Gemini, structured"),
                       ("explainer_open/free_form", "Llama, free-form"),
                       ("explainer_open/structured", "Llama, structured")):
        c = split[key]
        rows.append((f"{label}: simulator with explanation", c["simulator"], c["n"]))
        if "rules" in c:
            rows.append((f"{label}: explanation's rules alone", c["rules"], c["n"]))
    rows.insert(0, ("No explanation (control), Gemini's variants", split["explainer_gemini/free_form"]["control"],
                    split["explainer_gemini/free_form"]["n"]))
    rows.insert(4, ("No explanation (control), Llama's variants", split["explainer_open/free_form"]["control"],
                    split["explainer_open/free_form"]["n"]))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = list(range(len(rows)))[::-1]
    left = [0.0] * len(rows)
    for key, color, label in (("explainer", ORANGE, "Predicts the explainer's own decision"),
                              ("jev", BLUE, "Predicts Jev's decision"), ("neither", MUTED, "Neither / undecided")):
        vals = [c.get(key, 0) / n for _, c, n in rows]
        bars = ax.barh(y, vals, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for b, (_, c, n) in zip(bars, rows):
            if c.get(key, 0):
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, str(c[key]), ha="center",
                        va="center", color=SURFACE if color != MUTED else INK, fontsize=9, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y, [f"{label} (n = {n})" for label, _, n in rows], fontsize=8.5)
    for yi, (label, _, _) in zip(y, rows):
        if label.startswith("No explanation"):
            ax.get_yticklabels()[len(rows) - 1 - yi].set_color(MUTED)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hidden variants where the explainer would itself decide differently from Jev")
    fig.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=8.5)
    fig.suptitle("Whose decisions does the explanation predict: Jev's or the explainer's?", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.905, "Gemini's explanations predict Gemini. On these variants the explainer's own decision is "
             "the policy's in 18 of 20,\nso this benchmark can't yet separate 'the explainer's own story' from "
             "'the policy' (E2's paired targets can).", fontsize=8.8, color=INK_2, va="top")
    fig.subplots_adjust(left=0.4, right=0.98, top=0.8, bottom=0.2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_sets_up(out):
    findings = [
        ("Ceiling: the policy alone predicts\nJev on 90% of hidden variants", BLUE),
        ("Only 11 of 114 hidden variants\nseparate Jev from its policy", BLUE),
        ("Explanations restate the policy;\nnone describes Jev's lenient window", ORANGE),
        ("Wrong approvals get justified: 10 of 12\nexplanations use a false or invented premise", ORANGE),
        ("Simulator overrides its own rules;\n13% of replies garbled", AQUA),
        ("Whole run $0.41; Jev decisions\nreused at no cost", AQUA),
    ]
    uses = [
        "E1 Jev vs. explainer: add hidden variants\nJev's own threshold decides (window sweep);\nkeep the control",
        "E2 similar targets: pair targets that differ\nin one rule (e.g. a 30- vs 45-day window)\nso the target, not the policy, is tested",
        "E3 information ablation: Pilot 1 is the\n'input + choice' rung; date probes should\nreveal the threshold one decision hides",
        "Unsupported claims (E3 metric; stretch):\nscore invented facts and false premises\nwith the LLM judge",
        "Scoring and access: report A-direct with\nA-executed and Option B; add Claude\nif AWS unblocks it",
    ]
    links = {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [3], 4: [4], 5: [2, 4]}
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 6.6)
    ax.text(0.2, 6.35, "What Pilot 1 found", fontsize=11, fontweight="bold")
    ax.text(6.9, 6.35, "What it means for this phase (E1–E3)", fontsize=11, fontweight="bold")
    fy = [5.65 - i * 0.98 for i in range(len(findings))]
    uy = [5.55 - i * 1.17 for i in range(len(uses))]
    for i, (text, color) in enumerate(findings):
        ax.add_patch(FancyBboxPatch((0.2, fy[i] - 0.38), 4.0, 0.76, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=SURFACE, edgecolor=color, linewidth=2))
        ax.text(0.38, fy[i], text, va="center", fontsize=8.8)
    for j, text in enumerate(uses):
        ax.add_patch(FancyBboxPatch((6.9, uy[j] - 0.47), 4.55, 0.94, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=PANEL, edgecolor=GRID, linewidth=1.2))
        ax.text(7.08, uy[j], text, va="center", fontsize=8.6)
    for i, targets in links.items():
        for j in targets:
            ax.annotate("", xy=(6.9, uy[j]), xytext=(4.2, fy[i]),
                        arrowprops={"arrowstyle": "-|>", "color": findings[i][1], "linewidth": 1.1, "alpha": 0.75,
                                    "shrinkA": 2, "shrinkB": 2, "mutation_scale": 9})
    handles = [plt.Line2D([0], [0], color=c, linewidth=3) for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["Benchmark limits", "Explainer behavior", "Method and cost"], frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, -0.03), ncol=3, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manifest, cases, meta, variants, sims, raw_sims, expls, ledger, _ = load()
    summary_raw = analyze_run(RUN_DIR, repair=False)
    summary = analyze_run(RUN_DIR)  # leaves the repaired report in the run folder
    save_data(summary, summary_raw, sims, variants, meta, ledger)
    pilot0_summary = json.loads((REPO_ROOT / "results" / "runs" / PILOT0 / "summary.json").read_text())
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    fig_model_stack(summary, ledger, pilot0_summary, figs / "fig1_model_stack.png")
    fig_ceiling(summary, figs / "fig2_ceiling.png")
    dep = fig_departures(sims, variants, figs / "fig3_policy_departures.png")
    split = option_b_split(expls, variants, cases)
    (HERE / "data" / "option_b_split.json").write_text(json.dumps(split, indent=2) + "\n")
    fig_rules_track_policy(split, figs / "fig4_rules_track_policy.png")
    fig_justification(figs / "fig5_justifying_wrong_approvals.png")
    fig_disagreement(summary, figs / "fig6_explainer_vs_jev.png")
    quality = fig_data_quality(raw_sims, summary, figs / "fig7_simulator_parsing.png")
    fig_sets_up(figs / "fig8_what_pilot1_sets_up.png")
    whose = jev_or_explainer(sims, expls, variants)
    (HERE / "data" / "jev_or_explainer.json").write_text(json.dumps(whose, indent=2) + "\n")
    fig_jev_or_explainer(whose, figs / "fig9_jev_or_explainer.png")
    print("jev or explainer:", json.dumps(whose))
    print("departures:", dep)
    print("option B split:", json.dumps(split))
    print("parse outcomes:", quality)
    print("git:", manifest["git"])
    print(f"wrote {figs} and {HERE / 'data'}")


if __name__ == "__main__":
    main()
