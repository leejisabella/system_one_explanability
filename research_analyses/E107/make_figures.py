"""E.107 figures and data, regenerated from the saved run.

Usage:  uv run --group analysis python research_analyses/E107/make_figures.py

E.107 was evaluated inside E.106's run (results/runs/e106-v1-2026-11-04/, reviewer `dpo_review`),
so it is compared with the base and both SFT arms on the same items and stack. Reads that run (local
only; logs/e107.log is the GPU machine's DPO pipeline log) and E.102's frozen plan. Writes derived
tables to data/ (committed) and figures to figures/. No API calls, no GPU.
"""

from __future__ import annotations

import ast
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysone.analysis.e101 import bootstrap_diff, rate  # noqa: E402
from sysone.analysis.e102 import paired_bootstrap  # noqa: E402
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN, SOURCE = "e106-v1-2026-11-04", "e102-v1-2026-11-04"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN
SOURCE_DIR = REPO_ROOT / "results" / "runs" / SOURCE
SAMPLES = [("civil_toxicity_pool_v1", "Civil Comments"), ("wildguard_prompt_full_v1", "WildGuard prompts"),
           ("wildguard_response_full_v1", "WildGuard responses")]
REVIEWERS = [("base_local", "Base (untrained)"), ("sft_review", "SFT treatment (DPO's starting point)"),
             ("sft_plain", "SFT control"), ("dpo_review", "DPO")]
METRICS = [("override_when_wrong_jev", "Jev's wrong labels overridden"),
           ("accept_when_right_jev", "Jev's right labels accepted"),
           ("cascade_accuracy", "Cascade accuracy"), ("alone_accuracy", "Accuracy alone")]

# Reference palette (dataviz skill, light mode), as in E0–E.106: base gray, SFT treatment blue,
# DPO aqua (below 3:1 contrast, so every value is labelled directly).
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, AQUA = "#2a78d6", "#1baf7a"
COLOR = {"base_local": MUTED, "sft_review": BLUE, "dpo_review": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.parse_math": False,
})


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / "data" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def save(fig, name: str) -> None:
    fig.savefig(HERE / "figures" / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def flat(d: dict) -> dict:
    ci = d.get("ci95") or [None, None]
    return {"rate": d.get("rate"), "k": d.get("k"), "n": d.get("n"), "ci_lo": ci[0], "ci_hi": ci[1]}


def main() -> None:
    for sub in ("data", "figures"):
        (HERE / sub).mkdir(exist_ok=True)
    for name in ("report.md", "summary.json"):
        shutil.copy(RUN_DIR / name, HERE / "data" / name)
    S = json.loads((RUN_DIR / "summary.json").read_text())
    R = S["reviewers"]

    rows = []
    for who, _ in REVIEWERS:
        for sample, block in [("pooled", R[who]["pooled"])] + [(s, R[who]["samples"][s]) for s, _ in SAMPLES]:
            for metric in ("override_when_wrong_jev", "override_when_wrong_matched", "accept_when_right_jev",
                           "cascade_accuracy_subsample", "alone_accuracy_subsample", "jev_wrong_accepted_never_right",
                           "harmful_let_through"):
                if block.get(metric) and block[metric].get("n"):
                    rows.append({"reviewer": who, "sample": sample, "metric": metric, **flat(block[metric])})
            d = block.get("deference", {}).get("eligible")
            if d:
                rows.append({"reviewer": who, "sample": sample, "metric": "deference_right_alone",
                             "rate": d["diff"], "n": d["n_jev"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]})
            c = block.get("cascade_vs_best_single")
            if c and c.get("diff") is not None:
                rows.append({"reviewer": who, "sample": sample, "metric": "cascade_minus_jev_alone", "rate": c["diff"],
                             "n": c.get("n"), "ci_lo": (c.get("ci95") or [None])[0], "ci_hi": (c.get("ci95") or [None, None])[1]})
    write_csv("rates.csv", rows)

    vs = [{"reviewer": who, "metric": m, "diff": R[who]["pooled"]["vs_base"][m]["diff"],
           "ci_lo": R[who]["pooled"]["vs_base"][m]["ci95"][0], "ci_hi": R[who]["pooled"]["vs_base"][m]["ci95"][1],
           "n": R[who]["pooled"]["vs_base"][m]["n"]} for who in ("sft_review", "dpo_review") for m, _ in METRICS]
    write_csv("vs_base.csv", vs)
    crit = [{"criterion": name, "pass": c["pass"], "rule": c["rule"]} for name, c in S["criteria"]["dpo_review"].items()]
    write_csv("criteria.csv", crit)
    c = S["criteria"]["dpo_review"]
    write_csv("criterion2_per_pool.csv", [{"sample": s, **v} for s, v in c["2_beats_cheap_protocols"]["per_sample"].items()])
    write_csv("upstream_vs_base.csv", [{"upstream": u, "measure": m, "diff": v[m]["diff"], "ci_lo": v[m]["ci95"][0],
                                        "ci_hi": v[m]["ci95"][1], "n": v[m]["n"]}
                                       for u, v in c["4_transfers_to_held_out_upstreams"]["per_upstream"].items()
                                       for m in ("override_when_wrong", "accept_when_right")])
    vc = c["5_beats_control_arm"]["vs_control"]
    write_csv("dpo_vs_control.csv", [{"metric": m, "diff": vc[m]["diff"], "ci_lo": vc[m]["ci95"][0],
                                      "ci_hi": vc[m]["ci95"][1], "n": vc[m]["n"]}
                                     for m in ("override_when_wrong_jev", "accept_when_right_jev")])

    # Per-review flags, for DPO against its starting point (the SFT treatment), paired by item.
    plan = {(p["sample"], p["item_id"]): p for p in jsonl(SOURCE_DIR / "plan.jsonl")}
    reviews = [r for r in jsonl(RUN_DIR / "reviews.jsonl") if r["parse_ok"]]
    verdict: dict[str, dict] = defaultdict(dict)
    alone: dict[str, dict] = defaultdict(dict)
    review: dict[str, list] = defaultdict(list)
    for r in reviews:
        key = (r["sample"], r["item_id"])
        if r["condition"] == "alone" and r["sample_index"] == 0:
            alone[r["reviewer"]][key] = r["final_label"]
        elif r["condition"] in ("jev_label", "correct_label"):
            verdict[r["reviewer"]][(r["condition"], key)] = r["final_label"] == r["proposed_label"]
        if r["condition"] in ("jev_label", "matched_label") and r["proposed_label"] != plan[key]["label"]:
            review[r["reviewer"]].append((r["condition"], key, r["proposed_label"], r["final_label"]))
    vs_start = []
    for cond, metric, want in (("jev_label", "override_when_wrong_jev", False), ("correct_label", "accept_when_right_jev", True)):
        # Jev's wrong labels for overrides; the correct label on items Jev got right for accepts.
        jev_correct_wanted = cond == "correct_label"
        keys = sorted(k for k in verdict["dpo_review"] if k[0] == cond and k in verdict["sft_review"]
                      and plan[k[1]]["jev_correct"] == jev_correct_wanted)
        a = [verdict["dpo_review"][k] == want for k in keys]
        b = [verdict["sft_review"][k] == want for k in keys]
        diff = lambda idx: sum(a[i] - b[i] for i in idx) / len(idx)
        vs_start.append({"metric": metric, "diff": round(diff(range(len(keys))), 4), "n": len(keys),
                         **dict(zip(("ci_lo", "ci_hi"), paired_bootstrap(len(keys), diff, seed=107)))})
    write_csv("dpo_vs_sft_start.csv", vs_start)

    # Switching to a wrong proposal the reviewer disagreed with alone (as in E.106).
    switch, flags = [], {}
    for who, _ in REVIEWERS:
        for cond in ("jev_label", "matched_label"):
            f = [fin == prop for c_, k, prop, fin in review[who] if c_ == cond and k in alone[who] and alone[who][k] != prop]
            flags[(who, cond)] = f
            row = {"reviewer": who, "proposal": cond, **flat(rate(f))}
            if who != "base_local":
                d = bootstrap_diff(f, flags[("base_local", cond)], seed=106)
                row.update({"diff_vs_base": round(d["diff"], 4), "diff_ci_lo": round(d["ci95"][0], 4),
                            "diff_ci_hi": round(d["ci95"][1], 4)})
            switch.append(row)
    write_csv("switch_to_wrong_proposal.csv", switch)
    common = set.intersection(*(set(alone[w]) for w, _ in REVIEWERS))
    jw = [k for k in common if not plan[k]["jev_correct"]]
    write_csv("alone_agreement.csv", [{"reviewer": w, "items": len(common),
                                       "agrees_with_jev": rate([alone[w][k] == plan[k]["jev_label"] for k in common])["rate"],
                                       "accuracy": rate([alone[w][k] == plan[k]["label"] for k in common])["rate"],
                                       "accuracy_on_jev_wrong": rate([alone[w][k] == plan[k]["label"] for k in jw])["rate"],
                                       "n_jev_wrong": len(jw)} for w, _ in REVIEWERS])

    # DPO training metrics, from the pipeline log.
    log = (RUN_DIR / "logs" / "e107.log").read_text().replace("\r", "\n")
    curve = []
    for line in log.splitlines():
        start = max(line.find("{'loss'"), line.find("{'eval_loss'"))   # may follow a progress bar on the line
        if start >= 0:
            d = ast.literal_eval(line[start:])
            p = "eval_" if "eval_loss" in d else ""
            curve.append({"split": "val" if p else "train", "epoch": round(d["epoch"], 3),
                          "loss": round(d[f"{p}loss"], 4), "grad_norm": round(d["grad_norm"], 3) if not p else "",
                          "reward_accuracy": round(d[f"{p}rewards/accuracies"], 4),
                          "reward_margin": round(d[f"{p}rewards/margins"], 4),
                          "reward_chosen": round(d[f"{p}rewards/chosen"], 4),
                          "reward_rejected": round(d[f"{p}rewards/rejected"], 4)})
    write_csv("dpo_training.csv", curve)

    # Figure: base -> SFT treatment -> DPO on the four headline rates and switching.
    panels = [("override_when_wrong_jev", "Jev's wrong labels overridden"),
              ("accept_when_right_jev", "Jev's right labels accepted"),
              ("cascade_accuracy_subsample", "Cascade accuracy"), ("switch", "Switches to Jev's wrong label\n(items it gets right alone)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (metric, title) in zip(axes, panels):
        for x, who in enumerate(("base_local", "sft_review", "dpo_review")):
            d = (next(r for r in switch if r["reviewer"] == who and r["proposal"] == "jev_label") if metric == "switch"
                 else flat(R[who]["pooled"][metric]))
            ax.errorbar(x, d["rate"] * 100, yerr=[[(d["rate"] - d["ci_lo"]) * 100], [(d["ci_hi"] - d["rate"]) * 100]],
                        fmt="o", color=COLOR[who], ecolor=COLOR[who], elinewidth=2, capsize=3, markersize=7)
            ax.text(x + 0.12, d["rate"] * 100, f"{d['rate'] * 100:.0f}%", va="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(range(3), ["Base", "SFT", "DPO"])
        ax.set_xlim(-0.4, 2.6)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("% (pooled, 95% interval)")
    fig.suptitle("DPO landed back near the base on overrides and switching, and no better", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig1_base_sft_dpo.png")
    print("wrote", sorted(p.name for p in (HERE / "data").iterdir()), sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
