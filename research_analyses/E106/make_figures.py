"""E.106 figures and data, regenerated from the saved run.

Usage:  uv run --group analysis python research_analyses/E106/make_figures.py

Reads results/runs/e106-v1-2026-11-04/ (local only; its logs/ folder holds the GPU machine's
pipeline and vLLM logs, copied back after the run), E.102's frozen plan (e102-v1-2026-11-04) and
E.103's standard-prompt numbers (research_analyses/E103/data). Writes derived tables to data/
(committed) and figures to figures/. No API calls, no GPU.
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
from sysone.registry import REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN, SOURCE = "e106-v1-2026-11-04", "e102-v1-2026-11-04"
RUN_DIR = REPO_ROOT / "results" / "runs" / RUN
SOURCE_DIR = REPO_ROOT / "results" / "runs" / SOURCE
SAMPLES = [("civil_toxicity_pool_v1", "Civil Comments"), ("wildguard_prompt_full_v1", "WildGuard prompts"),
           ("wildguard_response_full_v1", "WildGuard responses")]
REVIEWERS = [("base_local", "Base (untrained)"), ("sft_review", "SFT, treatment (with Jev's proposal)"),
             ("sft_plain", "SFT, control (no proposal)")]
TRAINED = REVIEWERS[1:]
METRICS = [("override_when_wrong_jev", "Jev's wrong labels overridden"),
           ("accept_when_right_jev", "Jev's right labels accepted"),
           ("cascade_accuracy", "Cascade accuracy"), ("alone_accuracy", "Accuracy alone")]

# Reference palette (dataviz skill, light mode), as in E0–E.103. Two trained arms: blue and orange,
# always with a legend and direct value labels; the base is muted gray.
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29d"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COLOR = {"base_local": MUTED, "sft_review": BLUE, "sft_plain": ORANGE}

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
    return {"rate": d.get("rate"), "k": d.get("k"), "n": d.get("n"),
            "ci_lo": (d.get("ci95") or [None, None])[0], "ci_hi": (d.get("ci95") or [None, None])[1]}


def main() -> None:
    for sub in ("data", "figures"):
        (HERE / sub).mkdir(exist_ok=True)
    for name in ("report.md", "summary.json"):
        shutil.copy(RUN_DIR / name, HERE / "data" / name)
    S = json.loads((RUN_DIR / "summary.json").read_text())
    R = S["reviewers"]

    # Pooled and per-pool rates for every reviewer.
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
    write_csv("rates.csv", rows)

    vs = [{"reviewer": who, "metric": m, "diff": R[who]["pooled"]["vs_base"][m]["diff"],
           "ci_lo": R[who]["pooled"]["vs_base"][m]["ci95"][0], "ci_hi": R[who]["pooled"]["vs_base"][m]["ci95"][1],
           "n": R[who]["pooled"]["vs_base"][m]["n"]} for who, _ in TRAINED for m, _ in METRICS]
    write_csv("vs_base.csv", vs)

    crit = []
    for who, _ in TRAINED:
        for name, c in S["criteria"][who].items():
            crit.append({"reviewer": who, "criterion": name, "pass": c["pass"], "rule": c["rule"]})
    write_csv("criteria.csv", crit)
    cheap = [{"reviewer": who, "sample": s, **v} for who, _ in TRAINED
             for s, v in S["criteria"][who]["2_beats_cheap_protocols"]["per_sample"].items()]
    write_csv("criterion2_per_pool.csv", cheap)
    ups = [{"reviewer": who, "upstream": u, "measure": m, "diff": v[m]["diff"], "ci_lo": v[m]["ci95"][0],
            "ci_hi": v[m]["ci95"][1], "n": v[m]["n"]}
           for who, _ in TRAINED for u, v in S["criteria"][who]["4_transfers_to_held_out_upstreams"]["per_upstream"].items()
           for m in ("override_when_wrong", "accept_when_right")]
    write_csv("upstream_vs_base.csv", ups)
    vc = S["criteria"]["sft_review"]["5_beats_control_arm"]["vs_control"]
    write_csv("treatment_vs_control.csv", [{"metric": m, "diff": vc[m]["diff"], "ci_lo": vc[m]["ci95"][0],
                                            "ci_hi": vc[m]["ci95"][1], "n": vc[m]["n"]}
                                           for m in ("override_when_wrong_jev", "accept_when_right_jev")])

    # The base on our GPU stack vs. the same model on Bedrock (E.102's standard prompt, from E.103's data).
    bedrock = {r["sample"]: r for r in csv.DictReader((REPO_ROOT / "research_analyses/E103/data/protocols.csv").open())
               if r["protocol"] == "standard"}
    write_csv("base_gpu_vs_bedrock.csv", [
        {"sample": s, "metric": m, "gpu": R["base_local"]["samples"][s][m]["rate"], "bedrock": float(bedrock[s][key])}
        for s, _ in SAMPLES for m, key in (("override_when_wrong_jev", "override_when_wrong_jev"),
                                            ("accept_when_right_jev", "accept_when_right_jev"),
                                            ("cascade_accuracy_subsample", "cascade_accuracy_subsample"))])

    # Alone decisions (first sample) and how often each reviewer switches to a wrong proposal it
    # disagreed with alone: Jev's wrong labels vs. matched random wrong labels.
    plan = {(p["sample"], p["item_id"]): p for p in jsonl(SOURCE_DIR / "plan.jsonl")}
    alone: dict[str, dict] = defaultdict(dict)
    review: dict[str, list] = defaultdict(list)
    tokens: dict[tuple, list] = defaultdict(list)
    for r in jsonl(RUN_DIR / "reviews.jsonl"):
        if not r["parse_ok"]:
            continue
        key = (r["sample"], r["item_id"])
        cond = "upstream" if r["condition"].startswith("upstream") else r["condition"]
        tokens[(r["reviewer"], cond)].append(r["output_tokens"] or 0)
        if r["condition"] == "alone" and r["sample_index"] == 0:
            alone[r["reviewer"]][key] = r["final_label"]
        elif r["condition"] in ("jev_label", "matched_label") and r["proposed_label"] != plan[key]["label"]:
            review[r["reviewer"]].append((r["condition"], key, r["proposed_label"], r["final_label"]))
    common = set.intersection(*(set(alone[w]) for w, _ in REVIEWERS))
    jev_wrong = [k for k in common if not plan[k]["jev_correct"]]
    jev_right = [k for k in common if plan[k]["jev_correct"]]
    agree = []
    for who, _ in REVIEWERS:
        a = alone[who]
        agree.append({"reviewer": who, "items": len(common),
                      "agrees_with_jev": rate([a[k] == plan[k]["jev_label"] for k in common])["rate"],
                      "accuracy": rate([a[k] == plan[k]["label"] for k in common])["rate"],
                      "accuracy_on_jev_wrong": rate([a[k] == plan[k]["label"] for k in jev_wrong])["rate"],
                      "n_jev_wrong": len(jev_wrong),
                      "accuracy_on_jev_right": rate([a[k] == plan[k]["label"] for k in jev_right])["rate"],
                      "n_jev_right": len(jev_right),
                      **({"jev_wrong_gained_vs_base": sum(alone["base_local"][k] != plan[k]["label"] and a[k] == plan[k]["label"] for k in jev_wrong),
                          "jev_wrong_lost_vs_base": sum(alone["base_local"][k] == plan[k]["label"] and a[k] != plan[k]["label"] for k in jev_wrong)}
                         if who != "base_local" else {})})
    write_csv("alone_agreement.csv", agree)

    switch, flags = [], {}
    for who, _ in REVIEWERS:
        for cond in ("jev_label", "matched_label"):
            f = [fin == prop for c, k, prop, fin in review[who] if c == cond and k in alone[who] and alone[who][k] != prop]
            flags[(who, cond)] = f
            row = {"reviewer": who, "proposal": cond, **flat(rate(f))}
            if who != "base_local":
                d = bootstrap_diff(f, flags[("base_local", cond)], seed=106)
                row.update({"diff_vs_base": round(d["diff"], 4), "diff_ci_lo": round(d["ci95"][0], 4),
                            "diff_ci_hi": round(d["ci95"][1], 4)})
            switch.append(row)
    write_csv("switch_to_wrong_proposal.csv", switch)
    write_csv("output_tokens.csv", [{"reviewer": w, "condition": c,
                                     "median_output_tokens": sorted(tokens[(w, c)])[len(tokens[(w, c)]) // 2]}
                                    for w, _ in REVIEWERS for c in ("alone", "jev_label", "matched_label", "correct_label", "upstream")])

    # Training curves, from the pipeline log (the control arm's second, completed attempt).
    log = (RUN_DIR / "logs" / "e106.log").read_text().replace("\r", "\n").splitlines()
    curves, arm = [], None
    for line in log:
        if line.startswith("STAGE sft ") and line.split()[3] == "START":
            arm, step = line.split()[2], 0
            curves = [c for c in curves if c["arm"] != arm]   # a restarted arm replaces its failed attempt
        elif arm and line.startswith("{'"):
            d = ast.literal_eval(line)
            if "loss" in d and "eval_loss" not in d:
                step += 10
                curves.append({"arm": arm, "step": step, "loss": round(d["loss"], 4),
                               "token_accuracy": round(d["mean_token_accuracy"], 4), "epoch": round(d["epoch"], 3)})
            elif "eval_loss" in d:
                curves.append({"arm": arm, "step": "val", "loss": round(d["eval_loss"], 4),
                               "token_accuracy": round(d["eval_mean_token_accuracy"], 4), "epoch": round(d["epoch"], 3)})
    write_csv("training_curves.csv", curves)

    # Figure 1: paired change from the base, pooled.
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for y, (metric, label) in enumerate(reversed(METRICS)):
        for (who, name), dy in zip(TRAINED, (0.15, -0.15)):
            d = next(v for v in vs if v["reviewer"] == who and v["metric"] == metric)
            ax.errorbar(d["diff"] * 100, y + dy, xerr=[[(d["diff"] - d["ci_lo"]) * 100], [(d["ci_hi"] - d["diff"]) * 100]],
                        fmt="o", color=COLOR[who], ecolor=COLOR[who], elinewidth=2, capsize=3, markersize=6,
                        label=name if y == 0 else None)
            ax.text(d["ci_hi"] * 100 + 0.3, y + dy, f"{d['diff'] * 100:+.1f}", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(METRICS)), [label for _, label in reversed(METRICS)])
    ax.set_xlabel("Change from the untrained base on the same items (percentage points, all pools, 95% interval)")
    ax.set_title("Training raised acceptance of Jev's labels, not overrides of its mistakes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.5)
    save(fig, "fig1_vs_base.png")

    # Figure 2: switching to a wrong proposal the reviewer disagreed with alone.
    fig, ax = plt.subplots(figsize=(9, 3.4))
    conds = (("jev_label", "Jev's wrong label"), ("matched_label", "A random wrong label"))
    for y, (cond, label) in enumerate(reversed(conds)):
        for (who, name), dy in zip(REVIEWERS, (0.22, 0.0, -0.22)):
            d = next(v for v in switch if v["reviewer"] == who and v["proposal"] == cond)
            ax.errorbar(d["rate"] * 100, y + dy, xerr=[[(d["rate"] - d["ci_lo"]) * 100], [(d["ci_hi"] - d["rate"]) * 100]],
                        fmt="o", color=COLOR[who], ecolor=COLOR[who], elinewidth=2, capsize=3, markersize=6,
                        label=name if y == 0 else None)
            ax.text(d["ci_hi"] * 100 + 0.6, y + dy, f"{d['rate'] * 100:.0f}%", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(conds)), [label for _, label in reversed(conds)])
    ax.set_xlim(0, 45)
    ax.set_xlabel("Switched to the proposal, among items where the reviewer's own alone answer differed (%, 95% interval)")
    ax.set_title("After training, the reviewer gives up its own right answer to Jev more often")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    save(fig, "fig2_switching.png")
    print("wrote", sorted(p.name for p in (HERE / "data").iterdir()), sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
