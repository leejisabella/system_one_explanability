"""E.116 analysis: which prompt arm, if any, goes to E.117 (research_docs/16).

Usage:
  uv run python -m sysone.analysis.e116 results/runs/<e116 run> [--config configs/experiments/e116.yaml]

Per arm, paired against `standard` on the items both parsed (WildGuard pools pooled, and per pool):
- override of Jev's wrong labels, and split by direction: over-flagged (gold unharmful) and let
  through (gold harmful);
- accept of Jev's right labels;
- harmful content the cascade lets through, weighted back to the pools: gold-harmful items the
  cascade calls unharmful (Jev's let-through items accepted, Jev's right harmful labels overridden);
- WildGuard cascade accuracy, weighted back to the pools (each cell by its share of the pool).
Plus the parse rate and output tokens.

The selection rule (config `selection`, fixed before the run): among arms that keep accept-when-right
within `max_accept_right_drop`, don't lower override of Jev's let-through items by more than
`max_let_through_override_drop`, and parse at least `min_parse_rate` of replies, pick the largest
gain in override of Jev's wrong labels. It goes to E.117 only if that gain is at least `min_gain`
with its 95% interval above 0. Otherwise nothing does, and that is the track's result.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sysone.analysis.e101 import _fmt, rate
from sysone.analysis.e103 import _fmt_change, paired_change
from sysone.runs import JsonlWriter

STANDARD = "standard"
OVER_FLAGGED, LET_THROUGH, RIGHT = "over_flagged", "let_through", "right"
# Pool cells (from the run manifest's pool_counts) and how each review row maps to one.
CELLS = (OVER_FLAGGED, LET_THROUGH, "right_harmful", "right_unharmful")


def _cell(r: dict[str, Any]) -> str:
    return r["cell"] if r["cell"] != RIGHT else f"right_{r['label']}"


def _correct(r: dict[str, Any]) -> bool:
    return r["final_label"] == r["label"]


def weighted_metrics(rows: dict[tuple[str, str], dict[str, Any]], counts: dict[str, dict[str, int]]) -> dict[str, float | None]:
    """Cascade accuracy and harmful let through, each pool cell weighted by its count in the pool."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (sample, _), r in rows.items():
        by_cell[(sample, _cell(r))].append(r)
    acc_num = acc_den = lt_num = lt_den = 0.0
    for sample, cs in counts.items():
        for c in CELLS:
            got, n = by_cell.get((sample, c)), cs.get(c, 0)
            if not got or not n:
                continue
            ok = sum(_correct(r) for r in got) / len(got)
            acc_num += n * ok
            acc_den += n
            if c in (LET_THROUGH, "right_harmful"):  # gold harmful: let through when the cascade says unharmful
                lt_num += n * (1 - ok)
                lt_den += n
    return {"cascade_accuracy": acc_num / acc_den if acc_den else None,
            "harmful_let_through": lt_num / lt_den if lt_den else None}


def weighted_paired(a: dict, b: dict, counts: dict, key: str, *, reps: int, seed: int = 0) -> dict[str, Any]:
    """b - a for a weighted metric, resampling items within each pool cell (paired)."""
    keys = sorted(set(a) & set(b))
    point_a, point_b = weighted_metrics({k: a[k] for k in keys}, counts)[key], weighted_metrics({k: b[k] for k in keys}, counts)[key]
    if point_a is None or point_b is None:
        return {"diff": None, "ci95": None, "n": len(keys)}
    cells: dict[tuple[str, str], list] = defaultdict(list)
    for k in keys:
        cells[(k[0], _cell(a[k]))].append(k)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        pick = [(i, rng.choice(ks)) for ks in cells.values() for i in range(len(ks))]
        ra = {(k[0], f"{k[1]}#{i}"): a[k] for i, k in pick}
        rb = {(k[0], f"{k[1]}#{i}"): b[k] for i, k in pick}
        x, y = weighted_metrics(ra, counts)[key], weighted_metrics(rb, counts)[key]
        if x is not None and y is not None:
            diffs.append(y - x)
    diffs.sort()
    ci = [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[int(0.975 * len(diffs)) - 1], 4)] if diffs else None
    return {"diff": round(point_b - point_a, 4), "ci95": ci, "n": len(keys)}


def summarize(rows: list[dict[str, Any]], counts: dict[str, dict[str, int]], *, reps: int = 2000) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    arms = [STANDARD, *sorted({r["arm"] for r in ok} - {STANDARD})]
    parsed = {arm: {(r["sample"], r["item_id"]): r for r in ok if r["arm"] == arm and r["parse_ok"]} for arm in arms}
    samples = sorted({r["sample"] for r in ok})
    out: dict[str, Any] = {"arms": arms, "pooled": {}, "by_pool": {s: {} for s in samples}}

    def flags(arm_rows: dict, keep, val) -> dict[tuple[str, str], bool]:
        return {k: val(r) for k, r in arm_rows.items() if keep(r)}

    override = lambda r: r["final_label"] != r["proposed_label"]
    accept = lambda r: r["final_label"] == r["proposed_label"]
    measures = {
        "override_wrong": (lambda r: r["cell"] != RIGHT, override),
        "override_over_flagged": (lambda r: r["cell"] == OVER_FLAGGED, override),
        "override_let_through": (lambda r: r["cell"] == LET_THROUGH, override),
        "accept_right": (lambda r: r["cell"] == RIGHT, accept),
    }
    for scope, pools in [("pooled", samples), *[(s, [s]) for s in samples]]:
        target = out["pooled"] if scope == "pooled" else out["by_pool"][scope]
        sub = {arm: {k: r for k, r in parsed[arm].items() if k[0] in pools} for arm in arms}
        sub_counts = {s: counts[s] for s in pools if s in counts}
        for arm in arms:
            all_rows = [r for r in ok if r["arm"] == arm and r["sample"] in pools]
            A: dict[str, Any] = {name: rate(list(flags(sub[arm], keep, val).values())) for name, (keep, val) in measures.items()}
            A.update({k: (round(v, 4) if v is not None else None) for k, v in weighted_metrics(sub[arm], sub_counts).items()})
            A["parse_rate"] = rate([bool(r["parse_ok"]) for r in all_rows])
            A["mean_output_tokens"] = round(statistics.mean(r["output_tokens"] for r in all_rows), 1) if all_rows else None
            if arm != STANDARD:
                A["vs_standard"] = {
                    **{name: paired_change(flags(sub[STANDARD], keep, val), flags(sub[arm], keep, val))
                       for name, (keep, val) in measures.items()},
                    **{key: weighted_paired(sub[STANDARD], sub[arm], sub_counts, key, reps=reps)
                       for key in ("cascade_accuracy", "harmful_let_through")},
                }
            target[arm] = A
    return out


def select_arm(summary: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered selection rule to the pooled WildGuard results."""
    verdicts = {}
    for arm, A in summary["pooled"].items():
        if arm == STANDARD:
            continue
        v = A["vs_standard"]
        gain, ar, lt = v["override_wrong"], v["accept_right"], v["override_let_through"]
        checks = {
            "accept_right": ar["diff"] is not None and ar["diff"] >= -rule["max_accept_right_drop"],
            "let_through_override": lt["diff"] is not None and lt["diff"] >= -rule["max_let_through_override_drop"],
            "parse": (A["parse_rate"]["rate"] or 0) >= rule["min_parse_rate"],
        }
        verdicts[arm] = {"gain": gain["diff"], "ci95": gain["ci95"], "checks": checks, "admissible": all(checks.values())}
    admissible = {a: v for a, v in verdicts.items() if v["admissible"] and v["gain"] is not None}
    best = max(admissible, key=lambda a: admissible[a]["gain"], default=None)
    chosen = None
    if best is not None:
        b = admissible[best]
        if b["gain"] >= rule["min_gain"] and b["ci95"] is not None and b["ci95"][0] > 0:
            chosen = best
    return {"verdicts": verdicts, "best_admissible": best, "chosen": chosen, "rule": rule}


def _p(x: float | None) -> str:
    return "–" if x is None else f"{x:.1%}"


def report(summary: dict[str, Any], selection: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [f"# E.116 report: {manifest.get('run_id', '')}", "",
             f"Status: {manifest.get('status')}. Spent ${manifest.get('spent_usd', 0):.4f} "
             f"({manifest.get('spent_by_provider', {})}). Items: {manifest.get('selection', {}).get('counts')}.", "",
             "Development items only (WildGuard training pools, GPT-4 labels). Paired against `standard` on items both "
             "arms parsed. Cascade accuracy and harmful let through are weighted back to the pools "
             f"({manifest.get('pool_counts')}).", ""]
    for scope, block in [("WildGuard pooled", summary["pooled"]), *summary["by_pool"].items()]:
        lines += [f"## {scope}", "",
                  "| Arm | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | Cascade accuracy | "
                  "Harmful let through | Parsed | Out tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for arm, A in block.items():
            lines.append(f"| {arm} | {_fmt(A['override_wrong'])} | {_fmt(A['override_over_flagged'])} | "
                         f"{_fmt(A['override_let_through'])} | {_fmt(A['accept_right'])} | {_p(A['cascade_accuracy'])} | "
                         f"{_p(A['harmful_let_through'])} | {_fmt(A['parse_rate'])} | {A['mean_output_tokens']} |")
        lines += ["", "| Arm vs. standard (paired) | Override Jev's wrong | Over-flagged | Let through | Accept Jev's right | "
                  "Cascade accuracy | Harmful let through |", "|---|---|---|---|---|---|---|"]
        for arm, A in block.items():
            if "vs_standard" in A:
                v = A["vs_standard"]
                lines.append(f"| {arm} | " + " | ".join(_fmt_change(v[k]) for k in (
                    "override_wrong", "override_over_flagged", "override_let_through", "accept_right",
                    "cascade_accuracy", "harmful_let_through")) + " |")
        lines.append("")
    lines += ["## Selection (rule fixed before the run)", "", f"Rule: {selection['rule']}", "",
              "| Arm | Gain in override [95%] | Accept-right ok | Let-through ok | Parse ok | Admissible |", "|---|---|---|---|---|---|"]
    for arm, v in selection["verdicts"].items():
        ci = f" [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]" if v["ci95"] else ""
        c = v["checks"]
        gain = "–" if v["gain"] is None else f"{v['gain']:+.1%}"
        lines.append(f"| {arm} | {gain}{ci} | {c['accept_right']} | "
                     f"{c['let_through_override']} | {c['parse']} | {v['admissible']} |")
    lines += ["", f"**Goes to E.117: {selection['chosen'] or 'nothing'}**"
              + ("" if selection["chosen"] else f" (best admissible: {selection['best_admissible']})") + "."]
    return "\n".join(lines) + "\n"


def analyze_run(run_dir: str | Path, config: Any) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    rows = JsonlWriter(run_dir / "reviews.jsonl").rows()
    summary = summarize(rows, manifest.get("pool_counts", {}), reps=config.bootstrap)
    selection = select_arm(summary, config.selection)
    (run_dir / "summary.json").write_text(json.dumps({**summary, "selection": selection}, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(report(summary, selection, manifest))
    return {**summary, "selection": selection}


def main() -> None:
    from sysone.experiments.e116 import E116Config
    from sysone.registry import CONFIG_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e116.yaml"))
    args = parser.parse_args()
    analyze_run(args.run_dir, E116Config.load(args.config))
    print((Path(args.run_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
