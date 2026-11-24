"""E.119 analysis: gpt-oss-120b vs. gpt-oss-20b as the cascade's reviewer, on the same items.

Usage:
  uv run python -m sysone.analysis.e119 results/runs/<e119 run> [results/runs/<E.102 run>]

All comparisons are paired on the same items (the 20b's reviews are E.102's own):
- On E.102's random subsample (400 per pool): Jev alone, each reviewer alone (majority of its alone
  samples) and each cascade (Jev's label reviewed). **The decision rule:** the 120b cascade beats
  Jev alone, pooled, with the paired 95% interval above 0.
- On Jev's wrong labels (all of them): override by each reviewer, split by the 20b's alone record
  (never / sometimes / always right in its 5 samples). "Never" is where the shared blind spots are.
- Error overlap with Jev on the subsample (phi of the error indicators).
- Deference at equal difficulty, each reviewer by its own alone record (E.102's measure, from each
  run's summary.json).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sysone.analysis.e101 import _fmt, rate
from sysone.analysis.e102 import _fmt_std, _load, phi
from sysone.analysis.e103 import _fmt_change, paired_change

ALONE, JEV_LABEL, CORRECT_LABEL = "alone", "jev_label", "correct_label"
SMALL, LARGE = "20b", "120b"


def _index(reviews: list[dict[str, Any]]) -> tuple[dict, dict]:
    """(sample, item) -> [alone right?], and (sample, item, condition) -> final label, for parsed reviewer rows."""
    alone: dict[tuple[str, str], list[bool]] = defaultdict(list)
    review: dict[tuple[str, str, str], str] = {}
    for r in reviews:
        if r["status"] != "ok" or not r["parse_ok"] or r.get("role", "reviewer") != "reviewer":
            continue
        if r["condition"] == ALONE:
            alone[(r["sample"], r["item_id"])].append(r["final_label"] == r["label"])
        elif r["sample_index"] == 0:
            review[(r["sample"], r["item_id"], r["condition"])] = r["final_label"]
    return alone, review


def _bin(rights: list[bool]) -> str | None:
    if not rights:
        return None
    k = sum(rights)
    return "never" if k == 0 else "always" if k == len(rights) else "sometimes"


def compare(plan: list[dict[str, Any]], small: list[dict[str, Any]], large: list[dict[str, Any]],
            *, seed: int = 0) -> dict[str, Any]:
    idx = {SMALL: _index(small), LARGE: _index(large)}
    pools = sorted({p["sample"] for p in plan})
    out: dict[str, Any] = {"pools": {}, "pooled": {}}
    for scope, keep in [*[(s, {s}) for s in pools], ("pooled", set(pools))]:
        rows = [p for p in plan if p["sample"] in keep]
        sub = [p for p in rows if p["in_subsample"]]
        jev: dict[tuple, bool] = {}
        cascade: dict[str, dict[tuple, bool]] = {SMALL: {}, LARGE: {}}
        alone: dict[str, dict[tuple, bool]] = {SMALL: {}, LARGE: {}}
        for p in sub:
            key = (p["sample"], p["item_id"])
            jev[key] = p["jev_correct"]
            for m, (al, rv) in idx.items():
                cond = CORRECT_LABEL if p["jev_correct"] else JEV_LABEL
                final = rv.get((*key, cond))
                if final is not None:
                    cascade[m][key] = final == p["label"]
                if al.get(key):
                    alone[m][key] = sum(al[key]) > len(al[key]) / 2
        both = set(cascade[SMALL]) & set(cascade[LARGE]) & set(alone[SMALL]) & set(alone[LARGE])
        pick = lambda d: {k: v for k, v in d.items() if k in both}
        S: dict[str, Any] = {
            "n_subsample": len(both),
            "jev_alone": rate(list(pick(jev).values())),
            **{f"alone_{m}": rate(list(pick(alone[m]).values())) for m in (SMALL, LARGE)},
            **{f"cascade_{m}": rate(list(pick(cascade[m]).values())) for m in (SMALL, LARGE)},
            "cascade_120b_minus_jev": paired_change(pick(jev), pick(cascade[LARGE]), seed=seed),
            "cascade_20b_minus_jev": paired_change(pick(jev), pick(cascade[SMALL]), seed=seed),
            "cascade_120b_minus_20b": paired_change(pick(cascade[SMALL]), pick(cascade[LARGE]), seed=seed),
            "alone_120b_minus_20b": paired_change(pick(alone[SMALL]), pick(alone[LARGE]), seed=seed),
        }
        keys = sorted(both)
        jev_wrong = [not jev[k] for k in keys]
        S["phi_with_jev"] = {m: phi([not alone[m][k] for k in keys], jev_wrong) for m in (SMALL, LARGE)}
        S["phi_20b_120b"] = phi([not alone[SMALL][k] for k in keys], [not alone[LARGE][k] for k in keys])

        # Jev's wrong labels: override by each reviewer, by the 20b's alone record.
        by_bin: dict[str, dict[str, Any]] = {}
        wrong = [p for p in rows if not p["jev_correct"]]
        for b in ("never", "sometimes", "always", "all"):
            ov: dict[str, dict[tuple, bool]] = {SMALL: {}, LARGE: {}}
            right_alone_large: list[bool] = []
            for p in wrong:
                key = (p["sample"], p["item_id"])
                small_bin = _bin(idx[SMALL][0].get(key, []))
                if small_bin is None or (b != "all" and small_bin != b):
                    continue
                f_s, f_l = idx[SMALL][1].get((*key, JEV_LABEL)), idx[LARGE][1].get((*key, JEV_LABEL))
                if f_s is None or f_l is None:
                    continue
                ov[SMALL][key] = f_s != p["jev_label"]
                ov[LARGE][key] = f_l != p["jev_label"]
                al = idx[LARGE][0].get(key)
                if al:
                    right_alone_large.append(sum(al) > len(al) / 2)
            by_bin[b] = {"override_20b": rate(list(ov[SMALL].values())), "override_120b": rate(list(ov[LARGE].values())),
                         "change": paired_change(ov[SMALL], ov[LARGE], seed=seed),
                         "right_alone_120b": rate(right_alone_large)}
        S["jev_wrong_by_20b_record"] = by_bin
        (out["pooled"] if scope == "pooled" else out["pools"].setdefault(scope, {})).update(S)
    return out


def decide(summary: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    d = summary["pooled"][rule.get("comparison", "cascade_120b_minus_jev")]
    lo = d["ci95"][0] if d["ci95"] else None
    fixes = lo is not None and lo > rule.get("ci_lower_above", 0.0)
    return {"rule": rule, "diff": d["diff"], "ci95": d["ci95"], "stronger_reviewer_fixes_it": fixes}


def report(summary: dict[str, Any], decision: dict[str, Any], manifest: dict[str, Any],
           deference: dict[str, Any]) -> str:
    lines = [f"# E.119 comparison: {manifest.get('run_id', '')}", "",
             f"Status: {manifest.get('status')}. Spent ${manifest.get('spent_usd', 0):.4f} "
             f"({manifest.get('spent_by_provider', {})}). Reviewer: gpt-oss-120b vs. E.102's gpt-oss-20b, same items, "
             "prompt and settings.", "",
             "## Decision (rule fixed before the run)", "",
             f"120b cascade minus Jev alone, pooled subsample: **{_fmt_change(summary['pooled']['cascade_120b_minus_jev'])}**. "
             f"A stronger reviewer fixes it: **{'yes' if decision['stronger_reviewer_fixes_it'] else 'no'}**.", "",
             "## Accuracy on E.102's random subsample (paired)", "",
             "| Scope | Items | Jev alone | 20b alone | 120b alone | 20b cascade | 120b cascade | 120b cascade − Jev | "
             "120b cascade − 20b cascade | 120b alone − 20b alone |", "|---|---|---|---|---|---|---|---|---|---|"]
    for scope, S in [*summary["pools"].items(), ("pooled", summary["pooled"])]:
        lines.append(f"| {scope} | {S['n_subsample']} | {_fmt(S['jev_alone'])} | {_fmt(S['alone_20b'])} | "
                     f"{_fmt(S['alone_120b'])} | {_fmt(S['cascade_20b'])} | {_fmt(S['cascade_120b'])} | "
                     f"{_fmt_change(S['cascade_120b_minus_jev'])} | {_fmt_change(S['cascade_120b_minus_20b'])} | "
                     f"{_fmt_change(S['alone_120b_minus_20b'])} |")
    lines += ["", "Error overlap on the subsample (phi of error indicators, alone majority): "
              + "; ".join(f"{scope}: Jev–20b {S['phi_with_jev']['20b']}, Jev–120b {S['phi_with_jev']['120b']}, "
                          f"20b–120b {S['phi_20b_120b']}"
                          for scope, S in [*summary["pools"].items(), ("pooled", summary["pooled"])]) + ".", "",
              "## Jev's wrong labels, by the 20b's alone record (5 samples)", "",
              "| Scope | 20b record | 20b overrides | 120b overrides | Change (paired) | 120b right alone |",
              "|---|---|---|---|---|---|"]
    for scope, S in [*summary["pools"].items(), ("pooled", summary["pooled"])]:
        for b, v in S["jev_wrong_by_20b_record"].items():
            lines.append(f"| {scope} | {b} | {_fmt(v['override_20b'])} | {_fmt(v['override_120b'])} | "
                         f"{_fmt_change(v['change'])} | {_fmt(v['right_alone_120b'])} |")
    lines += ["", "## Deference at equal difficulty (each reviewer by its own alone record; E.102's measure, pooled)", ""]
    for m, d in deference.items():
        lines.append(f"- {m}: " + "; ".join(f"{k} {_fmt_std(v)}" for k, v in d.items()))
    return "\n".join(lines) + "\n"


def analyze_run(run_dir: str | Path, source_dir: str | Path, config: Any) -> dict[str, Any]:
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    plan = _load(run_dir / "plan.jsonl")
    summary = compare(plan, _load(source_dir / "reviews.jsonl"), _load(run_dir / "reviews.jsonl"))
    decision = decide(summary, config.rule or {})
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    deference = {}
    for m, d in ((SMALL + " (E.102)", source_dir), (LARGE, run_dir)):
        f = d / "summary.json"
        if f.exists():
            deference[m] = json.loads(f.read_text()).get("pooled_difficulty_matched", {})
    (run_dir / "comparison.json").write_text(json.dumps({**summary, "decision": decision}, indent=2, sort_keys=True) + "\n")
    (run_dir / "comparison.md").write_text(report(summary, decision, manifest, deference))
    return {**summary, "decision": decision}


def main() -> None:
    from sysone.experiments.e119 import E119Config
    from sysone.registry import CONFIG_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("source_dir", nargs="?")
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e119.yaml"))
    args = parser.parse_args()
    config = E119Config.load(args.config)
    source = args.source_dir or str(Path(RESULTS_DIR) / "runs" / config.source_run)
    analyze_run(args.run_dir, source, config)
    print((Path(args.run_dir) / "comparison.md").read_text())


if __name__ == "__main__":
    main()
