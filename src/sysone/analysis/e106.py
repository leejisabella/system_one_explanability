"""E.106–E.108 analysis: did post-training make the reviewer robust to Jev's errors? (research_docs/14)

Usage:
  uv run python -m sysone.analysis.e106 results/runs/<eval run> results/runs/<E.102 run> results/runs/<E.103 run> \
      [--config configs/experiments/e106.yaml]

Per reviewer and test pool, on the same self-hosted stack as the base:
- Jev's wrong labels overridden and Jev's right labels accepted (the trade-off E.103 showed must be
  reported together), cascade accuracy and the reviewer's own accuracy alone (subsample).
- Deference at equal difficulty (E.102's measure), with difficulty from the base reviewer's alone
  record on this stack, and Jev's wrong labels accepted on items the base never gets right alone
  (the shared blind spots), overall and by annotator agreement.
- Each held-out upstream model's wrong labels overridden.
- Paired changes against the base reviewer, and against the control arm (same items and labels,
  trained without proposals), on the same items.
Then the pre-registered success criteria (research_docs/14), each pass or fail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sysone.analysis.e101 import _fmt, rate
from sysone.analysis.e102 import STANDARDIZED, _fmt_std, _load, standardized_diff
from sysone.analysis.e103 import _fmt_change, paired_change
from sysone.cascade.proposals import ALONE, JEV_LABEL, MATCHED_LABEL
from sysone.experiments.e102 import CORRECT_LABEL
from sysone.experiments.e106 import UPSTREAM_WRONG

Key = tuple[str, str]


def _difficulty(rows: list[dict[str, Any]], reviewer: str) -> tuple[Counter, Counter]:
    k, n = Counter(), Counter()
    for r in rows:
        if r["reviewer"] == reviewer and r["condition"] == ALONE:
            key = (r["sample"], r["item_id"])
            n[key] += 1
            k[key] += r["final_label"] == r["label"]
    return k, n


def _reviewer_flags(rows: list[dict[str, Any]], plan: dict[Key, dict], upstream_right: dict[str, set[Key]],
                    k: Counter, n: Counter, full: int) -> dict[str, Any]:
    """Per-item outcomes for one reviewer, keyed (sample, item_id), so any two reviewers can be paired."""
    got = {(r["sample"], r["item_id"], r["condition"]): r for r in rows if r["condition"] != ALONE}
    first_alone = {(r["sample"], r["item_id"]): r for r in rows if r["condition"] == ALONE and r["sample_index"] == 0}
    accepted = lambda r: r["final_label"] == r["proposed_label"]
    f: dict[str, Any] = {"override_jev": {}, "override_matched": {}, "accept_right_jev": {}, "cascade": {}, "alone": {},
                         "harmful_passed": {}, "override_upstream": defaultdict(dict), "accept_right_upstream": defaultdict(dict),
                         "strata": {JEV_LABEL: [], MATCHED_LABEL: []}, "never_right_accepted": {}, "agreement": {}}
    for (s, i, cond), r in got.items():
        key = (s, i)
        p = plan.get(key)
        if p is None:
            continue
        if cond == JEV_LABEL:
            f["override_jev"][key] = not accepted(r)
            f["agreement"][key] = p["strata"].get("annotator_agreement", "unknown")
            if r["label"] == "harmful":
                f["harmful_passed"][key] = r["final_label"] == "unharmful"
        elif cond == MATCHED_LABEL:
            f["override_matched"][key] = not accepted(r)
        elif cond.startswith(f"{UPSTREAM_WRONG}:"):
            f["override_upstream"][cond.split(":", 1)[1]][key] = not accepted(r)
        if cond in (JEV_LABEL, MATCHED_LABEL) and n[key] == full:
            f["strata"][cond].append(((s, k[key], full), accepted(r)))
            if cond == JEV_LABEL and k[key] == 0:
                f["never_right_accepted"][key] = accepted(r)
    for key, p in plan.items():
        if not p["in_subsample"]:
            continue
        right = got.get((*key, CORRECT_LABEL))
        if right is not None:
            if p["jev_correct"]:
                f["accept_right_jev"][key] = accepted(right)
            for model, keys in upstream_right.items():
                if key in keys:
                    f["accept_right_upstream"][model][key] = accepted(right)
        r = right if p["jev_correct"] else got.get((*key, JEV_LABEL))
        if r is not None:
            f["cascade"][key] = r["final_label"] == r["label"]
        if key in first_alone:
            f["alone"][key] = first_alone[key]["final_label"] == first_alone[key]["label"]
    return f


def _by_sample(d: dict[Key, bool], sample: str) -> dict[Key, bool]:
    return {k: v for k, v in d.items() if k[0] == sample}


def summarize(plan_rows: list[dict], reviews: list[dict], upstream: list[dict], *, reviewers: list[str],
              difficulty_reviewer: str, upstreams: list[str], criteria: dict[str, Any]) -> dict[str, Any]:
    plan = {(p["sample"], p["item_id"]): p for p in plan_rows}
    ok = [r for r in reviews if r["status"] == "ok" and r["parse_ok"]]
    k, n = _difficulty(ok, difficulty_reviewer)
    full = max(n.values(), default=0)
    upstream_right: dict[str, set[Key]] = defaultdict(set)
    for r in upstream:
        if r["reviewer"] in upstreams and r["status"] == "ok" and r["parse_ok"] and r["final_label"] == r["label"]:
            upstream_right[r["reviewer"]].add((r["sample"], r["item_id"]))
    present = [a for a in reviewers if any(r["reviewer"] == a for r in ok)]
    flags = {a: _reviewer_flags([r for r in ok if r["reviewer"] == a], plan, upstream_right, k, n, full) for a in present}
    samples = sorted({p["sample"] for p in plan_rows})
    jev_alone = {key: p["jev_correct"] for key, p in plan.items() if p["in_subsample"]}

    out: dict[str, Any] = {"difficulty_reviewer": difficulty_reviewer, "reviewers": {}, "criteria": {}}
    for a in present:
        F = flags[a]
        R: dict[str, Any] = {"samples": {}, "pooled": {}}
        for s in [*samples, None]:
            pick = (lambda d: _by_sample(d, s)) if s else (lambda d: d)
            strata = {c: [x for x in F["strata"][c] if s is None or x[0][0] == s] for c in F["strata"]}
            S = {
                "override_when_wrong_jev": rate(list(pick(F["override_jev"]).values())),
                "override_when_wrong_matched": rate(list(pick(F["override_matched"]).values())),
                "accept_when_right_jev": rate(list(pick(F["accept_right_jev"]).values())),
                "cascade_accuracy_subsample": rate(list(pick(F["cascade"]).values())),
                "alone_accuracy_subsample": rate(list(pick(F["alone"]).values())),
                "jev_alone_accuracy_subsample": rate(list(pick(jev_alone).values())),
                "deference": {name: standardized_diff(strata[JEV_LABEL], strata[MATCHED_LABEL], keep, seed=1)
                              for name, keep in STANDARDIZED.items()},
                "jev_wrong_accepted_never_right": rate(list(pick(F["never_right_accepted"]).values())),
                "override_when_wrong_jev_by_agreement": {
                    g: rate([v for key, v in pick(F["override_jev"]).items() if F["agreement"].get(key) == g])
                    for g in sorted({F["agreement"][key] for key in pick(F["override_jev"])})},
                "harmful_let_through": rate(list(pick(F["harmful_passed"]).values())) if pick(F["harmful_passed"]) else None,
                "upstream": {m: {"override_when_wrong": rate(list(pick(F["override_upstream"][m]).values())),
                                 "accept_when_right": rate(list(pick(F["accept_right_upstream"][m]).values()))}
                             for m in upstreams},
            }
            if a != difficulty_reviewer and difficulty_reviewer in flags:
                B = flags[difficulty_reviewer]
                S["vs_base"] = {
                    "override_when_wrong_jev": paired_change(pick(B["override_jev"]), pick(F["override_jev"])),
                    "accept_when_right_jev": paired_change(pick(B["accept_right_jev"]), pick(F["accept_right_jev"])),
                    "cascade_accuracy": paired_change(pick(B["cascade"]), pick(F["cascade"])),
                    "alone_accuracy": paired_change(pick(B["alone"]), pick(F["alone"])),
                    "upstream": {m: {"override_when_wrong": paired_change(pick(B["override_upstream"][m]),
                                                                          pick(F["override_upstream"][m])),
                                     "accept_when_right": paired_change(pick(B["accept_right_upstream"][m]),
                                                                        pick(F["accept_right_upstream"][m]))}
                                 for m in upstreams},
                }
            control = (criteria.get("controls") or {}).get(a)
            if control in flags:
                C = flags[control]
                S["vs_control"] = {"control": control,
                                   "override_when_wrong_jev": paired_change(pick(C["override_jev"]), pick(F["override_jev"])),
                                   "accept_when_right_jev": paired_change(pick(C["accept_right_jev"]), pick(F["accept_right_jev"]))}
            # The better single model on these items, and the cascade's paired gain over it.
            best = max((("jev", jev_alone), ("reviewer_alone", F["alone"])),
                       key=lambda x: sum(pick(x[1]).values()) / max(1, len(pick(x[1]))))
            S["cascade_vs_best_single"] = {"best_single": best[0], **paired_change(pick(best[1]), pick(F["cascade"]))}
            if s:
                R["samples"][s] = S
            else:
                R["pooled"] = S
        out["reviewers"][a] = R
        if a != difficulty_reviewer and difficulty_reviewer in flags:
            out["criteria"][a] = evaluate_criteria(R, criteria, upstreams)
    return out


def _lower(d: dict[str, Any] | None) -> float | None:
    return d["ci95"][0] if d and d.get("ci95") else None


def evaluate_criteria(R: dict[str, Any], criteria: dict[str, Any], upstreams: list[str]) -> dict[str, Any]:
    """The pre-registered criteria (research_docs/14), each with the numbers it was decided on."""
    drop = float(criteria.get("max_accept_drop", 0.02))
    P = R["pooled"]
    vb = P["vs_base"]
    over, acc = vb["override_when_wrong_jev"], vb["accept_when_right_jev"]
    out: dict[str, Any] = {}
    out["1_trade_off"] = {
        "pass": (_lower(over) or 0) > 0 and acc["diff"] is not None and acc["diff"] >= -drop,
        "override_change": over, "accept_change": acc,
        "rule": f"pooled: override-when-wrong rises (95% interval above 0) and accept-when-right falls by at most {drop:.0%}"}
    bars = criteria.get("e103_best_override_change") or {}
    per = {s: {"change": S["vs_base"]["override_when_wrong_jev"]["diff"], "e103_best": bars.get(s)}
           for s, S in R["samples"].items()}
    out["2_beats_cheap_protocols"] = {
        "pass": bool(per) and all(v["e103_best"] is not None and v["change"] is not None and v["change"] > v["e103_best"]
                                  for v in per.values()),
        "per_sample": per, "rule": "in every pool, override-when-wrong rises by more than the best E.103 protocol's change"}
    cvb = P["cascade_vs_best_single"]
    out["3_cascade_beats_best_single"] = {
        "pass": (_lower(cvb) or 0) > 0, "change": cvb,
        "rule": "pooled: cascade accuracy beats the better of Jev alone and the reviewer alone (95% interval above 0)"}
    ups = {m: vb["upstream"][m] for m in upstreams}
    out["4_transfers_to_held_out_upstreams"] = {
        "pass": bool(ups) and all((_lower(u["override_when_wrong"]) or 0) > 0 and u["accept_when_right"]["diff"] is not None
                                  and u["accept_when_right"]["diff"] >= -drop for u in ups.values()),
        "per_upstream": ups, "rule": "pooled, for every held-out upstream model: criterion 1's rule"}
    vc = P.get("vs_control")
    out["5_beats_control_arm"] = {
        "pass": vc is not None and (_lower(vc["override_when_wrong_jev"]) or 0) > 0,
        "vs_control": vc, "rule": "pooled: overrides more of Jev's wrong labels than the arm trained on the same items "
                                  "without proposals (95% interval above 0)"}
    return out


def report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [f"# E.106 report: {manifest.get('run_id', '')}", "",
             f"Status: {manifest.get('status')}. Difficulty from `{summary['difficulty_reviewer']}`'s alone record. "
             "Paired changes are on the same items.", "", "## Pooled over the test pools", "",
             "| Reviewer | Jev's wrong overridden | Jev's right accepted | Cascade accuracy | Alone accuracy | "
             "Deference (right alone) | Jev's wrong accepted, never right alone |", "|---|---|---|---|---|---|---|"]
    for a, R in summary["reviewers"].items():
        P = R["pooled"]
        lines.append(f"| {a} | {_fmt(P['override_when_wrong_jev'])} | {_fmt(P['accept_when_right_jev'])} | "
                     f"{_fmt(P['cascade_accuracy_subsample'])} | {_fmt(P['alone_accuracy_subsample'])} | "
                     f"{_fmt_std(P['deference']['eligible'])} | {_fmt(P['jev_wrong_accepted_never_right'])} |")
    lines += ["", "| Reviewer vs. base (paired, pooled) | Jev's wrong overridden | Jev's right accepted | Cascade | Alone |",
              "|---|---|---|---|---|"]
    for a, R in summary["reviewers"].items():
        if "vs_base" in R["pooled"]:
            v = R["pooled"]["vs_base"]
            lines.append(f"| {a} | {_fmt_change(v['override_when_wrong_jev'])} | {_fmt_change(v['accept_when_right_jev'])} | "
                         f"{_fmt_change(v['cascade_accuracy'])} | {_fmt_change(v['alone_accuracy'])} |")
    for a, C in summary["criteria"].items():
        lines += ["", f"## Pre-registered criteria: {a}", "", "| Criterion | Pass | Rule |", "|---|---|---|"]
        lines += [f"| {name} | {'yes' if c['pass'] else 'no'} | {c['rule']} |" for name, c in C.items()]
    for a, R in summary["reviewers"].items():
        lines += ["", f"## {a} by pool", "",
                  "| Pool | Jev's wrong overridden | Jev's right accepted | Cascade | Harmful let through | Upstream wrong overridden |",
                  "|---|---|---|---|---|---|"]
        for s, S in R["samples"].items():
            ups = "; ".join(f"{m} {_fmt(u['override_when_wrong'])}" for m, u in S["upstream"].items())
            lines.append(f"| {s} | {_fmt(S['override_when_wrong_jev'])} | {_fmt(S['accept_when_right_jev'])} | "
                         f"{_fmt(S['cascade_accuracy_subsample'])} | {_fmt(S['harmful_let_through'])} | {ups} |")
    return "\n".join(lines) + "\n"


def analyze_run(run_dir: str | Path, source_dir: str | Path, upstream_dir: str | Path, config: Any) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary = summarize(_load(Path(source_dir) / "plan.jsonl"), _load(run_dir / "reviews.jsonl"),
                        _load(Path(upstream_dir) / "upstream.jsonl"), reviewers=config.reviewers,
                        difficulty_reviewer=config.difficulty_reviewer, upstreams=config.upstreams,
                        criteria=config.criteria)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(report(summary, manifest))
    return summary


def main() -> None:
    from sysone.experiments.e106 import E106Config
    from sysone.registry import CONFIG_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("source_dir")
    parser.add_argument("upstream_dir")
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e106.yaml"))
    args = parser.parse_args()
    analyze_run(args.run_dir, args.source_dir, args.upstream_dir, E106Config.load(args.config))
    print((Path(args.run_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
