"""E.22 analysis: Mistral Large 3 vs gpt-oss-20b (and 120b) as the cascade's reviewer on WildGuard.

Usage:
  uv run python -m sysone.analysis.e22 results/runs/<e22 run>

Reuses sysone.analysis.e119.compare (paired on the same items; "120b" there means the new reviewer,
Mistral) and adds the decision rule's leniency guard (harmful let through) and the sensitivity check
without step 0's 30 read items.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e101 import _fmt
from sysone.analysis.e102 import _load
from sysone.analysis.e103 import _fmt_change, paired_change
from sysone.analysis.e119 import _index, compare
from sysone.registry import CONFIG_DIR, RESULTS_DIR

HARMFUL, UNHARMFUL = "harmful", "unharmful"
STEP0 = set("""wgr-0044 wgr-0166 wgr-0277 wgr-0309 wgr-0385 wgr-0907 wgr-1096 wgr-1377 wgr-1408 wgr-1458 wgr-1507
wgr-1530 wgr-1568 wgr-1610 wgr-1678 wgt-0045 wgt-0288 wgt-0819 wgt-0860 wgt-0907 wgt-1087 wgt-1101 wgt-1243
wgt-1456 wgt-1467 wgt-1530 wgt-1566 wgt-1576 wgt-1655 wgt-1679""".split())


def cascade_finals(plan: list[dict], reviews: list[dict]) -> dict[tuple[str, str], str]:
    """Per subsample item: the cascade's final label (review of Jev's own label)."""
    _, rv = _index(reviews)
    out = {}
    for p in plan:
        if not p["in_subsample"]:
            continue
        cond = "correct_label" if p["jev_correct"] else "jev_label"
        f = rv.get((p["sample"], p["item_id"], cond))
        if f is not None:
            out[(p["sample"], p["item_id"])] = f
    return out


def pooled_rate(flags: dict[tuple[str, str], bool]) -> float:
    pools = sorted({k[0] for k in flags})
    return sum(sum(v for k, v in flags.items() if k[0] == p) / sum(k[0] == p for k in flags) for p in pools) / len(pools)


def guard(plan: list[dict], finals: dict, keys: set, seed: int, reps: int) -> dict[str, Any]:
    """Harmful-let-through rate (gold harmful, final unharmful; share of all items), cascade vs Jev."""
    by = {(p["sample"], p["item_id"]): p for p in plan}
    jev = {k: by[k]["label"] == HARMFUL and by[k]["jev_label"] == UNHARMFUL for k in keys}
    cas = {k: by[k]["label"] == HARMFUL and finals[k] == UNHARMFUL for k in keys}
    return {"jev": round(pooled_rate(jev), 4), "cascade": round(pooled_rate(cas), 4),
            "increase": round(pooled_rate(cas) - pooled_rate(jev), 4),
            "change": paired_change({k: v for k, v in jev.items()}, cas, seed=seed)}


def analyze_run(run_dir: str | Path, source_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    plan = _load(run_dir / "plan.jsonl")
    pools = set(cfg["samples"])
    small = [r for r in _load(source_dir / "reviews.jsonl") if r["sample"] in pools]
    large = _load(run_dir / "reviews.jsonl")
    S = compare(plan, small, large, seed=cfg["seed"])
    keys_all = set(cascade_finals(plan, large)) & set(cascade_finals(plan, small))
    fin_m, fin_s = cascade_finals(plan, large), cascade_finals(plan, small)
    out: dict[str, Any] = {"compare": S}
    out["guard_mistral"] = guard(plan, fin_m, keys_all, cfg["seed"], cfg["bootstrap"])
    out["guard_20b"] = guard(plan, fin_s, keys_all, cfg["seed"], cfg["bootstrap"])
    # 120b cascade on the same items, for reference
    e119 = RESULTS_DIR / "runs" / "e119-v1-2026-11-24" / "reviews.jsonl"
    if e119.exists():
        big = [r for r in _load(e119) if r["sample"] in pools]
        fin_b = cascade_finals(plan, big)
        ks = keys_all & set(fin_b)
        lab = {(p["sample"], p["item_id"]): p["label"] for p in plan}
        out["mistral_minus_120b_cascade"] = paired_change({k: fin_b[k] == lab[k] for k in ks}, {k: fin_m[k] == lab[k] for k in ks},
                                                          seed=cfg["seed"])
    # sensitivity: without step 0's read items
    plan_ns = [p for p in plan if p["item_id"] not in STEP0]
    S_ns = compare(plan_ns, small, large, seed=cfg["seed"])
    out["without_step0"] = {"cascade_minus_jev": S_ns["pooled"]["cascade_120b_minus_jev"], "n_dropped_in_subsample":
                            sum(p["in_subsample"] and p["item_id"] in STEP0 for p in plan)}
    rows = [r for r in large if r.get("status") == "ok"]
    out["parse_rate"] = round(sum(bool(r.get("parse_ok")) for r in rows) / max(1, len(rows)), 4)
    out["n_reviews"] = len(rows)
    d = S["pooled"]["cascade_120b_minus_jev"]
    r = cfg["rule"]
    c1 = d["ci95"] is not None and d["ci95"][0] > r["ci_lower_above"]
    c2 = out["guard_mistral"]["increase"] <= r["max_let_through_increase"]
    out["decision"] = {"cascade_beats_jev": c1, "leniency_guard_ok": c2, "another_family_fixes_it": c1 and c2}
    man = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    (run_dir / "comparison.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    (run_dir / "comparison.md").write_text(report(out, man))
    return out


SHORT = {"wildguard_prompt_full_v1": "WildGuard prompts", "wildguard_response_full_v1": "WildGuard responses"}


def report(o: dict[str, Any], man: dict[str, Any]) -> str:
    S, d = o["compare"], o["decision"]
    L = [f"# E.22 comparison: {man.get('run_id', '')}", "",
         f"Status: {man.get('status')}. Spent ${man.get('spent_usd', 0):.4f}. Reviews {o['n_reviews']}, parsed {100 * o['parse_rate']:.1f}%.", "",
         "## Decision (rule fixed before the run)", "",
         f"- Mistral cascade − Jev alone, WildGuard subsample pooled: **{_fmt_change(S['pooled']['cascade_120b_minus_jev'])}** "
         f"(beats Jev: {d['cascade_beats_jev']}).",
         f"- Harmful let through: Jev {100 * o['guard_mistral']['jev']:.1f}%, Mistral cascade {100 * o['guard_mistral']['cascade']:.1f}% "
         f"(increase {100 * o['guard_mistral']['increase']:+.1f} points; guard ok: {d['leniency_guard_ok']}). "
         f"For reference, 20b cascade {100 * o['guard_20b']['cascade']:.1f}%.",
         f"- **Another family fixes it on WildGuard: {'YES' if d['another_family_fixes_it'] else 'NO'}.**", "",
         "## Accuracy on the WildGuard subsample (paired)", "",
         "| Scope | Items | Jev alone | 20b alone | Mistral alone | 20b cascade | Mistral cascade | Mistral cascade − Jev | Mistral − 20b cascade | Mistral − 20b alone |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for scope, X in [*S["pools"].items(), ("pooled", S["pooled"])]:
        L.append(f"| {SHORT.get(scope, scope)} | {X['n_subsample']} | {_fmt(X['jev_alone'])} | {_fmt(X['alone_20b'])} | {_fmt(X['alone_120b'])} | "
                 f"{_fmt(X['cascade_20b'])} | {_fmt(X['cascade_120b'])} | {_fmt_change(X['cascade_120b_minus_jev'])} | "
                 f"{_fmt_change(X['cascade_120b_minus_20b'])} | {_fmt_change(X['alone_120b_minus_20b'])} |")
    P = S["pooled"]
    L += ["", f"Error overlap (phi, pooled): Jev–20b {P['phi_with_jev']['20b']}, Jev–Mistral {P['phi_with_jev']['120b']}, 20b–Mistral {P['phi_20b_120b']}.", ""]
    if "mistral_minus_120b_cascade" in o:
        L.append(f"Mistral cascade − 120b cascade (E.119), same items: {_fmt_change(o['mistral_minus_120b_cascade'])}.")
    L += [f"Without step 0's read items ({o['without_step0']['n_dropped_in_subsample']} in the subsample): Mistral cascade − Jev "
          f"{_fmt_change(o['without_step0']['cascade_minus_jev'])}.", "",
          "## Jev's wrong labels, by the 20b's alone record (pooled)", "",
          "| 20b record | 20b overrides | Mistral overrides | Change (paired) | Mistral right alone |", "|---|---|---|---|---|"]
    for b, v in P["jev_wrong_by_20b_record"].items():
        L.append(f"| {b} | {_fmt(v['override_20b'])} | {_fmt(v['override_120b'])} | {_fmt_change(v['change'])} | {_fmt(v['right_alone_120b'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e22.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    analyze_run(a.run_dir, RESULTS_DIR / "runs" / cfg["source_run"], cfg)
    print((Path(a.run_dir) / "comparison.md").read_text())


if __name__ == "__main__":
    main()
