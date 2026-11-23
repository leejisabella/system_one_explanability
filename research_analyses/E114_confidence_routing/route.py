"""E.114 (free, no API, no GPU): route only Jev's low-confidence labels to the reviewer.

Plan fixed in README.md before this was run. Cut points: percentiles of Jev's confidence on E.104's
training pools. Outcomes: E.106's existing reviews (jev_label on Jev's wrong items, correct_label on
Jev's right items), scored on E.102's 1,200-item random subsample, paired against Jev alone.

    uv run python research_analyses/E114_confidence_routing/route.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sysone.analysis.e102 import _load
from sysone.cascade.items import load_items
from sysone.cascade.proposals import JEV_LABEL, combine_jev
from sysone.cascade.tasks import TASKS
from sysone.experiments.e102 import CORRECT_LABEL
from sysone.registry import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
E104 = RUNS / "e104-v1-2026-11-04" / "jev_decisions.jsonl"
PLAN = RUNS / "e102-v1-2026-11-04" / "plan.jsonl"
E106 = RUNS / "e106-v1-2026-11-04" / "reviews.jsonl"
POOLS = {  # test pool -> train pool
    "civil_toxicity_pool_v1": "civil_toxicity_train_v1",
    "wildguard_prompt_full_v1": "wildguard_prompt_train_v1",
    "wildguard_response_full_v1": "wildguard_response_train_v1",
}
QS = [0, 10, 25, 50, 100]
REVIEWERS = ["base_local", "sft_review", "sft_plain", "dpo_review"]
SEED, N_BOOT = 114, 2000
OUT = Path(__file__).resolve().parent / "results.json"


def train_confidences(sample: str) -> np.ndarray:
    """Jev's combined confidence on each training item (same combination as E.102's plan and E.111)."""
    items = load_items(sample)[0]
    labels = TASKS[items[0].task].labels
    parsed = defaultdict(list)
    for r in _load(E104):
        if r["sample"] == sample and r["status"] == "ok" and r["parse_ok"]:
            parsed[r["item_id"]].append(r["probabilities"])
    conf = [d.confidence for i in items if (d := combine_jev(parsed.get(i.item_id, []), labels)) is not None]
    return np.array(conf)


def cuts() -> dict[str, dict[int, float]]:
    out = {}
    for test_s, train_s in POOLS.items():
        c = train_confidences(train_s)
        out[test_s] = {q: (-np.inf if q == 0 else np.inf if q == 100 else float(np.percentile(c, q))) for q in QS}
    return out


def outcomes(reviews: list[dict], plan: dict, reviewer: str) -> dict:
    """Per item: the reviewer's final label in the cascade condition E.106 used for that item."""
    got = {(r["sample"], r["item_id"], r["condition"]): r for r in reviews if r["reviewer"] == reviewer}
    out = {}
    for key, p in plan.items():
        cond = CORRECT_LABEL if p["jev_correct"] else JEV_LABEL
        r = got.get((*key, cond))
        if r is not None:
            out[key] = r["final_label"]
    return out


def boot_ci(diff: np.ndarray, rng) -> list[float]:
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
    return [round(100 * float(np.percentile(b, q)), 2) for q in (2.5, 97.5)]


def policy(rows: list[dict], final: dict, cut: dict[str, dict[int, float]], q: int, rng) -> dict:
    routed = np.array([r["jev_confidence"] < cut[r["sample"]][q] for r in rows])
    jev_ok = np.array([r["jev_correct"] for r in rows])
    rev_ok = np.array([final[(r["sample"], r["item_id"])] == r["label"] for r in rows])
    casc_ok = np.where(routed, rev_ok, jev_ok)
    diff = casc_ok.astype(float) - jev_ok.astype(float)
    harmful = np.array([r["label"] == "harmful" for r in rows])
    casc_label = [final[(r["sample"], r["item_id"])] if m else r["jev_label"] for r, m in zip(rows, routed)]
    return {
        "q": q, "n": len(rows), "reviewed": int(routed.sum()), "reviewed_share": round(float(routed.mean()), 4),
        "jev_acc": round(float(jev_ok.mean()), 4), "cascade_acc": round(float(casc_ok.mean()), 4),
        "change_pts": round(100 * float(diff.mean()), 2), "ci95_pts": boot_ci(diff, rng),
        "fixed": int((routed & ~jev_ok & rev_ok).sum()), "broken": int((routed & jev_ok & ~rev_ok).sum()),
        "jev_wrong_routed": int((routed & ~jev_ok).sum()), "jev_wrong": int((~jev_ok).sum()),
        "harmful_let_through": {"n_harmful": int(harmful.sum()),
                                "jev": int(sum(h and r["jev_label"] == "unharmful" for h, r in zip(harmful, rows))),
                                "cascade": int(sum(h and l == "unharmful" for h, l in zip(harmful, casc_label)))},
    }


def two_part(plan_all: list[dict], final: dict, cut, q: int) -> dict:
    """Precision check: wrong items from the whole test pool, right items from the subsample."""
    per = {}
    for s in POOLS:
        rows = [p for p in plan_all if p["sample"] == s]
        e = np.mean([not p["jev_correct"] for p in rows])
        wrong = [p for p in rows if not p["jev_correct"] and (p["sample"], p["item_id"]) in final]
        right = [p for p in rows if p["jev_correct"] and p["in_subsample"] and (p["sample"], p["item_id"]) in final]
        fix = np.mean([p["jev_confidence"] < cut[s][q] and final[(s, p["item_id"])] == p["label"] for p in wrong])
        brk = np.mean([p["jev_confidence"] < cut[s][q] and final[(s, p["item_id"])] != p["label"] for p in right])
        per[s] = {"jev_err": round(float(e), 4), "n_wrong": len(wrong), "n_right": len(right),
                  "gain_pts": round(100 * float(e * fix - (1 - e) * brk), 2)}
    per["pooled_gain_pts"] = round(float(np.mean([v["gain_pts"] for v in per.values()])), 2)
    return per


def main() -> None:
    plan_all = list(_load(PLAN))
    plan = {(p["sample"], p["item_id"]): p for p in plan_all if p["in_subsample"]}
    reviews = [r for r in _load(E106) if r["status"] == "ok" and r["parse_ok"]]
    cut = cuts()
    out = {"cuts": {s: {str(q): v for q, v in c.items() if q not in (0, 100)} for s, c in cut.items()}, "reviewers": {}}
    for reviewer in REVIEWERS:
        rng = np.random.default_rng(SEED)
        final_sub = outcomes(reviews, plan, reviewer)
        missing = len(plan) - len(final_sub)
        rows = [p for k, p in plan.items() if k in final_sub]
        final_all = outcomes(reviews, {(p["sample"], p["item_id"]): p for p in plan_all}, reviewer)
        R = {"missing_subsample_outcomes": missing, "pooled": [], "by_pool": {}, "two_part": {}}
        for q in QS:
            R["pooled"].append(policy(rows, final_sub, cut, q, rng))
            R["two_part"][str(q)] = two_part(plan_all, final_all, cut, q)
        for s in POOLS:
            R["by_pool"][s] = [policy([r for r in rows if r["sample"] == s], final_sub, cut, q, rng) for q in QS]
        out["reviewers"][reviewer] = R
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for reviewer, R in out["reviewers"].items():
        print(f"\n== {reviewer} (missing {R['missing_subsample_outcomes']})")
        for P in R["pooled"]:
            print(f"q={P['q']:>3} reviewed {P['reviewed_share']:.1%} cascade {P['cascade_acc']:.1%} vs jev {P['jev_acc']:.1%} "
                  f"change {P['change_pts']:+.2f} {P['ci95_pts']} fixed {P['fixed']} broken {P['broken']} "
                  f"wrong routed {P['jev_wrong_routed']}/{P['jev_wrong']} harmful {P['harmful_let_through']} "
                  f"| two-part {R['two_part'][str(P['q'])]['pooled_gain_pts']:+.2f}")
    print("\ncuts", json.dumps(out["cuts"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
