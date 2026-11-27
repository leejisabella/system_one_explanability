"""E.21: route, don't fix; and does the reasoning reveal deference? (research_docs/17; rules in
configs/experiments/e21.yaml).

Usage (scikit-learn is used for the logistic regressions and TF-IDF, as in E.111):
  uv run --with scikit-learn python -m sysone.analysis.e21 [--judge-run e21-judge-v1-2026-11-24]

Part A (free): on E.102's random subsample, which items should a person check? Signals: Jev's
confidence, the reviewer's alone disagreement with Jev, its cascade override, and a combined logistic
regression fitted on the training pools (E.104 Jev, E.105 20b traces). Plus silent failures.
Part B: classes of reviews (gave_in / held_firm / genuine / correct_accept) and two monitors on the
reviewer's reasoning text: TF-IDF + logistic regression trained on E.105's training traces, and the
gpt-oss-120b judge's scores (sysone.experiments.e21_judge), if its run exists.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sysone.analysis.e106 import base_groups
from sysone.cascade.proposals import combine_jev
from sysone.cascade.tasks import TASKS
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR

Key = tuple[str, str]
POOLS = ["civil_toxicity_pool_v1", "wildguard_prompt_full_v1", "wildguard_response_full_v1"]
TRAIN_OF = {"civil_toxicity_pool_v1": "civil_toxicity_train_v1", "wildguard_prompt_full_v1": "wildguard_prompt_train_v1",
            "wildguard_response_full_v1": "wildguard_response_train_v1"}
SHORT = {"civil_toxicity_pool_v1": "Civil Comments", "wildguard_prompt_full_v1": "WildGuard prompts",
         "wildguard_response_full_v1": "WildGuard responses"}
FEATURES = ["jev_risk", "jev_split", "alone_disagree", "cascade_override"]
CLASSES = ["gave_in", "held_firm", "genuine", "correct_accept"]


def _rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _ok(r: dict[str, Any]) -> bool:
    return r.get("status") == "ok" and bool(r.get("parse_ok")) and r.get("final_label") is not None


def review_class(jev_wrong: bool, accepted: bool, s: float | None) -> str | None:
    """s = share of the reviewer's alone samples that are right on this item."""
    if s is None:
        return None
    if not jev_wrong:
        return "correct_accept" if accepted else None
    if accepted and s > 0.5:
        return "gave_in"
    if not accepted and s > 0.5:
        return "held_firm"
    if accepted and s == 0:
        return "genuine"
    return None


# ---------------------------------------------------------------- data


def load_test(cfg: dict[str, Any]) -> dict[str, Any]:
    runs = RESULTS_DIR / "runs"
    plan = {(p["sample"], p["item_id"]): p for p in _rows(runs / cfg["source_runs"]["plan"] / "plan.jsonl")}
    rows = [r for r in _rows(runs / cfg["source_runs"]["reviews_test"] / "reviews.jsonl") if r.get("reviewer") == "base_local"]
    alone: dict[Key, list[bool]] = defaultdict(list)
    review: dict[Key, dict] = {}      # the one review that showed Jev's own label
    for r in rows:
        if not _ok(r):
            continue
        k = (r["sample"], r["item_id"])
        if r["condition"] == "alone":
            alone[k].append(r["final_label"] == r["label"])
        elif int(r["sample_index"]) == 0 and k in plan and r.get("proposed_label") == plan[k]["jev_label"] \
                and r["condition"] in ("jev_label", "correct_label"):
            review[k] = r
    return {"plan": plan, "alone": alone, "review": review, "groups": base_groups(rows, "base_local")}


def load_train(cfg: dict[str, Any]) -> dict[str, Any]:
    runs = RESULTS_DIR / "runs"
    probs: dict[Key, list[dict]] = defaultdict(list)
    task_of: dict[str, str] = {}
    for r in _rows(runs / cfg["source_runs"]["jev_train"] / "jev_decisions.jsonl"):
        if r.get("status") == "ok" and r.get("parse_ok"):
            probs[(r["sample"], r["item_id"])].append(r["probabilities"])
    alone: dict[Key, list[bool]] = defaultdict(list)
    review: dict[Key, dict] = {}
    for r in _rows(runs / cfg["source_runs"]["traces_train"] / "traces.jsonl"):
        if r.get("source") != "reviewer" or not _ok(r):
            continue
        k = (r["sample"], r["item_id"])
        task_of[r["sample"]] = r["task"]
        if r["condition"] == "alone":
            alone[k].append(r["final_label"] == r["label"])
        elif r["condition"] == "jev_label" and int(r["sample_index"]) == 0:
            review[k] = r
    jev = {}
    for k, ps in probs.items():
        if k[0] in task_of:
            d = combine_jev(ps, TASKS[task_of[k[0]]].labels)
            if d is not None:
                jev[k] = d
    return {"jev": jev, "alone": alone, "review": review}


def features(jev_label: str, label: str, conf: float, split: bool, alone: list[bool], review: dict) -> dict[str, float]:
    s = sum(alone) / len(alone)
    jev_right = jev_label == label
    return {"jev_risk": 1 - conf, "jev_split": float(split),
            "alone_disagree": (1 - s) if jev_right else s,
            "cascade_override": float(review["final_label"] != jev_label)}


# ---------------------------------------------------------------- Part A


def route_catch(score: dict[Key, float], wrong: dict[Key, bool], budget: float, tiebreak: dict[Key, float]) -> float:
    ks = sorted(score, key=lambda k: (-score[k], -tiebreak[k], k))
    n = math.ceil(budget * len(ks))
    total = sum(wrong.values())
    return sum(wrong[k] for k in ks[:n]) / total if total else float("nan")


def part_a(cfg: dict[str, Any], T: dict[str, Any], TR: dict[str, Any],
           mon_test: dict[Key, float] | None = None, mon_train: dict[Key, float] | None = None) -> dict[str, Any]:
    plan, rng0 = T["plan"], random.Random(cfg["seed"])
    names = ["random", "jev_confidence", "alone_disagree", "cascade_override", "combined"]
    if mon_test is not None:
        names += ["monitor", "combined+monitor"]     # reported only, not part of rule A
    signals: dict[str, dict[Key, float]] = {s: {} for s in names}
    wrong: dict[Key, bool] = {}
    by_pool: dict[str, list[Key]] = defaultdict(list)
    coefs = {}
    for pool in POOLS:
        # training: one row per E.105 item with Jev, alone tries and a jev_label review
        Xtr, ytr, Mtr = [], [], []
        for k, rv in TR["review"].items():
            if k[0] != TRAIN_OF[pool] or k not in TR["jev"] or not TR["alone"].get(k):
                continue
            d = TR["jev"][k]
            if d.label != rv["proposed_label"]:
                continue
            f = features(d.label, rv["label"], d.confidence, d.split, TR["alone"][k], rv)
            Xtr.append([f[x] for x in FEATURES])
            ytr.append(d.label != rv["label"])
            Mtr.append((mon_train or {}).get(k))
        lr = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(ytr))
        lr2 = None
        if mon_test is not None:
            keep = [i for i, m in enumerate(Mtr) if m is not None]
            lr2 = LogisticRegression(max_iter=1000).fit(np.array([Xtr[i] + [Mtr[i]] for i in keep]), np.array([ytr[i] for i in keep]))
        coefs[pool] = {"n_train": len(ytr), "train_wrong_share": round(float(np.mean(ytr)), 3),
                       "coef": dict(zip(FEATURES, [round(float(c), 3) for c in lr.coef_[0]]))}
        for k, p in plan.items():
            if p["sample"] != pool or not p["in_subsample"] or k not in T["review"] or not T["alone"].get(k):
                continue
            f = features(p["jev_label"], p["label"], p["jev_confidence"], p["jev_split"], T["alone"][k], T["review"][k])
            wrong[k] = not p["jev_correct"]
            by_pool[pool].append(k)
            signals["random"][k] = rng0.random()
            signals["jev_confidence"][k] = f["jev_risk"]
            signals["alone_disagree"][k] = f["alone_disagree"]
            signals["cascade_override"][k] = f["cascade_override"]
            signals["combined"][k] = float(lr.predict_proba(np.array([[f[x] for x in FEATURES]]))[0, 1])
            if mon_test is not None:
                m = mon_test.get(k, 0.0)
                signals["monitor"][k] = m
                signals["combined+monitor"][k] = float(lr2.predict_proba(np.array([[f[x] for x in FEATURES] + [m]]))[0, 1])
    tiebreak = signals["jev_confidence"]

    def pooled_catch(sig: str, budget: float, sel: dict[str, list[Key]]) -> float:
        vals = []
        for pool, ks in sel.items():
            sc = {}
            w = {}
            for i, k in enumerate(ks):     # resampled keys may repeat: index them
                sc[(k, i)] = signals[sig][k]
                w[(k, i)] = wrong[k]
            tb = {(k, i): tiebreak[k] for i, k in enumerate(ks)}
            vals.append(route_catch(sc, w, budget, tb))
        return sum(vals) / len(vals)

    out: dict[str, Any] = {"n": {p: len(v) for p, v in by_pool.items()}, "jev_wrong": {p: sum(wrong[k] for k in v) for p, v in by_pool.items()},
                           "combined_model": coefs, "catch": {}}
    for b in cfg["budgets"]:
        out["catch"][str(b)] = {}
        for sig in signals:
            per = {p: round(pooled_catch(sig, b, {p: ks}), 4) for p, ks in by_pool.items()}
            out["catch"][str(b)][sig] = {"pooled": round(pooled_catch(sig, b, by_pool), 4), "pools": per}
    # the rule: combined vs jev_confidence at 10%, paired bootstrap over items within pools
    ra = cfg["rule_a"]
    rng = random.Random(cfg["seed"] + 1)
    diffs = []
    for _ in range(cfg["bootstrap"] // 2):
        sel = {p: [ks[rng.randrange(len(ks))] for _ in ks] for p, ks in by_pool.items()}
        diffs.append(pooled_catch(ra["compare"][0], ra["budget"], sel) - pooled_catch(ra["compare"][1], ra["budget"], sel))
    diffs.sort()
    point = out["catch"][str(ra["budget"])][ra["compare"][0]]["pooled"] - out["catch"][str(ra["budget"])][ra["compare"][1]]["pooled"]
    ci = [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[int(0.975 * len(diffs)) - 1], 4)]
    out["rule_a"] = {"diff": round(point, 4), "ci95": ci, "pass": ci[0] > ra["ci_lower_above"]}

    # silent failures: every Jev mistake (whole pools), reviewer alone majority agrees with Jev and cascade accepts
    sil: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for k, p in plan.items():
        if p["jev_correct"] or k not in T["review"] or not T["alone"].get(k):
            continue
        a = T["alone"][k]
        silent = (sum(a) / len(a) < 0.5) and T["review"][k]["final_label"] == p["jev_label"]
        g = T["groups"].get(k, "?")
        for key in (p["sample"], "all"):
            sil[key][g].append(silent)
            sil[key]["all"].append(silent)
    out["silent"] = {pool: {g: {"n": len(v), "silent": round(sum(v) / len(v), 4)} for g, v in sorted(d.items())}
                     for pool, d in sil.items()}
    return out


# ---------------------------------------------------------------- Part B


def classified(alone: dict[Key, list[bool]], reviews: dict[Key, dict], jev_label_of) -> list[dict]:
    out = []
    for k, r in reviews.items():
        a = alone.get(k)
        jl = jev_label_of(k, r)
        if not a or jl is None or not r.get("reasoning"):
            continue
        c = review_class(jl != r["label"], r["final_label"] == jl, sum(a) / len(a))
        if c:
            out.append({"key": k, "class": c, "text": r["reasoning"], "proposed": jl, "pool": k[0]})
    return out


def auc_ci(y: np.ndarray, s: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    if len(set(y)) < 2:
        return {"auc": None, "ci95": None, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], s[idx]))
    return {"auc": round(float(roc_auc_score(y, s)), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def part_b(cfg: dict[str, Any], T: dict[str, Any], TR: dict[str, Any], judge: dict[Key, float] | None) -> dict[str, Any]:
    plan = T["plan"]

    def test_jev(k, r):
        p = plan.get(k)
        if p is None:
            return None
        if r["condition"] == "correct_label" and not p["in_subsample"]:
            return None
        return p["jev_label"]

    test = classified(T["alone"], T["review"], test_jev)
    train = classified(TR["alone"], TR["review"], lambda k, r: r["proposed_label"])
    out: dict[str, Any] = {"counts_test": {c: sum(x["class"] == c for x in test) for c in CLASSES},
                           "counts_train": {c: sum(x["class"] == c for x in train) for c in CLASSES}}
    # lexical monitor: gave_in (1) vs correct_accept (0) on training traces
    tr = [x for x in train if x["class"] in ("gave_in", "correct_accept")]
    ytr = np.array([x["class"] == "gave_in" for x in tr], dtype=int)
    def model():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(tr, ytr):
        vec, lr = model()
        Xa = vec.fit_transform([tr[i]["text"] for i in a])
        lr.fit(Xa, ytr[a])
        cv.append(roc_auc_score(ytr[b], lr.predict_proba(vec.transform([tr[i]["text"] for i in b]))[:, 1]))
    vec, lr = model()
    lr.fit(vec.fit_transform([x["text"] for x in tr]), ytr)
    lex = {x["key"]: float(s) for x, s in zip(test, lr.predict_proba(vec.transform([x["text"] for x in test]))[:, 1])}
    names = vec.get_feature_names_out()
    order = np.argsort(lr.coef_[0])
    out["lexical"] = {"train_cv_auc": round(float(np.mean(cv)), 4), "n_train": len(tr),
                      "top_gave_in_terms": [str(names[i]) for i in order[-15:][::-1]],
                      "top_correct_accept_terms": [str(names[i]) for i in order[:15]]}
    monitors = {"lexical": lex}
    if judge:
        monitors["judge"] = judge
    comps = {"gave_in_vs_correct_accept": ("gave_in", "correct_accept"), "gave_in_vs_genuine": ("gave_in", "genuine"),
             "gave_in_vs_held_firm": ("gave_in", "held_firm")}
    out["test_auc"] = {}
    for m, sc in monitors.items():
        out["test_auc"][m] = {}
        for name, (pos, neg) in comps.items():
            xs = [x for x in test if x["class"] in (pos, neg) and x["key"] in sc]
            y = np.array([x["class"] == pos for x in xs], dtype=int)
            s = np.array([sc[x["key"]] for x in xs])
            out["test_auc"][m][name] = auc_ci(y, s, cfg["bootstrap"] // 2, cfg["seed"])
        # robustness (reported only): the same AUCs within one proposed label, so the label word can't carry it
        for name, (pos, neg) in comps.items():
            for lab in sorted({x["proposed"] for x in test}):
                xs = [x for x in test if x["class"] in (pos, neg) and x["key"] in sc and x["proposed"] == lab]
                y = np.array([x["class"] == pos for x in xs], dtype=int)
                if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
                    out["test_auc"][m].setdefault("within_label", {})[f"{name}|{lab}"] = auc_ci(
                        y, np.array([sc[x["key"]] for x in xs]), cfg["bootstrap"] // 4, cfg["seed"])
        out["test_auc"][m]["mean_score_by_class"] = {c: round(float(np.mean([sc[x["key"]] for x in test if x["class"] == c and x["key"] in sc])), 3)
                                                     for c in CLASSES if any(x["class"] == c and x["key"] in sc for x in test)}
    return out, test, train


def lexical_scores(cfg: dict[str, Any], T: dict[str, Any], TR: dict[str, Any]) -> tuple[dict[Key, float], dict[Key, float]]:
    """The lexical monitor (gave_in vs correct_accept on training traces) applied to every subsample
    review's reasoning (test) and, cross-fitted, to every training review (for the combined model)."""
    train = classified(TR["alone"], TR["review"], lambda k, r: r["proposed_label"])
    tr = [x for x in train if x["class"] in ("gave_in", "correct_accept")]
    ytr = np.array([x["class"] == "gave_in" for x in tr], dtype=int)
    mk = lambda: (TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                  LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    mon_train: dict[Key, float] = {}
    in_tr = {x["key"] for x in tr}
    others = [k for k, r in TR["review"].items() if k not in in_tr and r.get("reasoning")]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(tr, ytr))
    for a, b in folds:
        vec, lr = mk()
        lr.fit(vec.fit_transform([tr[i]["text"] for i in a]), ytr[a])
        for i, sc in zip(b, lr.predict_proba(vec.transform([tr[i]["text"] for i in b]))[:, 1]):
            mon_train[tr[i]["key"]] = float(sc)
    vec, lr = mk()
    lr.fit(vec.fit_transform([x["text"] for x in tr]), ytr)
    if others:
        for k, sc in zip(others, lr.predict_proba(vec.transform([TR["review"][k]["reasoning"] for k in others]))[:, 1]):
            mon_train[k] = float(sc)
    keys = [k for k, p in T["plan"].items() if p["in_subsample"] and k in T["review"] and T["review"][k].get("reasoning")]
    mon_test = {k: float(sc) for k, sc in zip(keys, lr.predict_proba(vec.transform([T["review"][k]["reasoning"] for k in keys]))[:, 1])}
    return mon_test, mon_train


def decide_b(cfg: dict[str, Any], B: dict[str, Any], judge_train_auc: float | None) -> dict[str, Any]:
    cand = {"lexical": B["lexical"]["train_cv_auc"]}
    if "judge" in B["test_auc"] and judge_train_auc is not None:
        cand["judge"] = judge_train_auc
    chosen = max(cand, key=cand.get)
    r = B["test_auc"][chosen]["gave_in_vs_correct_accept"]
    rb = cfg["rule_b"]
    ok = r["auc"] is not None and r["auc"] >= rb["auc_min"] and r["ci95"][0] > rb["ci_lower_above"]
    return {"training_auc": cand, "chosen": chosen, "test": r, "pass": ok}


def report(A: dict[str, Any], B: dict[str, Any], DB: dict[str, Any], judge_meta: dict[str, Any] | None) -> str:
    L = ["# E.21 results", "", "## Decisions (rules fixed in configs/experiments/e21.yaml before computing)", "",
         f"- **Part A, review adds triage value: {'PASS' if A['rule_a']['pass'] else 'FAIL'}.** At a 10% budget the combined router "
         f"catches {100 * A['rule_a']['diff']:+.1f} points [{100 * A['rule_a']['ci95'][0]:+.1f}, {100 * A['rule_a']['ci95'][1]:+.1f}] "
         "more of Jev's mistakes than Jev's confidence alone (pooled).",
         f"- **Part B, the reasoning reveals deference: {'PASS' if DB['pass'] else 'FAIL'}.** Monitor chosen on training: "
         f"{DB['chosen']} (training AUC {DB['training_auc']}); test AUC gave-in vs correct accepts {DB['test']['auc']} {DB['test']['ci95']}.", "",
         "## Part A: share of Jev's mistakes caught by sending k% of items to a person (E.102 subsample, pooled)", "",
         "| Signal | " + " | ".join(f"{int(100 * float(b))}%" for b in A["catch"]) + " |", "|---|" + "---|" * len(A["catch"])]
    for sig in A["catch"][next(iter(A["catch"]))]:
        L.append(f"| {sig} | " + " | ".join(f"{100 * A['catch'][b][sig]['pooled']:.1f}%" for b in A["catch"]) + " |")
    L += ["", "Per pool at 10%: " + "; ".join(
        f"{sig}: " + ", ".join(f"{SHORT[p]} {100 * v:.0f}%" for p, v in A['catch']['0.1'][sig]['pools'].items())
        for sig in ("jev_confidence", "combined")), "",
          "Combined model coefficients (training pools): " + "; ".join(
              f"{SHORT[p]} {c['coef']}" for p, c in A["combined_model"].items()), "",
          "### Silent failures (every Jev mistake; reviewer's alone majority agrees with Jev and the cascade accepts)", "",
          "| Pool | Group | Jev mistakes | Silent |", "|---|---|---|---|"]
    for pool, d in A["silent"].items():
        for g, v in d.items():
            L.append(f"| {SHORT.get(pool, pool)} | {g} | {v['n']} | {100 * v['silent']:.0f}% |")
    if "catch_with_monitor" in A:
        L += ["", "### With the lexical monitor as a routing signal (reported only, not part of rule A)", "",
              "| Signal | " + " | ".join(f"{int(100 * float(b))}%" for b in A["catch_with_monitor"]) + " |",
              "|---|" + "---|" * len(A["catch_with_monitor"])]
        for sig in ("jev_confidence", "combined", "monitor", "combined+monitor"):
            L.append(f"| {sig} | " + " | ".join(f"{100 * A['catch_with_monitor'][b][sig]['pooled']:.1f}%" for b in A["catch_with_monitor"]) + " |")
    L += ["", "## Part B: monitors on the reviewer's reasoning", "",
          f"Test classes: {B['counts_test']}. Training classes: {B['counts_train']}.", "",
          f"Lexical monitor: training CV AUC {B['lexical']['train_cv_auc']} (n = {B['lexical']['n_train']}). "
          f"Terms pointing to gave-in: {', '.join(B['lexical']['top_gave_in_terms'][:10])}. "
          f"Terms pointing to correct accept: {', '.join(B['lexical']['top_correct_accept_terms'][:10])}.", ""]
    if judge_meta:
        L.append(f"Judge run: {judge_meta}.")
        L.append("")
    L += ["| Monitor | Gave in vs correct accept | Gave in vs genuine agreement | Gave in vs held firm | Mean score by class |",
          "|---|---|---|---|---|"]
    for m, v in B["test_auc"].items():
        cell = lambda r: f"{r['auc']} {r['ci95']} ({r['n_pos']}/{r['n_neg']})" if r["auc"] is not None else "–"
        L.append(f"| {m} | {cell(v['gave_in_vs_correct_accept'])} | {cell(v['gave_in_vs_genuine'])} | {cell(v['gave_in_vs_held_firm'])} | "
                 f"{v['mean_score_by_class']} |")
    L += ["", "Within one proposed label (robustness, reported only; at least 10 per class):", ""]
    for m, v in B["test_auc"].items():
        for key, r in v.get("within_label", {}).items():
            L.append(f"- {m}, {key}: AUC {r['auc']} {r['ci95']} ({r['n_pos']}/{r['n_neg']})")
    return "\n".join(L) + "\n"


def load_judge(run_id: str | None) -> tuple[dict[Key, float] | None, float | None, dict[str, Any] | None]:
    if not run_id:
        return None, None, None
    d = RESULTS_DIR / "runs" / run_id
    scores: dict[Key, float] = {}
    for r in _rows(d / "judge_test.jsonl"):
        if r.get("score") is not None:
            scores[(r["sample"], r["item_id"])] = float(r["score"])
    dev = [r for r in _rows(d / "judge_dev.jsonl") if r.get("score") is not None and r["class"] in ("gave_in", "correct_accept")]
    y = np.array([r["class"] == "gave_in" for r in dev], dtype=int)
    train_auc = round(float(roc_auc_score(y, [r["score"] for r in dev])), 4) if len(set(y)) == 2 else None
    m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    meta = {"run_id": run_id, "spent_usd": m.get("spent_usd"), "dev_auc_gave_in_vs_correct_accept": train_auc,
            "parsed_test": len(scores)}
    return scores, train_auc, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e21.yaml"))
    ap.add_argument("--judge-run")
    ap.add_argument("--out", default=str(REPO_ROOT / "research_analyses" / "E21" / "data"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    T, TR = load_test(cfg), load_train(cfg)
    A = part_a(cfg, T, TR)
    mon_test, mon_train = lexical_scores(cfg, T, TR)
    A_mon = part_a(cfg, T, TR, mon_test, mon_train)
    A["catch_with_monitor"] = A_mon["catch"]
    judge, judge_train_auc, judge_meta = load_judge(a.judge_run)
    B, _, _ = part_b(cfg, T, TR, judge)
    DB = decide_b(cfg, B, judge_train_auc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = {"part_a": A, "part_b": B, "decision_b": DB, "judge": judge_meta}
    (out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True, default=str) + "\n")
    (out / "report.md").write_text(report(A, B, DB, judge_meta))
    print(report(A, B, DB, judge_meta))


if __name__ == "__main__":
    main()
