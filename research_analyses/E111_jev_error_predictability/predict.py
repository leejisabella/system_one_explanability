"""E.111 (free check, no API, no GPU): how predictable are Jev's mistakes?

Fits a "Jev is wrong" classifier on E.104's training pools (Jev's decisions + gold labels) and
scores it on E.102's frozen test pools, split by round 2's groups A/B/C (E.106 base, alone).

Two feature sets, per pool:
  conf  Jev's own signals only: mean probability of its label, split across samples, which label it chose
  text  conf + TF-IDF of the item text (word 1-2 grams)

Reported per pool: AUC for "Jev wrong" (all test items, and each group's mistakes vs. Jev's right
items), and a flip rule: flip Jev's label when p(wrong) > t, with t chosen by 5-fold CV on the
training pool only, then frozen. Accuracy change vs. Jev alone with a 95% bootstrap interval.

    uv run --with scikit-learn python research_analyses/E111_jev_error_predictability/predict.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sysone.analysis.e102 import _load
from sysone.analysis.e106 import base_groups
from sysone.cascade.items import load_items
from sysone.cascade.proposals import combine_jev
from sysone.cascade.tasks import TASKS
from sysone.registry import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
E104 = RUNS / "e104-v1-2026-11-04" / "jev_decisions.jsonl"
PLAN = RUNS / "e102-v1-2026-11-04" / "plan.jsonl"
E106 = RUNS / "e106-v1-2026-11-04" / "reviews.jsonl"
POOLS = {  # train pool -> test pool
    "civil_toxicity_train_v1": "civil_toxicity_pool_v1",
    "wildguard_prompt_train_v1": "wildguard_prompt_full_v1",
    "wildguard_response_train_v1": "wildguard_response_full_v1",
}
OUT = Path(__file__).resolve().parent / "results.json"
SEED = 111


def train_rows(sample: str) -> list[dict]:
    """One row per training item: Jev's combined decision (same rule as E.102's plan) and the gold label."""
    items = {i.item_id: i for i in load_items(sample)[0]}
    labels = TASKS[next(iter(items.values())).task].labels
    parsed = defaultdict(list)
    for r in _load(E104):
        if r["sample"] == sample and r["status"] == "ok" and r["parse_ok"]:
            parsed[r["item_id"]].append(r["probabilities"])
    rows = []
    for iid, i in sorted(items.items()):
        d = combine_jev(parsed.get(iid, []), labels)
        if d is not None:
            rows.append({"item_id": iid, "text": i.text, "label": i.label, "jev_label": d.label,
                         "jev_confidence": d.confidence, "jev_split": d.split, "jev_correct": d.label == i.label})
    return rows


def test_rows(sample: str, plan: list[dict]) -> list[dict]:
    texts = {i.item_id: i.text for i in load_items(sample)[0]}
    return [{**p, "text": texts[p["item_id"]]} for p in plan if p["sample"] == sample]


def conf_features(rows: list[dict], positive: str) -> np.ndarray:
    c = np.array([r["jev_confidence"] for r in rows])
    return np.column_stack([c, (c - 0.5) ** 2, [float(r["jev_split"]) for r in rows],
                            [float(r["jev_label"] == positive) for r in rows]])


def fit_predict(train: list[dict], test: list[dict], positive: str, use_text: bool):
    """Out-of-fold p(wrong) on train (for the threshold) and p(wrong) on test."""
    y = np.array([not r["jev_correct"] for r in train], dtype=int)

    def build(tr, te):
        a, b = conf_features(tr, positive), conf_features(te, positive)
        if not use_text:
            return a, b
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        ta = vec.fit_transform([r["text"] for r in tr])
        tb = vec.transform([r["text"] for r in te])
        return hstack([csr_matrix(a), ta]).tocsr(), hstack([csr_matrix(b), tb]).tocsr()

    def model():
        return LogisticRegression(C=1.0 if use_text else 10.0, max_iter=5000, class_weight="balanced")

    oof = np.zeros(len(train))
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
        a, b = build([train[i] for i in tr_i], [train[i] for i in va_i])
        oof[va_i] = model().fit(a, y[tr_i]).predict_proba(b)[:, 1]
    a, b = build(train, test)
    return oof, model().fit(a, y).predict_proba(b)[:, 1]


def best_threshold(p: np.ndarray, wrong: np.ndarray) -> float:
    """Threshold maximizing accuracy of 'flip when p > t' (flipping a wrong label makes it right: binary tasks)."""
    grid = np.unique(np.round(p, 3))
    gains = [((p > t) & wrong).sum() - ((p > t) & ~wrong).sum() for t in grid]
    return float(grid[int(np.argmax(gains))]) if max(gains) > 0 else 1.0


def flip_gain(p, wrong, t, rng, n_boot=2000):
    gain_per = np.where(p > t, np.where(wrong, 1.0, -1.0), 0.0)
    boots = [gain_per[rng.integers(0, len(p), len(p))].mean() for _ in range(n_boot)]
    return {"threshold": round(t, 3), "flipped": int((p > t).sum()), "fixed": int(((p > t) & wrong).sum()),
            "broken": int(((p > t) & ~wrong).sum()), "jev_acc": round(1 - wrong.mean(), 4),
            "gain_pts": round(100 * gain_per.mean(), 2),
            "ci95_pts": [round(100 * float(np.percentile(boots, q)), 2) for q in (2.5, 97.5)]}


def auc(y, p):
    return round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None


def main() -> None:
    plan = list(_load(PLAN))
    reviews = [r for r in _load(E106) if r["status"] == "ok" and r["parse_ok"]]
    groups = base_groups(reviews, "base_local")
    rng = np.random.default_rng(SEED)
    out = {}
    for train_s, test_s in POOLS.items():
        tr, te = train_rows(train_s), test_rows(test_s, plan)
        labels = TASKS[load_items(test_s)[0][0].task].labels
        positive = labels[0] if labels[0] in ("toxic", "harmful") else labels[1]
        wrong = np.array([not r["jev_correct"] for r in te])
        g = np.array([groups.get((test_s, r["item_id"]), "-") if not r["jev_correct"] else "right" for r in te])
        res = {"n_train": len(tr), "train_jev_err": round(1 - np.mean([r["jev_correct"] for r in tr]), 4),
               "n_test": len(te), "test_jev_err": round(float(wrong.mean()), 4),
               "test_groups": dict(Counter(g))}
        for name, use_text in (("conf", False), ("text", True)):
            oof, p = fit_predict(tr, te, positive, use_text)
            ytr = np.array([not r["jev_correct"] for r in tr])
            res[name] = {
                "train_oof_auc": auc(ytr.astype(int), oof),
                "test_auc": auc(wrong.astype(int), p),
                "test_auc_by_group": {k: auc(np.r_[np.ones((g == k).sum()), np.zeros((g == "right").sum())],
                                             np.r_[p[g == k], p[g == "right"]]) for k in ("A", "B", "C")},
                "flip_rule": flip_gain(p, wrong, best_threshold(oof, ytr), rng),
            }
        out[test_s] = res
        print(test_s, json.dumps(res, indent=1))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
