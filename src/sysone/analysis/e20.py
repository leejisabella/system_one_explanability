"""E.20: a correlation ceiling for the cascade (research_docs/17; rules in configs/experiments/e20.yaml).

Free: re-analyses saved runs, no API calls.

Usage:
  uv run python -m sysone.analysis.e20 [--out research_analyses/E20/data]

Every task is binary and the reviewer is never told who proposed a label, so its review of "label L on
item i" is the same whichever upstream proposed L. Per item and reviewer we keep:
  s[i]      share of the reviewer's alone samples that are right;
  o_right   observed P(final correct) when the right label was shown (mean over such reviews);
  o_wrong   observed P(final correct) when the wrong label was shown (jev_label, matched_label,
            upstream_wrong reviews pooled).
A cascade with upstream U then has per-item accuracy o_right if U is right, o_wrong if U is wrong.

Part 1, the ceiling: the reviewer's alone vote share against U's label, used as a detector of U's
mistakes, with a separate threshold per label U proposed. The best gain over U alone of any such policy:
    sum over label directions of max(0, max_t [#(U wrong, flagged) - #(U right, flagged)]) / n.
Part 2, the deference model: P(correct) = s + (1 - s) d_r when the right label is shown, s (1 - d_w) when
the wrong one is; d_w, d_r fitted per pool by moments on Jev -> base_local only, then used to predict the
other upstreams' cascades. The independence model replaces s by the pool's mean alone accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e102 import phi
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR

Key = tuple[str, str]
POOLS = ["civil_toxicity_pool_v1", "wildguard_prompt_full_v1", "wildguard_response_full_v1"]
SHORT = {"civil_toxicity_pool_v1": "Civil Comments", "wildguard_prompt_full_v1": "WildGuard prompts",
         "wildguard_response_full_v1": "WildGuard responses"}


def _rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- loading


class Reviewer:
    """One reviewer variant's per-item alone shares and observed review outcomes."""

    def __init__(self, name: str):
        self.name = name
        self.alone: dict[Key, list[bool]] = defaultdict(list)
        self._right: dict[Key, list[bool]] = defaultdict(list)
        self._wrong: dict[Key, list[bool]] = defaultdict(list)

    def add_review(self, r: dict[str, Any]) -> None:
        k = (r["sample"], r["item_id"])
        ok = r["final_label"] == r["label"]
        (self._right if r["proposed_label"] == r["label"] else self._wrong)[k].append(ok)

    def s(self, k: Key) -> float | None:
        a = self.alone.get(k)
        return sum(a) / len(a) if a else None

    def o(self, k: Key, shown_right: bool) -> float | None:
        v = (self._right if shown_right else self._wrong).get(k)
        return sum(v) / len(v) if v else None

    @property
    def n_alone(self) -> int:
        counts = sorted(len(v) for v in self.alone.values())
        return counts[len(counts) // 2] if counts else 0


def _ok(r: dict[str, Any]) -> bool:
    return r.get("status") == "ok" and bool(r.get("parse_ok")) and r.get("final_label") is not None


def load_reviewers(src: dict[str, str]) -> dict[str, Reviewer]:
    runs = RESULTS_DIR / "runs"
    out: dict[str, Reviewer] = {}

    def get(name: str) -> Reviewer:
        return out.setdefault(name, Reviewer(name))

    for r in _rows(runs / src["reviews_20b"] / "reviews.jsonl"):           # E.106: base_local + arms
        if not _ok(r) or r.get("role", "reviewer") != "reviewer":
            continue
        rv = get(r["reviewer"])
        if r["condition"] == "alone":
            rv.alone[(r["sample"], r["item_id"])].append(r["final_label"] == r["label"])
        elif r.get("proposed_label"):
            rv.add_review(r)
    bed = get("e102_bedrock")
    for r in _rows(runs / src["reviews_20b_bedrock"] / "reviews.jsonl"):   # E.102: Bedrock 20b + Qwen ref
        if not _ok(r):
            continue
        name = "e102_bedrock" if r["reviewer"] == "reviewer" else "qwen_reference"
        rv = get(name)
        if r["condition"] == "alone":
            rv.alone[(r["sample"], r["item_id"])].append(r["final_label"] == r["label"])
        elif r.get("proposed_label"):
            rv.add_review(r)
    for r in _rows(runs / src["protocols"] / "reviews.jsonl"):            # E.103 protocols
        if not _ok(r) or not r.get("proposed_label"):
            continue
        rv = get(f"e103_{r['protocol']}")
        rv.add_review(r)
    for name in [n for n in out if n.startswith("e103_")]:
        out[name].alone = bed.alone                                    # protocols use E.102's alone answers
    for r in _rows(runs / src["reviews_120b"] / "reviews.jsonl"):          # E.119: 120b
        if not _ok(r):
            continue
        rv = get("e119_120b")
        if r["condition"] == "alone":
            rv.alone[(r["sample"], r["item_id"])].append(r["final_label"] == r["label"])
        elif r.get("proposed_label"):
            rv.add_review(r)
    return out


def load_upstreams(src: dict[str, str]) -> tuple[dict[Key, dict], dict[str, dict[Key, bool]]]:
    """The plan rows, and per upstream: item -> is U right?"""
    runs = RESULTS_DIR / "runs"
    plan = {(p["sample"], p["item_id"]): p for p in _rows(runs / src["plan"] / "plan.jsonl")}
    ups: dict[str, dict[Key, bool]] = {"jev": {k: p["jev_correct"] for k, p in plan.items()},
                                       "matched": {k: p["matched_correct"] for k, p in plan.items()}}
    for r in _rows(runs / src["upstream"] / "upstream.jsonl"):
        k = (r["sample"], r["item_id"])
        if _ok(r) and k in plan:
            ups.setdefault(r["reviewer"], {})[k] = r["final_label"] == plan[k]["label"]
    return plan, ups


# ---------------------------------------------------------------- statistics


def ceiling_gain(items: list[tuple[bool, float, str]]) -> float:
    """items: (U right?, reviewer disagreement share with U's label, U's label). Best gain / n."""
    n = len(items)
    if not n:
        return float("nan")
    total = 0.0
    for direction in {lab for _, _, lab in items}:
        sub = [(r, a) for r, a, lab in items if lab == direction]
        best = 0
        for t in sorted({a for _, a in sub if a > 0}):
            g = sum((not r) - r for r, a in sub if a >= t)
            best = max(best, g)
        total += best
    return total / n


def fit_d(pairs_right: list[tuple[float, float]], pairs_wrong: list[tuple[float, float]]) -> tuple[float, float]:
    """Method of moments. pairs_*: (s, observed P(correct)). Returns (d_w, d_r), clipped to [0, 1]."""
    s_w, o_w = sum(s for s, _ in pairs_wrong), sum(o for _, o in pairs_wrong)
    d_w = 1 - o_w / s_w if s_w > 0 else 0.0
    s_r, o_r = sum(s for s, _ in pairs_right), sum(o for _, o in pairs_right)
    miss = sum(1 - s for s, _ in pairs_right)
    d_r = (o_r - s_r) / miss if miss > 0 else 0.0
    return min(max(d_w, 0.0), 1.0), min(max(d_r, 0.0), 1.0)


def predict(s: float, shown_right: bool, d_w: float, d_r: float) -> float:
    return s + (1 - s) * d_r if shown_right else s * (1 - d_w)


def _pctl(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if x == x)
    return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else float("nan")


def ci(xs: list[float]) -> list[float]:
    return [round(_pctl(xs, 0.025), 4), round(_pctl(xs, 0.975), 4)]


# ---------------------------------------------------------------- the analysis


class Data:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.plan, self.ups = load_upstreams(cfg["source_runs"])
        self.revs = load_reviewers(cfg["source_runs"])
        self.sub = {p: sorted(k for k, r in self.plan.items() if r["sample"] == p and r["in_subsample"]) for p in POOLS}
        self.label = {k: r["label"] for k, r in self.plan.items()}
        self.labelset: dict[str, set[str]] = defaultdict(set)
        for r in self.plan.values():
            self.labelset[r["sample"]] |= {r["label"], r["jev_label"], r["matched_label"]}

    def u_label(self, u: str, k: Key) -> str:
        lab = self.label[k]
        if self.ups[u][k]:
            return lab
        return next(x for x in sorted(self.labelset[k[0]]) if x != lab)

    def cascade_items(self, u: str, rv: Reviewer, pool: str) -> list[Key]:
        """Subsample items where U decided and the reviewer has the review U's label calls for."""
        out = []
        for k in self.sub[pool]:
            if k not in self.ups[u]:
                continue
            if rv.o(k, self.ups[u][k]) is not None:
                out.append(k)
        return out


def observed_gain_boot(D: Data, u: str, rv: Reviewer, items: dict[str, list[Key]], reps: int, seed: int):
    """Observed cascade accuracy and gain over U, pooled with equal pool weight, with bootstrap."""
    def stat(sel: dict[str, list[Key]]):
        acc, gain = [], []
        for p, ks in sel.items():
            if not ks:
                continue
            o = [rv.o(k, D.ups[u][k]) for k in ks]
            a = sum(o) / len(ks)
            acc.append(a)
            gain.append(a - sum(D.ups[u][k] for k in ks) / len(ks))
        return sum(acc) / len(acc), sum(gain) / len(gain)

    point = stat(items)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        boots.append(stat({p: [ks[rng.randrange(len(ks))] for _ in ks] for p, ks in items.items() if ks}))
    return {"acc": round(point[0], 4), "acc_ci": ci([b[0] for b in boots]),
            "gain": round(point[1], 4), "gain_ci": ci([b[1] for b in boots]), "n": sum(len(v) for v in items.values())}


def ceiling_for(D: Data, u: str, rv: Reviewer, reps: int, seed: int) -> dict[str, Any]:
    per_pool, items_by_pool = {}, {}
    for p in POOLS:
        rows = []
        for k in D.sub[p]:
            if k not in D.ups[u] or rv.s(k) is None:
                continue
            right = D.ups[u][k]
            s = rv.s(k)
            rows.append((right, (1 - s) if right else s, D.u_label(u, k)))
        items_by_pool[p] = rows
        per_pool[p] = {"ceiling": round(ceiling_gain(rows), 4), "n": len(rows),
                       "u_acc": round(sum(r for r, _, _ in rows) / len(rows), 4) if rows else None}
    pooled = sum(v["ceiling"] for v in per_pool.values()) / len(POOLS)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        boots.append(sum(ceiling_gain([rows[rng.randrange(len(rows))] for _ in rows]) for rows in items_by_pool.values())
                     / len(POOLS))
    # error overlap: U's errors vs the reviewer's alone-majority errors
    ks = [k for p in POOLS for k in D.sub[p] if k in D.ups[u] and rv.s(k) is not None]
    ph = phi([not D.ups[u][k] for k in ks], [rv.s(k) <= 0.5 for k in ks])
    r_acc = sum(rv.s(k) > 0.5 for k in ks) / len(ks) if ks else None
    return {"pooled": round(pooled, 4), "ci": ci(boots), "pools": per_pool, "phi": ph,
            "reviewer_alone_majority_acc": round(r_acc, 4) if r_acc is not None else None,
            "n_alone": rv.n_alone}


def deference_test(D: Data, reps: int, seed: int) -> dict[str, Any]:
    fit_u, fit_r = D.cfg["fit_cascade"]["upstream"], D.cfg["fit_cascade"]["reviewer"]
    rv = D.revs[fit_r]

    def fit(items_by_pool: dict[str, list[Key]], indep: bool) -> dict[str, tuple[float, float]]:
        out = {}
        for p, ks in items_by_pool.items():
            sbar = sum(rv.s(k) for k in ks) / len(ks)
            sv = (lambda k: sbar) if indep else rv.s
            right = [(sv(k), rv.o(k, True)) for k in ks if D.ups[fit_u][k]]
            wrong = [(sv(k), rv.o(k, False)) for k in ks if not D.ups[fit_u][k]]
            out[p] = fit_d(right, wrong)
        return out

    def items_for(u: str) -> dict[str, list[Key]]:
        return {p: [k for k in D.cascade_items(u, rv, p) if rv.s(k) is not None] for p in POOLS}

    def pred(u: str, ks_by_pool, d, indep: bool, fit_pool_items) -> float:
        accs = []
        for p, ks in ks_by_pool.items():
            sbar = sum(rv.s(k) for k in fit_pool_items[p]) / len(fit_pool_items[p])
            d_w, d_r = d[p]
            accs.append(sum(predict(sbar if indep else rv.s(k), D.ups[u][k], d_w, d_r) for k in ks) / len(ks))
        return sum(accs) / len(accs)

    fit_items = items_for(fit_u)
    d_item, d_ind = fit(fit_items, False), fit(fit_items, True)
    out: dict[str, Any] = {"fit": {"upstream": fit_u, "reviewer": fit_r,
                                   "d_item": {p: {"d_w": round(a, 4), "d_r": round(b, 4)} for p, (a, b) in d_item.items()},
                                   "d_independence": {p: {"d_w": round(a, 4), "d_r": round(b, 4)} for p, (a, b) in d_ind.items()}},
                           "cascades": {}}
    for u in [fit_u, *D.cfg["held_out_cascades"]]:
        ks = items_for(u)
        eligible = {p: [k for k in D.sub[p] if k in D.ups[u]] for p in POOLS}
        obs = observed_gain_boot(D, u, rv, ks, reps, seed)
        p_item, p_ind = pred(u, ks, d_item, False, fit_items), pred(u, ks, d_ind, True, fit_items)
        # joint bootstrap of the prediction error (refit d on resampled fit items, same resample of U's items)
        rng = random.Random(seed + 1)
        errs = []
        for _ in range(reps // 4):
            fi = {p: [v[rng.randrange(len(v))] for _ in v] for p, v in fit_items.items()}
            ki = {p: [v[rng.randrange(len(v))] for _ in v] for p, v in ks.items()}
            o = sum(sum(rv.o(k, D.ups[u][k]) for k in v) / len(v) for v in ki.values()) / len(POOLS)
            errs.append(pred(u, ki, fit(fi, False), False, fi) - o)
        out["cascades"][u] = {
            "observed": obs, "predicted_item": round(p_item, 4), "predicted_independence": round(p_ind, 4),
            "inside_ci": obs["acc_ci"][0] <= p_item <= obs["acc_ci"][1],
            "error_item": round(p_item - obs["acc"], 4), "error_independence": round(p_ind - obs["acc"], 4),
            "error_item_ci_joint": ci(errs),
            "coverage": {p: f"{len(ks[p])}/{len(eligible[p])}" for p in POOLS},
            "u_alone_acc": round(sum(sum(D.ups[u][k] for k in v) / len(v) for v in ks.values()) / len(POOLS), 4),
        }
    held = [out["cascades"][u] for u in D.cfg["held_out_cascades"]]
    mae_item = sum(abs(c["error_item"]) for c in held) / len(held)
    mae_ind = sum(abs(c["error_independence"]) for c in held) / len(held)
    out["mae_item"], out["mae_independence"] = round(mae_item, 4), round(mae_ind, 4)
    out["r1_all_inside_ci"] = all(c["inside_ci"] for c in held)
    out["r1_beats_independence"] = mae_item < mae_ind
    out["r1_pass"] = out["r1_all_inside_ci"] and out["r1_beats_independence"]
    return out


def transfer_120b(D: Data, reps: int, seed: int) -> dict[str, Any]:
    """Jev -> 120b predicted from 120b's alone shares with 20b's d, and with 120b's own fitted d."""
    small, big = D.revs["base_local"], D.revs["e119_120b"]
    res: dict[str, Any] = {}
    for label, d_src in [("d_from_20b", small), ("d_own", big)]:
        accs, obs = [], []
        for p in POOLS:
            ks = [k for k in D.cascade_items("jev", big, p) if big.s(k) is not None]
            fk = [k for k in D.cascade_items("jev", d_src, p) if d_src.s(k) is not None]
            d_w, d_r = fit_d([(d_src.s(k), d_src.o(k, True)) for k in fk if D.ups["jev"][k]],
                             [(d_src.s(k), d_src.o(k, False)) for k in fk if not D.ups["jev"][k]])
            accs.append(sum(predict(big.s(k), D.ups["jev"][k], d_w, d_r) for k in ks) / len(ks))
            obs.append(sum(big.o(k, D.ups["jev"][k]) for k in ks) / len(ks))
        res[label] = {"predicted": round(sum(accs) / 3, 4), "observed": round(sum(obs) / 3, 4)}
    return res


def frontier(D: Data, reps: int, seed: int) -> dict[str, Any]:
    """Per variant, Jev upstream: fix rate on Jev's wrong labels and break rate on its right labels,
    on the subsample, per pool and pooled; plus each variant's fitted (d_w, d_r) and alone accuracy."""
    out = {}
    for name, rv in sorted(D.revs.items()):
        if name == "qwen_reference":
            continue
        pools = {}
        for p in POOLS:
            ks = D.cascade_items("jev", rv, p)
            if not ks:
                continue
            wrong = [k for k in ks if not D.ups["jev"][k]]
            right = [k for k in ks if D.ups["jev"][k]]
            fix = sum(rv.o(k, False) for k in wrong) / len(wrong)
            brk = 1 - sum(rv.o(k, True) for k in right) / len(right)
            sk = [k for k in ks if rv.s(k) is not None]
            d_w, d_r = fit_d([(rv.s(k), rv.o(k, True)) for k in sk if D.ups["jev"][k]],
                             [(rv.s(k), rv.o(k, False)) for k in sk if not D.ups["jev"][k]])
            alone_fix = sum(rv.s(k) for k in sk if not D.ups["jev"][k]) / max(1, sum(not D.ups["jev"][k] for k in sk))
            alone_brk = 1 - sum(rv.s(k) for k in sk if D.ups["jev"][k]) / max(1, sum(D.ups["jev"][k] for k in sk))
            err = len(wrong) / len(ks)
            pools[p] = {"fix": round(fix, 4), "break": round(brk, 4), "jev_err": round(err, 4),
                        "gain": round(err * fix - (1 - err) * brk, 4), "d_w": round(d_w, 4), "d_r": round(d_r, 4),
                        "alone_fix": round(alone_fix, 4), "alone_break": round(alone_brk, 4),
                        "n_wrong": len(wrong), "n_right": len(right)}
        if len(pools) == 3:
            out[name] = {"pools": pools, "n_alone": rv.n_alone,
                         "pooled_gain": round(sum(v["gain"] for v in pools.values()) / 3, 4)}
    return out


def roc_points(D: Data, u: str, rv: Reviewer) -> dict[str, list[list[float]]]:
    """Per pool: (FPR, TPR) of 'reviewer's alone disagreement share >= t' as a detector of U's mistakes."""
    out = {}
    for p in POOLS:
        rows = [(D.ups[u][k], (1 - rv.s(k)) if D.ups[u][k] else rv.s(k)) for k in D.sub[p]
                if k in D.ups[u] and rv.s(k) is not None]
        n_w = sum(not r for r, _ in rows)
        n_r = len(rows) - n_w
        pts = [[0.0, 0.0]]
        for t in sorted({a for _, a in rows if a > 0}, reverse=True):
            pts.append([round(sum(r for r, a in rows if a >= t) / n_r, 4), round(sum((not r) for r, a in rows if a >= t) / n_w, 4)])
        pts.append([1.0, 1.0])
        out[p] = pts
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    reps, seed = cfg["bootstrap"], cfg["seed"]
    D = Data(cfg)
    S: dict[str, Any] = {"reviewers": {n: {"n_alone": r.n_alone, "alone_items": len(r.alone)} for n, r in D.revs.items()}}
    S["deference"] = deference_test(D, reps, seed)
    S["transfer_120b"] = transfer_120b(D, reps, seed)
    S["frontier"] = frontier(D, reps, seed)
    S["roc_base_local_jev"] = roc_points(D, "jev", D.revs["base_local"])

    # ceilings: every upstream x every reviewer that has alone answers on the subsample
    reviewers_5 = {"base_local", "e102_bedrock", "e119_120b"}
    ceil = {}
    for u in D.ups:
        for name, rv in D.revs.items():
            if name.startswith("e103_") or not rv.alone:
                continue
            ceil[f"{u}|{name}"] = ceiling_for(D, u, rv, reps // 4, seed)
    # other upstream models as single-answer "reviewers" of each other (more pairs for ceiling vs phi)
    for u in D.ups:
        for v in ["explainer_open", "reviewer_qwen", "decision_alt"]:
            if u == v:
                continue
            rv = Reviewer(f"model_{v}")
            for k, right in D.ups[v].items():
                rv.alone[k] = [right]
            ceil[f"{u}|model_{v}"] = ceiling_for(D, u, rv, reps // 4, seed)
    S["ceilings"] = ceil
    pairs = [(c["phi"], c["pooled"]) for c in ceil.values() if c["phi"] is not None]
    S["ceiling_vs_phi_spearman"] = spearman([a for a, _ in pairs], [b for _, b in pairs])

    # R2: observed cascades against their ceilings (5-sample reviewers; protocols use E.102's alone)
    r2 = {}
    for name, rv in D.revs.items():
        five = name in reviewers_5 or name.startswith("e103_")
        for u in D.ups:
            items = {p: D.cascade_items(u, rv, p) for p in POOLS}
            if sum(len(v) for v in items.values()) < 600:
                continue
            obs = observed_gain_boot(D, u, rv, items, reps // 4, seed)
            c_name = "e102_bedrock" if name.startswith("e103_") else name
            c = ceil.get(f"{u}|{c_name}")
            if c is None:
                continue
            r2[f"{u}|{name}"] = {"observed_gain": obs["gain"], "gain_ci": obs["gain_ci"], "ceiling": c["pooled"],
                                 "above_ceiling": obs["gain_ci"][0] > c["pooled"], "in_rule": five,
                                 "observed_acc": obs["acc"], "n": obs["n"]}
    S["r2"] = r2
    S["r2_pass"] = not any(v["above_ceiling"] for v in r2.values() if v["in_rule"])
    S["decision"] = {"r1_overlap_and_deference_explain_cascade": S["deference"]["r1_pass"],
                     "r1_all_inside_ci": S["deference"]["r1_all_inside_ci"],
                     "r1_beats_independence": S["deference"]["r1_beats_independence"],
                     "r2_ceiling_holds": S["r2_pass"]}
    return S


# ---------------------------------------------------------------- report


def _pct(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def _pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def _ci(c: list[float]) -> str:
    return f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


NAMES = {"jev": "Jev", "explainer_open": "Llama 4 Maverick", "reviewer_qwen": "Qwen3-32B", "decision_alt": "Nova Lite",
         "matched": "matched random", "base_local": "gpt-oss-20b (E.106)", "e102_bedrock": "gpt-oss-20b (E.102, Bedrock)",
         "e119_120b": "gpt-oss-120b (E.119)", "qwen_reference": "Qwen3-32B (E.102 reference)",
         "sft_review": "SFT review (E.106)", "sft_plain": "SFT plain (E.106)", "dpo_review": "DPO (E.107)",
         "consist_review": "consistency (E.108)", "balanced_review": "balanced proposals (E.108)",
         "e103_blind_first": "blind-first (E.103)", "e103_warning": "warning (E.103)",
         "e103_calibrated_warning": "calibrated warning (E.103)", "e103_consider_opposite": "consider-the-opposite (E.103)",
         "model_explainer_open": "Llama (1 answer)", "model_reviewer_qwen": "Qwen3-32B (1 answer)",
         "model_decision_alt": "Nova (1 answer)"}


def report(S: dict[str, Any]) -> str:
    d = S["decision"]
    L = ["# E.20 results", "",
         "## Decision (rules fixed in configs/experiments/e20.yaml before computing)", "",
         f"- **R1, overlap and deference explain the cascade: {'PASS' if d['r1_overlap_and_deference_explain_cascade'] else 'FAIL'}** "
         f"(every held-out prediction inside the observed 95% interval: {d['r1_all_inside_ci']}; "
         f"item-level model beats the independence model: {d['r1_beats_independence']}).",
         f"- **R2, the ceiling holds: {'PASS' if d['r2_ceiling_holds'] else 'FAIL'}.**", "",
         "## Part 2: the deference model, fitted on Jev → gpt-oss-20b only", ""]
    f = S["deference"]["fit"]
    L.append("Fitted per pool (item-level model): " + "; ".join(
        f"{SHORT[p]} d_w {v['d_w']:.2f}, d_r {v['d_r']:.2f}" for p, v in f["d_item"].items()) + ".")
    L += ["", "| Upstream → 20b | Upstream alone | Observed cascade [95%] | Predicted (item-level) | Predicted (independence) | Coverage |",
          "|---|---|---|---|---|---|"]
    for u, c in S["deference"]["cascades"].items():
        o = c["observed"]
        tag = " (fitted)" if u == f["upstream"] else ""
        L.append(f"| {NAMES.get(u, u)}{tag} | {_pct(c['u_alone_acc'])} | {_pct(o['acc'])} [{_pct(o['acc_ci'][0])}, {_pct(o['acc_ci'][1])}] | "
                 f"{_pct(c['predicted_item'])}{' ✓' if c['inside_ci'] else ' ✗'} | {_pct(c['predicted_independence'])} | "
                 f"{', '.join(c['coverage'].values())} |")
    L += ["", f"Mean absolute error over the held-out cascades: item-level {100 * S['deference']['mae_item']:.1f} points, "
              f"independence {100 * S['deference']['mae_independence']:.1f} points.", ""]
    t = S["transfer_120b"]
    L += [f"Jev → 120b (E.119): observed {_pct(t['d_own']['observed'])}; predicted from the 120b's alone answers with the 20b's "
          f"deference {_pct(t['d_from_20b']['predicted'])}, with its own fitted deference {_pct(t['d_own']['predicted'])}.", "",
          "## Part 1: the ceiling", "",
          "Best gain over the upstream alone of any policy that uses only the reviewer's own alone answers "
          "(thresholds on its vote share, chosen separately per proposed label, in-sample, so optimistic). Pooled, equal pool weight.", "",
          "| Upstream | Reviewer | Alone answers | phi (errors) | Ceiling [95%] |", "|---|---|---|---|---|"]
    for key, c in sorted(S["ceilings"].items(), key=lambda kv: (kv[0].split("|")[0] != "jev", kv[0])):
        u, r = key.split("|")
        L.append(f"| {NAMES.get(u, u)} | {NAMES.get(r, r)} | {c['n_alone']} | {c['phi']} | {_pts(c['pooled'])} {_ci(c['ci'])} |")
    L += ["", f"Spearman correlation of the ceiling with phi across all pairs: {S['ceiling_vs_phi_spearman']}.", "",
          "## R2: observed cascades against their ceilings", "",
          "| Upstream | Reviewer | Observed gain [95%] | Ceiling | Above? | In rule |", "|---|---|---|---|---|---|"]
    for key, v in sorted(S["r2"].items()):
        u, r = key.split("|")
        L.append(f"| {NAMES.get(u, u)} | {NAMES.get(r, r)} | {_pts(v['observed_gain'])} {_ci(v['gain_ci'])} | {_pts(v['ceiling'])} | "
                 f"{'yes' if v['above_ceiling'] else 'no'} | {'yes' if v['in_rule'] else 'no (1 alone answer)'} |")
    L += ["", "## Frontier (Jev upstream): fix rate on Jev's mistakes vs break rate on its right labels", "",
          "| Reviewer | Pool | Fix | Break | Gain | d_w | d_r | Alone fix | Alone break |", "|---|---|---|---|---|---|---|---|---|"]
    for name, v in S["frontier"].items():
        for p, x in v["pools"].items():
            L.append(f"| {NAMES.get(name, name)} | {SHORT[p]} | {_pct(x['fix'])} | {_pct(x['break'])} | {_pts(x['gain'])} | "
                     f"{x['d_w']:.2f} | {x['d_r']:.2f} | {_pct(x['alone_fix'])} | {_pct(x['alone_break'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e20.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "research_analyses" / "E20" / "data"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    S = run(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(report(S))
    print(report(S))


if __name__ == "__main__":
    main()
