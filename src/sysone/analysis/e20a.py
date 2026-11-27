"""E.20a: selective override, out of sample (rules in configs/experiments/e20a.yaml). Free.

Usage:
  uv run python -m sysone.analysis.e20a [--out research_analyses/E20a/data]

Per pool and per label Jev proposed, override Jev only if the reviewer's first n alone answers all
disagree ("unanimous") or at least half do ("majority"), or never. The rule is chosen on the training
pools (E.104 Jev decisions, E.105 alone tries, reweighted to Jev's real error rates) and tested once on
E.102's subsample.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from sysone.cascade.proposals import combine_jev
from sysone.cascade.tasks import TASKS
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR

Key = tuple[str, str]
POOLS = ["civil_toxicity_pool_v1", "wildguard_prompt_full_v1", "wildguard_response_full_v1"]
TRAIN_OF = {"civil_toxicity_pool_v1": "civil_toxicity_train_v1", "wildguard_prompt_full_v1": "wildguard_prompt_train_v1",
            "wildguard_response_full_v1": "wildguard_response_train_v1"}
SHORT = {"civil_toxicity_pool_v1": "Civil Comments", "wildguard_prompt_full_v1": "WildGuard prompts",
         "wildguard_response_full_v1": "WildGuard responses"}
CONSERVATIVE = ["never", "unanimous", "majority"]


def _rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _ok(r: dict[str, Any]) -> bool:
    return r.get("status") == "ok" and bool(r.get("parse_ok")) and r.get("final_label") is not None


def overrides(rule: str, answers: list[str], jev_label: str) -> bool:
    dis = sum(a != jev_label for a in answers)
    if rule == "never" or not answers:
        return False
    if rule == "unanimous":
        return dis == len(answers)
    if rule == "majority":
        return 2 * dis >= len(answers)
    raise ValueError(rule)


def first_n(answers: list[tuple[int, str]], n: int | None) -> list[str]:
    a = [lab for _, lab in sorted(answers)]
    return a if n is None else a[:n]


def load_alone(path: Path, reviewer: str | None, role_ok=lambda r: True) -> dict[Key, list[tuple[int, str]]]:
    out: dict[Key, list[tuple[int, str]]] = defaultdict(list)
    for r in _rows(path):
        if not _ok(r) or r.get("condition") != "alone" or not role_ok(r):
            continue
        if reviewer is not None and r.get("reviewer") != reviewer:
            continue
        out[(r["sample"], r["item_id"])].append((int(r["sample_index"]), r["final_label"]))
    return out


def choose(cfg: dict[str, Any]) -> dict[str, Any]:
    runs = RESULTS_DIR / "runs"
    src = cfg["source_runs"]
    n = cfg["n_answers"]
    probs: dict[Key, list[dict]] = defaultdict(list)
    for r in _rows(runs / src["jev_train"] / "jev_decisions.jsonl"):
        if r.get("status") == "ok" and r.get("parse_ok"):
            probs[(r["sample"], r["item_id"])].append(r["probabilities"])
    gold: dict[Key, str] = {}
    task_of: dict[str, str] = {}
    alone: dict[Key, list[tuple[int, str]]] = defaultdict(list)
    for r in _rows(runs / src["traces_train"] / "traces.jsonl"):
        if r.get("source") != "reviewer" or not _ok(r):
            continue
        k = (r["sample"], r["item_id"])
        gold[k] = r["label"]
        task_of[r["sample"]] = r["task"]
        if r["condition"] == "alone":
            alone[k].append((int(r["sample_index"]), r["final_label"]))
    jev = {k: combine_jev(ps, TASKS[task_of[k[0]]].labels) for k, ps in probs.items() if k[0] in task_of}
    # whole-pool Jev error rates per direction need gold labels for every training item
    from sysone.cascade.items import load_items
    out: dict[str, Any] = {}
    for pool in POOLS:
        tp = TRAIN_OF[pool]
        items = {i.item_id: i.label for i in load_items(tp)[0]}
        labels = TASKS[task_of[tp]].labels
        n_all = 0
        joint = {lab: {"wrong": 0, "right": 0} for lab in labels}
        for iid, g in items.items():
            d = jev.get((tp, iid))
            if d is None:
                continue
            n_all += 1
            joint[d.label]["wrong" if d.label != g else "right"] += 1
        chosen, detail = {}, {}
        for lab in labels:
            p_w, p_r = joint[lab]["wrong"] / n_all, joint[lab]["right"] / n_all
            mix = [(k, d) for k, d in jev.items() if d is not None and k[0] == tp and d.label == lab
                   and k in gold and len(alone.get(k, [])) >= n]
            gains = {}
            for rule in CONSERVATIVE:
                ow = [overrides(rule, first_n(alone[k], n), lab) for k, d in mix if d.label != gold[k]]
                orr = [overrides(rule, first_n(alone[k], n), lab) for k, d in mix if d.label == gold[k]]
                if not ow or not orr:
                    gains[rule] = 0.0 if rule == "never" else float("-inf")
                    continue
                gains[rule] = p_w * (sum(ow) / len(ow)) - p_r * (sum(orr) / len(orr))
            best = max(CONSERVATIVE, key=lambda r: (round(gains[r], 9), -CONSERVATIVE.index(r)))
            chosen[lab] = best
            detail[lab] = {"jev_wrong_share": round(p_w, 4), "jev_right_share": round(p_r, 4),
                           "train_gain": {r: round(g, 5) for r, g in gains.items()},
                           "n_mix_wrong": sum(d.label != gold[k] for k, d in mix), "n_mix_right": sum(d.label == gold[k] for k, d in mix)}
        out[pool] = {"chosen": chosen, "detail": detail, "n_train_items": n_all}
    return out


def evaluate(cfg: dict[str, Any], policy: dict[str, dict[str, str]], alone: dict[Key, list[tuple[int, str]]],
             n: int | None, *, map5: bool = False) -> dict[str, Any]:
    runs = RESULTS_DIR / "runs"
    plan = [p for p in _rows(runs / cfg["source_runs"]["plan"] / "plan.jsonl") if p["in_subsample"]]
    per: dict[str, list[tuple[bool, bool]]] = defaultdict(list)   # (jev right, policy right)
    overrides_n: dict[str, int] = defaultdict(int)
    for p in plan:
        k = (p["sample"], p["item_id"])
        ans = first_n(alone.get(k, []), n)
        if len(ans) < (n or 1):
            continue
        rule = policy[p["sample"]][p["jev_label"]]
        flip = overrides(rule, ans, p["jev_label"])
        overrides_n[p["sample"]] += flip
        per[p["sample"]].append((p["jev_correct"], (not p["jev_correct"]) if flip else p["jev_correct"]))

    def stat(sel):
        accs = [(sum(b for _, b in v) - sum(a for a, _ in v)) / len(v) for v in sel.values()]
        pol = [sum(b for _, b in v) / len(v) for v in sel.values()]
        return sum(accs) / len(accs), sum(pol) / len(pol)

    gain, acc = stat(per)
    rng = random.Random(cfg["seed"])
    boots = []
    for _ in range(cfg["bootstrap"]):
        boots.append(stat({p: [v[rng.randrange(len(v))] for _ in v] for p, v in per.items()})[0])
    boots.sort()
    ci = [round(boots[int(0.025 * len(boots))], 4), round(boots[int(0.975 * len(boots)) - 1], 4)]
    pools = {}
    for p, v in per.items():
        fixed = sum((not a) and b for a, b in v)
        broken = sum(a and (not b) for a, b in v)
        pools[p] = {"n": len(v), "jev_acc": round(sum(a for a, _ in v) / len(v), 4), "policy_acc": round(sum(b for _, b in v) / len(v), 4),
                    "fixed": fixed, "broken": broken, "overrides": overrides_n[p]}
    return {"gain": round(gain, 4), "ci95": ci, "policy_acc": round(acc, 4), "pools": pools}


def in_sample_best(cfg, alone, n) -> dict[str, Any]:
    """The best rule per pool and direction chosen on the test subsample itself (optimistic, for comparison)."""
    runs = RESULTS_DIR / "runs"
    plan = [p for p in _rows(runs / cfg["source_runs"]["plan"] / "plan.jsonl") if p["in_subsample"]]
    pol = {}
    for pool in POOLS:
        pol[pool] = {}
        for lab in {p["jev_label"] for p in plan if p["sample"] == pool}:
            rows = [p for p in plan if p["sample"] == pool and p["jev_label"] == lab and len(first_n(alone.get((pool, p["item_id"]), []), n)) >= (n or 1)]
            def g(rule):
                return sum((1 if not p["jev_correct"] else -1) * overrides(rule, first_n(alone[(pool, p["item_id"])], n), lab) for p in rows)
            pol[pool][lab] = max(CONSERVATIVE, key=lambda r: (g(r), -CONSERVATIVE.index(r)))
    return pol


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    runs = RESULTS_DIR / "runs"
    src = cfg["source_runs"]
    n = cfg["n_answers"]
    C = choose(cfg)
    policy = {p: v["chosen"] for p, v in C.items()}
    bed = load_alone(runs / src["test_reviewer"] / "reviews.jsonl", "reviewer")
    vll = load_alone(runs / src["test_reviewer_alt"] / "reviews.jsonl", "base_local")
    S = {"choice": C, "policy": policy,
         "test_bedrock_n2": evaluate(cfg, policy, bed, n),
         "test_bedrock_n5": evaluate(cfg, policy, bed, None),
         "test_vllm_n2": evaluate(cfg, policy, vll, n),
         "test_vllm_n5": evaluate(cfg, policy, vll, None)}
    ins = in_sample_best(cfg, bed, n)
    S["in_sample_best_policy_bedrock_n2"] = ins
    S["in_sample_best_bedrock_n2"] = evaluate(cfg, ins, bed, n)
    r = S["test_bedrock_n2"]
    S["decision"] = {"selective_override_beats_jev": r["ci95"][0] > cfg["rule"]["ci_lower_above"],
                     "gain": r["gain"], "ci95": r["ci95"]}
    return S


def _p(x: float) -> str:
    return f"{100 * x:+.1f}"


def report(S: dict[str, Any]) -> str:
    d = S["decision"]
    L = ["# E.20a results", "",
         f"**Decision (rule fixed before computing): selective override beats Jev: {'YES' if d['selective_override_beats_jev'] else 'NO'}.** "
         f"Gain over Jev alone on E.102's subsample (Bedrock 20b, first 2 alone answers): {_p(d['gain'])} "
         f"[{_p(d['ci95'][0])}, {_p(d['ci95'][1])}].", "", "## Policy chosen on the training pools", "",
         "| Pool | Jev proposed | Chosen rule | Training gain: never / unanimous / majority (points) | Jev wrong / right share of pool |",
         "|---|---|---|---|---|"]
    for pool, v in S["choice"].items():
        for lab, det in v["detail"].items():
            g = det["train_gain"]
            L.append(f"| {SHORT[pool]} | {lab} | **{v['chosen'][lab]}** | {_p(g['never'])} / {_p(g['unanimous'])} / {_p(g['majority'])} | "
                     f"{100 * det['jev_wrong_share']:.1f}% / {100 * det['jev_right_share']:.1f}% |")
    L += ["", "## Test (E.102 subsample, 400 per pool, pooled with equal weight)", "",
          "| Variant | Gain over Jev [95%] | Policy accuracy | Fixed / broken (CC, WG-p, WG-r) |", "|---|---|---|---|"]
    names = [("test_bedrock_n2", "**Bedrock 20b, 2 answers (the rule)**"), ("test_bedrock_n5", "Bedrock 20b, 5 answers"),
             ("test_vllm_n2", "vLLM 20b, 2 answers"), ("test_vllm_n5", "vLLM 20b, 5 answers"),
             ("in_sample_best_bedrock_n2", "In-sample best rule on the test items (optimistic)")]
    for key, label in names:
        r = S[key]
        fb = ", ".join(f"{r['pools'][p]['fixed']}/{r['pools'][p]['broken']}" for p in POOLS if p in r["pools"])
        L.append(f"| {label} | {_p(r['gain'])} [{_p(r['ci95'][0])}, {_p(r['ci95'][1])}] | {100 * r['policy_acc']:.1f}% | {fb} |")
    L += ["", f"In-sample best policy: {S['in_sample_best_policy_bedrock_n2']}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e20a.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "research_analyses" / "E20a" / "data"))
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
