"""Build E.24's minimal-pair probe set from the writers' drafts (research_docs/17, E.24).

Steps (each re-runnable; nothing calls a model):
  check    flatten drafts/*.jsonl into items, validate the triplet structure and labels, and compare
           every item with every cascade pool (test and training) by character 8-gram overlap.
           Writes leakage.json; items above the threshold must be rewritten.
  blind    write blind/labelset.jsonl: every item shuffled, with no label, family or pair id, for the
           independent subagent labellers (they write blind/labels_<name>.jsonl: {"key", "label"}).
  freeze   merge the blind labels, keep triplets whose every item both labellers agree with, and
           write items.jsonl (cascade Item format) plus manifest.json (counts, sha256, drops).

Usage:
  python3 scripts/build_probes_e24.py check
  python3 scripts/build_probes_e24.py blind
  python3 scripts/build_probes_e24.py freeze --labellers a b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "probes" / "minimal_pairs_v1"
POOLS = ROOT / "data" / "cascade"
ROLES = ("neutral", "trigger", "harmful")
LABELS = {"wildguard_prompt_harm": ("harmful", "unharmful"), "wildguard_response_harm": ("harmful", "unharmful"),
          "civil_comments_toxicity": ("toxic", "non_toxic")}
DATASET = {"wildguard_prompt_harm": "probe_wildguard_prompt", "wildguard_response_harm": "probe_wildguard_response",
           "civil_comments_toxicity": "probe_civil_toxicity"}
EXPECTED = {"P5": ("unharmful", "harmful", "harmful")}   # every other family: benign, benign, harmful
# Triplets the second minimality review (minimality_review_v2.jsonl) rated closest to failing: kept,
# but excluded in E.24's sensitivity analysis.
NEAR_FAIL = ["C2-16", "P4-18", "P1-19", "P3-18", "P4-11"]
NGRAM, THRESHOLD = 8, 0.5   # flag if half of an item's 8-grams appear in one pool item


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def grams(t: str) -> set[str]:
    t = _norm(t)
    return {t[i:i + NGRAM] for i in range(max(len(t) - NGRAM + 1, 1))}


def load_triplets() -> list[dict]:
    rows = []
    for f in sorted((OUT / "drafts").glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def flatten(triplets: list[dict]) -> list[dict]:
    items, problems = [], []
    for t in triplets:
        pos, neg = LABELS[t["task"]]
        want = EXPECTED.get(t["family"], (neg, neg, pos))
        for role, label in zip(ROLES, want):
            got = t[role]["label"]
            if got != label:
                problems.append(f"{t['pair_id']} {role}: label {got}, expected {label}")
            if t["task"] == "wildguard_response_harm" and not t[role]["text"].startswith("User prompt:\n"):
                problems.append(f"{t['pair_id']} {role}: response format")
            items.append({"item_id": f"probe-{t['pair_id']}-{role}", "dataset": DATASET[t["task"]], "task": t["task"],
                          "text": t[role]["text"], "label": got,
                          "strata": {"family": t["family"], "role": role},
                          "meta": {"pair_id": t["pair_id"], "family": t["family"], "role": role,
                                   "edit_note": t.get("edit_note", "")}})
    ids = Counter(i["item_id"] for i in items)
    problems += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    texts = Counter(_norm(i["text"]) for i in items)
    problems += [f"duplicate text: {k[:60]}" for k, v in texts.items() if v > 1]
    return items, problems


def check() -> None:
    triplets = load_triplets()
    items, problems = flatten(triplets)
    print(f"{len(triplets)} triplets, {len(items)} items; by family: {dict(Counter(t['family'] for t in triplets))}")
    for p in problems:
        print("PROBLEM", p)
    # Index the probe items' 8-grams (small) and stream the pools through it.
    probe_grams = [grams(it["text"]) for it in items]
    index: dict[str, list[int]] = {}
    for j, g in enumerate(probe_grams):
        for x in g:
            index.setdefault(x, []).append(j)
    best = [(0.0, None)] * len(items)
    n_pool = 0
    for f in sorted(POOLS.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            n_pool += 1
            hits = Counter(j for x in grams(d["text"]) if x in index for j in index[x])
            for j, n in hits.items():
                share = n / len(probe_grams[j])
                if share > best[j][0]:
                    best[j] = (share, f"{f.parent.name}/{d['item_id']}")
    print(f"compared against {n_pool} pool items")
    report = [{"item_id": it["item_id"], "max_share": round(b[0], 3), "nearest": b[1]} for it, b in zip(items, best)]
    report.sort(key=lambda r: -r["max_share"])
    flagged = [r for r in report if r["max_share"] >= THRESHOLD]
    (OUT / "leakage.json").write_text(json.dumps({"ngram": NGRAM, "threshold": THRESHOLD, "flagged": flagged,
                                                   "top20": report[:20]}, indent=2) + "\n")
    print(f"leakage: {len(flagged)} items at or above {THRESHOLD}; highest shares: "
          + ", ".join(f"{r['item_id']} {r['max_share']}" for r in report[:5]))


def blind() -> None:
    items, problems = flatten(load_triplets())
    if problems:
        raise SystemExit("fix the problems reported by `check` first")
    rng = random.Random(24)
    rows = [{"key": hashlib.sha256(i["item_id"].encode()).hexdigest()[:12], "task": i["task"], "text": i["text"]}
            for i in items]
    rng.shuffle(rows)
    (OUT / "blind").mkdir(exist_ok=True)
    (OUT / "blind" / "labelset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "blind" / "key_map.json").write_text(json.dumps(
        {hashlib.sha256(i["item_id"].encode()).hexdigest()[:12]: i["item_id"] for i in items}, indent=1) + "\n")
    print(f"wrote {len(rows)} blind items to {OUT / 'blind' / 'labelset.jsonl'}")


def freeze(labellers: list[str]) -> None:
    items, problems = flatten(load_triplets())
    if problems:
        raise SystemExit("fix the problems reported by `check` first")
    key_map = json.loads((OUT / "blind" / "key_map.json").read_text())
    votes: dict[str, dict[str, str]] = {}
    unsure: set[str] = set()
    for name in labellers:
        for line in (OUT / "blind" / f"labels_{name}.jsonl").read_text().splitlines():
            d = json.loads(line)
            votes.setdefault(key_map[d["key"]], {})[name] = d["label"]
            if not d.get("confident", True):
                unsure.add(key_map[d["key"]])
    bad_pairs: dict[str, list[str]] = {}
    for it in items:
        v = votes.get(it["item_id"], {})
        if len(v) < len(labellers) or any(x != it["label"] for x in v.values()):
            bad_pairs.setdefault(it["meta"]["pair_id"], []).append(f"{it['meta']['role']}: intended {it['label']}, got {v}")
    kept = [i for i in items if i["meta"]["pair_id"] not in bad_pairs]
    for i in kept:
        i["meta"]["labeller_unsure"] = i["item_id"] in unsure
        i["meta"]["near_fail_minimality"] = i["meta"]["pair_id"] in NEAR_FAIL
    text = "".join(json.dumps(i, sort_keys=True) + "\n" for i in kept)
    (OUT / "items.jsonl").write_text(text)
    manifest = {"name": "minimal_pairs_v1", "experiment": "E.24", "labellers": labellers,
                "triplets_kept": len(kept) // 3, "triplets_dropped": len(bad_pairs),
                "kept_by_family": dict(Counter(i["meta"]["family"] for i in kept if i["meta"]["role"] == "neutral")),
                "dropped": bad_pairs, "near_fail_minimality": NEAR_FAIL,
                "labeller_unsure_items": sorted(i["item_id"] for i in kept if i["meta"]["labeller_unsure"]), "items_sha256": hashlib.sha256(text.encode()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "dropped"}, indent=2))
    for p, why in bad_pairs.items():
        print("DROPPED", p, why)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "blind", "freeze"])
    ap.add_argument("--labellers", nargs="+", default=["a", "b"])
    a = ap.parse_args()
    {"check": check, "blind": blind, "freeze": lambda: freeze(a.labellers)}[a.step]()


if __name__ == "__main__":
    main()
