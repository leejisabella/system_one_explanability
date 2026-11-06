"""Step 0 of round 2 (research_docs/15): is WildGuard's group C a benchmark ceiling?

Usage:  uv run python research_analyses/E108/step0_group_c.py

Draws 15 WildGuard prompts and 15 WildGuard responses from group C (Jev wrong, and the base never
right in 5 tries alone, from E.106's base run), at random by hash, and writes a reading sheet:
the text, the gold label, Jev's label and the base's 5 alone answers with their reasoning.
The sheet holds WildGuard's harmful text, so it goes to the gitignored run folder
(results/runs/e106-v1-2026-11-04/step0/), never to this folder. What gets committed here is
data/group_c_readings.csv: one row per item with its ID and the reader's classification
(doubtful_gold, ambiguous or model_mistake) and a one-line note, filled in by hand.

The rule, fixed before reading: 10 or more model_mistake of 30 means the ceiling assumption
doesn't hold for WildGuard.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from sysone.analysis.e102 import _load
from sysone.analysis.e106 import GROUP_C, base_groups
from sysone.cascade.items import load_items
from sysone.registry import REPO_ROOT

HERE = Path(__file__).resolve().parent
RUN_DIR = REPO_ROOT / "results" / "runs" / "e106-v1-2026-11-04"
PLAN = REPO_ROOT / "results" / "runs" / "e102-v1-2026-11-04" / "plan.jsonl"
POOLS = {"wildguard_prompt_full_v1": 15, "wildguard_response_full_v1": 15}
SEED = "e108-step0"
CLASSES = ("doubtful_gold", "ambiguous", "model_mistake")


def by_hash(sample: str, item_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{sample}:{item_id}".encode()).hexdigest()


def main() -> None:
    reviews = [r for r in _load(RUN_DIR / "reviews.jsonl") if r["status"] == "ok" and r["parse_ok"]]
    groups = base_groups(reviews, "base_local")
    plan = {(p["sample"], p["item_id"]): p for p in _load(PLAN)}
    jev_wrong = {k for k, p in plan.items() if not p["jev_correct"]}
    print("Jev's mistakes by pool and group:",
          dict(sorted(Counter((k[0], groups.get(k)) for k in jev_wrong).items())))
    alone = {}
    for r in reviews:
        if r["reviewer"] == "base_local" and r["condition"] == "alone":
            alone.setdefault((r["sample"], r["item_id"]), []).append(r)

    out_dir = RUN_DIR / "step0"
    out_dir.mkdir(exist_ok=True)
    sheet, rows = [], []
    for sample, n in POOLS.items():
        texts = {i.item_id: i for i in load_items(sample)[0]}
        pick = sorted((k for k in jev_wrong if k[0] == sample and groups.get(k) == GROUP_C), key=lambda k: by_hash(*k))[:n]
        for key in sorted(pick):
            p, item = plan[key], texts[key[1]]
            sheet += [f"## {sample} / {key[1]}", "", f"Gold: **{p['label']}**. Jev: {p['jev_label']}. "
                      f"Strata: {p['strata']}", "", "```", item.text, "```", ""]
            for r in sorted(alone[key], key=lambda r: r["sample_index"]):
                sheet += [f"- Alone try {r['sample_index']}: **{r['final_label']}**. "
                          + (r.get("reasoning") or "").replace("\n", " ")[:600], ""]
            rows.append({"sample": sample, "item_id": key[1], "gold": p["label"], "jev": p["jev_label"],
                         "classification": "", "note": ""})
    (out_dir / "reading_sheet.md").write_text("# Step 0: WildGuard group C reading sheet\n\n"
                                              f"Classify each as one of {', '.join(CLASSES)}.\n\n" + "\n".join(sheet))
    target = HERE / "data" / "group_c_readings.csv"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        print(f"{target} exists; not overwritten (it holds the readings)")
    else:
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    filled = [r for r in csv.DictReader(target.open()) if r["classification"]]
    counts = Counter(r["classification"] for r in filled)
    print(f"wrote {out_dir / 'reading_sheet.md'} ({len(rows)} items); readings so far: {dict(counts)}")
    if len(filled) == len(rows):
        verdict = "does NOT hold" if counts["model_mistake"] >= 10 else "holds"
        print(f"rule: {counts['model_mistake']} model_mistake of {len(rows)} -> the ceiling assumption {verdict} for WildGuard")


if __name__ == "__main__":
    main()
