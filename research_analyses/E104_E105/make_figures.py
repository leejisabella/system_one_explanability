"""E.104 and E.105 data tables, regenerated from the saved runs. Tables only: no figures.

Usage:  uv run python research_analyses/E104_E105/make_figures.py

Reads results/runs/e104-v1-2026-11-04/ and e105-v1-2026-11-04/ (both local only), the exported
datasets in data/posttrain/ and the training pools, and writes derived tables to data/ (committed,
so the numbers survive if results/ is lost). No API calls.

The 30-item check of dropped Civil Comments items is a hand reading: REVIEW below records the
category given to each sampled item, and the script re-draws the same 30 (seed 105).
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from sysone.experiments.e101 import load_samples
from sysone.experiments.e104 import E104Config
from sysone.registry import CONFIG_DIR, REPO_ROOT

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
E104, E105 = "e104-v1-2026-11-04", "e105-v1-2026-11-04"
RUNS = REPO_ROOT / "results" / "runs"
DATASETS = ["part2_all_v1", "part2_no_responses_v1"]
CIVIL = "civil_toxicity_train_v1"
SEED, N_REVIEW = 105, 30

# The hand reading of the 30 sampled dropped items (Civil Comments, Jev wrong, never solved).
#   gold_doubtful: the gold label looks wrong, or rests on a 5-of-10 split, and the models look right
#   models_stricter: mild snark the annotators let pass and the models call toxic
#   ambiguous: either label is defensible
#   model_miss: the models miss something a careful reader would catch (none were found)
REVIEW = {
    "cc-7184599": "gold_doubtful", "cc-7127486": "models_stricter", "cc-7070555": "ambiguous",
    "cc-7126799": "gold_doubtful", "cc-7169670": "ambiguous", "cc-7020137": "ambiguous",
    "cc-7000249": "gold_doubtful", "cc-7063078": "models_stricter", "cc-7116766": "gold_doubtful",
    "cc-7167551": "gold_doubtful", "cc-7125329": "models_stricter", "cc-7068513": "models_stricter",
    "cc-7060069": "models_stricter", "cc-7030273": "gold_doubtful", "cc-7179513": "ambiguous",
    "cc-7030462": "gold_doubtful", "cc-7163782": "models_stricter", "cc-7186893": "gold_doubtful",
    "cc-7027569": "gold_doubtful", "cc-7019080": "gold_doubtful", "cc-7141882": "gold_doubtful",
    "cc-7081012": "models_stricter", "cc-7071003": "gold_doubtful", "cc-7030531": "gold_doubtful",
    "cc-7179180": "ambiguous", "cc-7089838": "ambiguous", "cc-7132414": "gold_doubtful",
    "cc-7078316": "models_stricter", "cc-7177340": "gold_doubtful", "cc-7097291": "models_stricter",
}


def write_csv(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for run in (E104, E105):
        shutil.copy(RUNS / run / "report.md", OUT / f"{run.split('-')[0]}_report.md")
    m104 = json.loads((RUNS / E104 / "manifest.json").read_text())
    m105 = json.loads((RUNS / E105 / "manifest.json").read_text())
    json.dump({run: {k: m[k] for k in ("run_id", "git", "started_at", "finished_at", "spent_usd", "status")}
               for run, m in ((E104, m104), (E105, m105))}, open(OUT / "runs.json", "w"), indent=1)

    samples, _ = load_samples(E104Config.load(CONFIG_DIR / "experiments" / "e104.yaml"))
    items = {i.item_id: i for its in samples.values() for i in its}
    mix = jsonl(RUNS / E104 / "mix.jsonl")
    traces: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for t in jsonl(RUNS / E105 / "traces.jsonl"):
        traces[(t["sample"], t["item_id"], t["format"])].append(t)

    def solved(sample: str, iid: str) -> bool:
        return any(t["parse_ok"] and t["final_label"] == t["label"]
                   for fmt in ("alone", "review") for t in traces[(sample, iid, fmt)])

    # Jev's error rate on each training pool, beside its rate on the matching test pool (E.102).
    write_csv("jev_error_rates.csv", [{"pool": s, "jev_error_rate": r}
                                      for s, r in m104["jev_error_rate_by_sample"].items()])

    # How many examples of each group reached a correct trace, and who wrote it.
    rows = []
    by = defaultdict(list)
    for m in mix:
        by[(m["sample"], m["group"])].append(m["item_id"])
    for (sample, group), iids in sorted(by.items()):
        n = len(iids)
        s = sum(solved(sample, i) for i in iids)
        rows.append({"pool": sample, "group": group, "items": n, "solved_in_some_format": s,
                     "never_solved": n - s, "never_solved_share": round((n - s) / n, 4)})
    write_csv("solve_rates.csv", rows)

    # What ended up in each exported file, by group.
    rows = []
    for ds in DATASETS:
        man = json.loads((REPO_ROOT / "benchmarks" / "posttrain" / ds / "manifest.json").read_text())
        for fname, info in sorted(man["files"].items()):
            total = info["n"]
            for group, n in sorted(info["groups"].items()):
                rows.append({"dataset": ds, "file": fname, "group": group, "records": n,
                             "share": round(n / total, 4)})
    write_csv("dataset_composition.csv", rows)

    # Source of the SFT treatment traces (reviewer's own vs. teacher).
    rows = []
    for ds in DATASETS:
        for arm in ("sft_review", "sft_plain"):
            c = Counter((r["meta"].get("sample"), r["meta"].get("trace_source")) for r in
                        jsonl(REPO_ROOT / "data" / "posttrain" / ds / f"{arm}.train.jsonl"))
            for (sample, source), n in sorted(c.items(), key=str):
                rows.append({"dataset": ds, "arm": arm, "pool": sample, "trace_source": source, "records": n})
    write_csv("trace_sources.csv", rows)

    # Civil Comments, Jev wrong: kept vs. dropped, by how sure the annotators were.
    rows, dropped, unanimous = [], [], 0
    for kept in (True, False):
        iids = [i for i in by[(CIVIL, "jev_wrong")] if solved(CIVIL, i) == kept]
        tox = [items[i].meta["toxicity"] for i in iids]
        raters = [items[i].meta["toxicity_annotator_count"] for i in iids]
        rows.append({"jev_wrong_items": "kept" if kept else "dropped (never solved)", "n": len(iids),
                     "borderline_30_70": round(sum(0.3 <= x < 0.7 for x in tox) / len(tox), 4),
                     "median_distance_from_50pct": round(statistics.median(abs(x - 0.5) for x in tox), 3),
                     "median_raters": statistics.median(raters),
                     "share_5_or_fewer_raters": round(sum(n <= 5 for n in raters) / len(raters), 4),
                     "gold_non_toxic_share": round(sum(items[i].label == "non_toxic" for i in iids) / len(iids), 4)})
        if not kept:
            dropped = iids
            for i in iids:
                labels = {t["final_label"] for f in ("alone", "review") for t in traces[(CIVIL, i, f)] if t["parse_ok"]}
                unanimous += len(labels) == 1
    rows[-1]["models_unanimous_share"] = round(unanimous / len(dropped), 4)
    rows[0]["models_unanimous_share"] = ""
    write_csv("civil_dropped_vs_kept.csv", rows)

    random.seed(SEED)
    sample30 = random.sample(sorted(dropped), N_REVIEW)
    assert set(sample30) == set(REVIEW), "the 30-item draw changed; re-read the new items"
    rows = []
    for i in sample30:
        it = items[i]
        votes = Counter(t["final_label"] for f in ("alone", "review") for t in traces[(CIVIL, i, f)] if t["parse_ok"])
        rows.append({"item_id": i, "gold": it.label, "toxic_share": round(it.meta["toxicity"], 3),
                     "raters": it.meta["toxicity_annotator_count"], "model_votes": json.dumps(dict(votes)),
                     "category": REVIEW[i], "text": it.text})
    write_csv("civil_dropped_sample30.csv", rows)
    print(Counter(REVIEW.values()))
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
