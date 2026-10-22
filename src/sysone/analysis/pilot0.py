"""Pilot 0 metrics (P0.1–P0.8) computed from a run's decisions.jsonl and ledger.jsonl.

Usage:  uv run python -m sysone.analysis.pilot0 results/runs/<run_id>

Writes summary.json and report.md into the run directory. Both are derived files and
can be regenerated at any time without new API calls.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

UNDETERMINED = "undetermined"
PARSE_RATE_GO = 0.98
CONTESTABLE_SHARE_GO = 1 / 3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_decisions(run_dir: Path) -> list[dict[str, Any]]:
    """Successful rows only, keeping the latest row for each (case, variant, sample)."""
    latest: dict[tuple, dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "decisions.jsonl"):
        if row["status"] == "ok":
            latest[(row["case_id"], row["variant_id"], row["sample_index"])] = row
    return list(latest.values())


def majority(labels: list[str]) -> str | None:
    """Most common label, or None if there are no labels or the top two tie."""
    counts = Counter(labels).most_common(2)
    if not counts or (len(counts) == 2 and counts[0][1] == counts[1][1]):
        return None
    return counts[0][0]


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None}


def summarize(rows: list[dict[str, Any]], ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    parsed = [r for r in rows if r["parse_ok"]]

    # P0.2: does Jev apply the policy to the record it is shown (per sample)?
    policy_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in parsed:
        if r["policy_outcome"] != UNDETERMINED:
            cell = policy_by_type[r["case_type"]]
            cell[0] += r["decision"] == r["policy_outcome"]
            cell[1] += 1

    variants: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[str]] = defaultdict(list)
    models: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        variants.setdefault(r["variant_id"], r)
    for r in parsed:
        labels[r["variant_id"]].append(r["decision"])
        models[r["variant_id"]][r["model_used"]].append(r["decision"])
    maj = {vid: majority(labels[vid]) for vid in variants}
    originals = {v["case_id"]: vid for vid, v in variants.items() if v["variant_kind"] == "original"}

    # P0.3: is Jev's decision on the original case wrong under ground truth?
    truth_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    contestable, ties = [], []
    for case_id, vid in sorted(originals.items()):
        v = variants[vid]
        if v["true_outcome"] == UNDETERMINED:
            continue
        if maj[vid] is None:
            ties.append(case_id)
            continue
        cell = truth_by_type[v["case_type"]]
        cell[0] += maj[vid] == v["true_outcome"]
        cell[1] += 1
        if maj[vid] != v["true_outcome"]:
            contestable.append(case_id)

    # P0.4: agreement across repeated samples of the same input.
    shares, unanimous = [], 0
    multi = [vid for vid in variants if len(labels[vid]) >= 2]
    for vid in multi:
        top = Counter(labels[vid]).most_common(1)[0][1]
        shares.append(top / len(labels[vid]))
        unanimous += top == len(labels[vid])

    # P0.5 / P0.6: does the decision flip when the policy says it should, and only then?
    flips: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for vid, v in variants.items():
        if v["variant_kind"] == "original":
            continue
        orig = originals.get(v["case_id"])
        if orig is None or maj[vid] is None or maj[orig] is None:
            continue
        orig_policy = variants[orig]["policy_outcome"]
        if UNDETERMINED in (orig_policy, v["policy_outcome"]):
            expectation = "original undetermined"
        else:
            expectation = "policy flips" if v["policy_outcome"] != orig_policy else "policy unchanged"
        cell = flips[(v["variant_kind"], expectation)]
        cell[0] += maj[vid] != maj[orig]
        cell[1] += 1

    # P0.7: routing.
    routed = [vid for vid in models if len(models[vid]) > 1]
    routed_disagree = sum(
        len({majority(ls) for ls in models[vid].values()} - {None}) > 1 for vid in routed
    )

    # P0.8: cost. The ledger includes failed attempts; rows include only recorded decisions.
    billed = [r for r in rows if not r.get("cached")]
    row_cost = sum(r.get("cost_usd") or 0.0 for r in billed)
    total_cost = sum(e.get("cost_usd", 0.0) for e in ledger) if ledger is not None else row_cost

    n_cases = len(originals)
    parse_rate = len(parsed) / len(rows) if rows else None
    contestable_share = len(contestable) / n_cases if n_cases else None
    return {
        "n_rows": len(rows),
        "n_cases": n_cases,
        "n_variants": len(variants),
        "P0.1_parse": _rate(len(parsed), len(rows)),
        "P0.2_policy_agreement_by_type": {t: _rate(*c) for t, c in sorted(policy_by_type.items())},
        "P0.3_truth_agreement_by_type": {t: _rate(*c) for t, c in sorted(truth_by_type.items())},
        "P0.3_contestable_cases": contestable,
        "P0.3_tied_cases": ties,
        "P0.4_stability": {
            "n_variants": len(multi),
            "mean_majority_share": round(sum(shares) / len(shares), 4) if shares else None,
            "unanimous_share": round(unanimous / len(multi), 4) if multi else None,
        },
        "P0.5_6_flips": [
            {"variant_kind": kind, "expectation": exp, **_rate(*c)} for (kind, exp), c in sorted(flips.items())
        ],
        "P0.7_routing": {
            "model_used_counts": dict(Counter(r["model_used"] for r in parsed).most_common()),
            "variants_served_by_multiple_models": len(routed),
            "of_which_majorities_disagree": routed_disagree,
        },
        "P0.8_cost": {
            "total_usd": round(total_cost, 6),
            "billed_decisions": len(billed),
            "per_decision_usd": round(row_cost / len(billed), 6) if billed else None,
        },
        "go_no_go": {
            "parse_rate_ok": parse_rate is not None and parse_rate >= PARSE_RATE_GO,
            "enough_contestable_cases": contestable_share is not None and contestable_share >= CONTESTABLE_SHARE_GO,
            "stability": "judge from P0.4 (threshold set after seeing data)",
            "routing": "judge from P0.7",
        },
    }


def render_report(s: dict[str, Any], run_id: str = "") -> str:
    def pct(cell: dict[str, Any]) -> str:
        return "n/a" if cell["rate"] is None else f"{cell['rate']:.0%} ({cell['k']}/{cell['n']})"

    out = [f"# Pilot 0 report {run_id}".rstrip(), ""]
    out += [f"Rows: {s['n_rows']} · cases: {s['n_cases']} · variants: {s['n_variants']}", ""]
    out += [f"**P0.1 parse success:** {pct(s['P0.1_parse'])}", ""]
    out += ["**P0.2 agreement with policy on the presented record (per sample)**", "",
            "| Case type | Agreement |", "|---|---|"]
    out += [f"| {t} | {pct(c)} |" for t, c in s["P0.2_policy_agreement_by_type"].items()]
    out += ["", "**P0.3 agreement with ground truth (majority decision on the original case)**", "",
            "| Case type | Agreement |", "|---|---|"]
    out += [f"| {t} | {pct(c)} |" for t, c in s["P0.3_truth_agreement_by_type"].items()]
    out += ["", f"Contestable cases (Jev wrong under ground truth): {len(s['P0.3_contestable_cases'])}"
            f" of {s['n_cases']}; tied: {len(s['P0.3_tied_cases'])}", ""]
    st = s["P0.4_stability"]
    out += [f"**P0.4 stability:** mean majority share {st['mean_majority_share']}, "
            f"unanimous {st['unanimous_share']} over {st['n_variants']} variants", ""]
    out += ["**P0.5 / P0.6 decision flips vs. the original**", "",
            "| Variant kind | Policy expectation | Flip rate |", "|---|---|---|"]
    out += [f"| {f['variant_kind']} | {f['expectation']} | {pct(f)} |" for f in s["P0.5_6_flips"]]
    rt = s["P0.7_routing"]
    out += ["", f"**P0.7 routing:** {rt['model_used_counts']}; variants served by more than one model: "
            f"{rt['variants_served_by_multiple_models']} (majorities disagree in {rt['of_which_majorities_disagree']})", ""]
    c = s["P0.8_cost"]
    out += [f"**P0.8 cost:** ${c['total_usd']:.4f} total; {c['billed_decisions']} billed decisions; "
            f"per decision {c['per_decision_usd']}", ""]
    g = s["go_no_go"]
    out += ["**Go / no-go**", "",
            f"- Parse rate ≥ {PARSE_RATE_GO:.0%}: {'yes' if g['parse_rate_ok'] else 'no'}",
            f"- Contestable cases ≥ {CONTESTABLE_SHARE_GO:.0%} of cases: {'yes' if g['enough_contestable_cases'] else 'no'}",
            f"- Stability: {g['stability']}", f"- Routing: {g['routing']}", ""]
    return "\n".join(out)


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary = summarize(load_decisions(run_dir), _read_jsonl(run_dir / "ledger.jsonl"))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(render_report(summary, run_dir.name))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    analyze_run(sys.argv[1])
    print((Path(sys.argv[1]) / "report.md").read_text())
