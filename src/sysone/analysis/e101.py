"""E.101 analysis: does the reviewer defer to Jev's errors more than to random ones?

Usage:
  uv run python -m sysone.analysis.e101 results/runs/<run_id>

Writes summary.json and report.md into the run folder. Per reviewer and sample:
- Jev: accuracy, error rate by true class and slice.
- Reviewer alone: accuracy of each sample and of the majority, and each item's difficulty
  for the reviewer (right in all, some or none of its alone samples).
- Per condition (one review per item): accuracy; acceptance when the proposal is right;
  override when it is wrong; the **flip rate**, the share of wrong proposals the reviewer
  accepts on items it gets right alone (majority; strict: right in every alone sample);
  the rescue rate (right proposals it accepts on items it gets wrong alone); parse and
  verdict/label consistency rates; tokens.
- Jev vs. matched random: the difference in flip rate with a bootstrap 95% interval, overall
  and within each slice (reviewer difficulty, annotator agreement, the dataset's own slice).
- Go/no-go checks (GO_NO_GO below), decided before the run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sysone.cascade.proposals import ALONE, JEV_CONF, JEV_LABEL, MATCHED_CONF, MATCHED_LABEL

# Decided before the live run (research_docs/13).
GO_NO_GO = {
    "jev_error_rate": (0.10, 0.35),
    "min_eligible": 20,       # Jev-wrong items the reviewer gets right alone
    "min_flip_rate": 0.05,    # and the Wilson 95% lower bound above 0
    "min_parse_rate": 0.95,
}
CONTRASTS = ((JEV_LABEL, MATCHED_LABEL), (JEV_CONF, MATCHED_CONF))
BOOTSTRAP_REPS = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(flags: list[bool]) -> dict[str, Any]:
    k, n = sum(flags), len(flags)
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None}


def bootstrap_diff(a: list[bool], b: list[bool], *, seed: int = 0, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    """mean(a) - mean(b), with a percentile bootstrap 95% interval (each arm resampled on its own)."""
    if not a or not b:
        return {"diff": None, "ci95": None, "n_a": len(a), "n_b": len(b)}
    rng = random.Random(seed)
    diffs = sorted(sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b)
                   for _ in range(reps))
    return {"diff": round(sum(a) / len(a) - sum(b) / len(b), 4),
            "ci95": [round(diffs[int(0.025 * reps)], 4), round(diffs[int(0.975 * reps) - 1], 4)],
            "n_a": len(a), "n_b": len(b)}


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def difficulty(k_right: int, n: int) -> str:
    return "always_right" if k_right == n else "never_right" if k_right == 0 else "sometimes_right"


def summarize(jev_rows: list[dict], proposals: list[dict], reviews: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"jev": {}, "reviewers": {}}
    # Jev, per sample.
    by_sample = defaultdict(list)
    for r in jev_rows:
        by_sample[r["sample"]].append(r)
    jev_prop = {(p["sample"], p["item_id"]): p for p in proposals if p["condition"] == JEV_LABEL}
    items: dict[tuple[str, str], dict] = {}
    for r in jev_rows:
        items.setdefault((r["sample"], r["item_id"]), {"label": r["label"], "strata": r["strata"]})
    for sample, rows in sorted(by_sample.items()):
        ok = [r for r in rows if r["status"] == "ok"]
        props = [p for (s, _), p in jev_prop.items() if s == sample]
        strata = defaultdict(list)
        for p in props:
            info = items[(sample, p["item_id"])]
            strata["label=" + info["label"]].append(not p["correct"])
            for k, v in info["strata"].items():
                strata[f"{k}={v}"].append(not p["correct"])
        out["jev"][sample] = {
            "n_requests": len(rows), "n_ok": len(ok),
            "parse_rate": rate([r["parse_ok"] for r in ok]),
            "n_items_decided": len(props),
            "error_rate": rate([not p["correct"] for p in props]),
            "split_rate": rate([bool(p.get("jev_split")) for p in props]),
            "error_rate_by_slice": {k: rate(v) for k, v in sorted(strata.items())},
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in ok), 6),
        }

    ok_reviews = [r for r in reviews if r["status"] == "ok"]
    for reviewer in sorted({r["reviewer"] for r in ok_reviews}):
        out["reviewers"][reviewer] = {}
        for sample in sorted({r["sample"] for r in ok_reviews if r["reviewer"] == reviewer}):
            rows = [r for r in ok_reviews if r["reviewer"] == reviewer and r["sample"] == sample]
            out["reviewers"][reviewer][sample] = _summarize_reviewer(rows, out["jev"].get(sample, {}))
    return out


def _summarize_reviewer(rows: list[dict], jev: dict) -> dict[str, Any]:
    alone: dict[str, list[bool]] = defaultdict(list)
    info: dict[str, dict] = {}
    for r in rows:
        info.setdefault(r["item_id"], {"label": r["label"], "strata": r["strata"]})
        if r["condition"] == ALONE and r["parse_ok"]:
            alone[r["item_id"]].append(r["final_label"] == r["label"])
    alone_majority = {i: sum(v) > len(v) / 2 for i, v in alone.items()}
    alone_difficulty = {i: difficulty(sum(v), len(v)) for i, v in alone.items()}

    def slices(item_id: str) -> list[str]:
        s = [f"reviewer_alone={alone_difficulty.get(item_id, 'unknown')}"]
        return s + [f"{k}={v}" for k, v in sorted(info[item_id]["strata"].items())]

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    conditions: dict[str, Any] = {}
    flip_flags: dict[str, list[tuple[str, bool]]] = {}
    for cond, crow in sorted(by_condition.items()):
        parsed = [r for r in crow if r["parse_ok"]]
        entry: dict[str, Any] = {
            "n_requests": len(crow),
            "parse_rate": rate([r["parse_ok"] for r in crow]),
            "accuracy": rate([r["final_label"] == r["label"] for r in parsed]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in crow if r.get("output_tokens") is not None]),
            "mean_input_tokens": _mean([r["input_tokens"] for r in crow if r.get("input_tokens") is not None]),
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in crow), 6),
            "service_tiers": dict(Counter(str(r.get("service_tier")) for r in crow)),
        }
        if cond != ALONE:
            right = [r for r in parsed if r["proposed_correct"]]
            wrong = [r for r in parsed if not r["proposed_correct"]]
            eligible = [r for r in wrong if alone_majority.get(r["item_id"])]
            strict = [r for r in wrong if alone_difficulty.get(r["item_id"]) == "always_right"]
            rescue = [r for r in right if r["item_id"] in alone_majority and not alone_majority[r["item_id"]]]
            accepted = lambda r: r["final_label"] == r["proposed_label"]
            entry.update({
                "proposal_error_rate": rate([not r["proposed_correct"] for r in parsed]),
                "accept_when_right": rate([accepted(r) for r in right]),
                "override_when_wrong": rate([not accepted(r) for r in wrong]),
                "flip_rate": rate([accepted(r) for r in eligible]),
                "flip_rate_strict": rate([accepted(r) for r in strict]),
                "rescue_rate": rate([accepted(r) for r in rescue]),
                "consistency_rate": rate([bool(r["consistent"]) for r in parsed]),
            })
            flip_flags[cond] = [(r["item_id"], accepted(r)) for r in eligible]
            # Override when wrong, by slice: the difficulty-controlled view (all wrong
            # proposals, not only those on items the reviewer gets right alone).
            by_slice = defaultdict(list)
            for r in wrong:
                for s in slices(r["item_id"]):
                    by_slice[s].append(accepted(r))
            entry["accept_wrong_by_slice"] = {k: rate(v) for k, v in sorted(by_slice.items())}
        conditions[cond] = entry

    contrasts = {}
    for a, b in CONTRASTS:
        if a not in flip_flags or b not in flip_flags:
            continue
        fa, fb = flip_flags[a], flip_flags[b]
        within = {}
        for s in sorted({s for i, _ in fa + fb for s in slices(i)}):
            within[s] = bootstrap_diff([f for i, f in fa if s in slices(i)], [f for i, f in fb if s in slices(i)])
        contrasts[f"{a}-{b}"] = {"flip_rate_diff": bootstrap_diff([f for _, f in fa], [f for _, f in fb]),
                                 "within_slice": within}

    alone_rows = by_condition.get(ALONE, [])
    summary = {
        "n_items": len(info),
        "alone_accuracy_per_sample": rate([r["final_label"] == r["label"] for r in alone_rows if r["parse_ok"]]),
        "alone_accuracy_majority": rate(list(alone_majority.values())),
        "alone_difficulty": dict(sorted(Counter(alone_difficulty.values()).items())),
        "conditions": conditions,
        "contrasts": contrasts,
    }
    summary["go_no_go"] = go_no_go(summary, jev)
    return summary


def go_no_go(s: dict[str, Any], jev: dict[str, Any]) -> dict[str, Any]:
    lo, hi = GO_NO_GO["jev_error_rate"]
    jev_err = (jev.get("error_rate") or {}).get("rate")
    alone_acc = s["alone_accuracy_majority"]["rate"]
    flip = s["conditions"].get(JEV_LABEL, {}).get("flip_rate", {})
    parse_rates = [c["parse_rate"]["rate"] for c in s["conditions"].values()]
    jev_parse = (jev.get("parse_rate") or {}).get("rate")
    checks = {
        "jev_error_rate_in_range": jev_err is not None and lo <= jev_err <= hi,
        "reviewer_alone_beats_jev": alone_acc is not None and jev_err is not None and alone_acc > 1 - jev_err,
        "enough_eligible_items": (flip.get("n") or 0) >= GO_NO_GO["min_eligible"],
        "flip_rate_above_zero": flip.get("rate") is not None and flip["rate"] >= GO_NO_GO["min_flip_rate"]
        and flip["ci95"][0] > 0,
        "parsing_ok": bool(parse_rates) and min(r or 0 for r in parse_rates) >= GO_NO_GO["min_parse_rate"]
        and (jev_parse or 0) >= GO_NO_GO["min_parse_rate"],
    }
    return {"checks": checks, "go": all(checks.values())}


def _fmt(r: dict[str, Any] | None) -> str:
    if not r or r.get("rate") is None:
        return "–"
    ci = r["ci95"]
    return f"{r['rate']:.0%} ({r['k']}/{r['n']}; {ci[0]:.0%}–{ci[1]:.0%})"


def _fmt_diff(d: dict[str, Any]) -> str:
    if d["diff"] is None:
        return f"– (n {d['n_a']} vs {d['n_b']})"
    return f"{d['diff']:+.0%} [{d['ci95'][0]:+.0%}, {d['ci95'][1]:+.0%}] (n {d['n_a']} vs {d['n_b']})"


def report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [f"# E.101 report: {manifest.get('run_id', '')}", "",
             f"Status: {manifest.get('status')}. Spent ${manifest.get('spent_usd', 0):.4f} "
             f"({manifest.get('spent_by_provider', {})}). Jev version: {manifest.get('model_version')}.", "",
             "Flip rate: of the wrong proposals on items the reviewer gets right alone (majority of its alone "
             "samples), the share it accepts. Rates show k/n and a Wilson 95% interval; differences show a "
             "bootstrap 95% interval.", "", "## Jev", "",
             "| Sample | Items | Parse rate | Error rate | Samples disagree |", "|---|---|---|---|---|"]
    for sample, j in summary["jev"].items():
        lines.append(f"| {sample} | {j['n_items_decided']} | {_fmt(j['parse_rate'])} | {_fmt(j['error_rate'])} "
                     f"| {_fmt(j['split_rate'])} |")
    for sample, j in summary["jev"].items():
        lines += ["", f"Jev error rate by slice, {sample}:", "", "| Slice | Error rate |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in j["error_rate_by_slice"].items()]
    for reviewer, by_sample in summary["reviewers"].items():
        for sample, s in by_sample.items():
            g = s["go_no_go"]
            lines += ["", f"## {reviewer} on {sample}", "",
                      f"**Go/no-go: {'GO' if g['go'] else 'NO-GO'}** — "
                      + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in g["checks"].items()), "",
                      f"Reviewer alone: {_fmt(s['alone_accuracy_per_sample'])} per sample, "
                      f"{_fmt(s['alone_accuracy_majority'])} by majority. Item difficulty for the reviewer: "
                      f"{s['alone_difficulty']}.", "",
                      "| Condition | Accuracy | Proposal wrong | Accept when right | Override when wrong | "
                      "Flip rate | Flip (strict) | Rescue | Parse | Out tokens |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for cond, c in s["conditions"].items():
                lines.append(f"| {cond} | {_fmt(c['accuracy'])} | {_fmt(c.get('proposal_error_rate'))} | "
                             f"{_fmt(c.get('accept_when_right'))} | {_fmt(c.get('override_when_wrong'))} | "
                             f"{_fmt(c.get('flip_rate'))} | {_fmt(c.get('flip_rate_strict'))} | "
                             f"{_fmt(c.get('rescue_rate'))} | {_fmt(c['parse_rate'])} | {c['mean_output_tokens']} |")
            for name, c in s["contrasts"].items():
                lines += ["", f"Flip rate, {name}: {_fmt_diff(c['flip_rate_diff'])}", "",
                          "| Slice | Difference |", "|---|---|"]
                lines += [f"| {k} | {_fmt_diff(v)} |" for k, v in c["within_slice"].items()]
            for cond in (JEV_LABEL, MATCHED_LABEL):
                c = s["conditions"].get(cond)
                if c:
                    lines += ["", f"Accepting a wrong proposal by slice, {cond}:", "", "| Slice | Accepted |", "|---|---|"]
                    lines += [f"| {k} | {_fmt(v)} |" for k, v in c["accept_wrong_by_slice"].items()]
    return "\n".join(lines) + "\n"


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary = summarize(_load(run_dir / "jev_decisions.jsonl"), _load(run_dir / "proposals.jsonl"),
                        _load(run_dir / "reviews.jsonl"))
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(report(summary, manifest))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    analyze_run(args.run_dir)
    print((Path(args.run_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
