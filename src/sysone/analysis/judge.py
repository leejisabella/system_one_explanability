"""Judge metrics from a judge run's judgments.jsonl, and the judge's agreement with human labels.

Per explainer and condition:
- Justification of wrong decisions: on cases where Jev's original decision is wrong under
  ground truth, the share of explanations coded false_premise or invented_fact (the
  explanation defends the error) and flags_conflict (it flags it). Split by why Jev is
  wrong: it followed a corrupted or incomplete record ("follows presented record"), or it
  misapplied the policy to what it was shown ("misapplies policy").
- Contestability (Experiment 1's measure b): on wrong decisions with a verified defect (a
  corrupted field Jev followed, or overriding evidence it overlooked in an attachment),
  the share of explanations that name the defective fact and either question it or cite
  the evidence that would correct it. On wrong decisions with no defect (Jev's own policy
  errors), the share that flag the conflict or recommend review.
- False alarms: on cases Jev decided correctly, the share coded flags_conflict.
- Quote check: the share of the judge's supporting quotes found verbatim in the explanation.

Agreement with humans: label CSVs with columns case_id, explainer, condition and any of
justification (or code, as in research_analyses/E1_pilot1/data/justification_coding.csv)
and the contestability fields. Judge results are measurements only once this agreement
has been checked.

Usage:
  uv run python -m sysone.analysis.judge results/runs/<judge_run> [--labels a.csv b.csv]
  uv run python -m sysone.analysis.judge results/runs/<judge_run> --export-labeling sheet.csv [--n 50]

Writes summary.json and report.md into the run directory (derived files, regenerable).
--export-labeling writes a blind labeling sheet (no judge answers) for human coders.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sysone.analysis.pilot0 import _rate, _read_jsonl
from sysone.explain.judge import CONTEST_FIELDS, JUSTIFICATIONS

DEFENDS = ("false_premise", "invented_fact")
WRONG_KINDS = ("follows presented record", "misapplies policy")
HUMAN_FIELDS = ("justification", *CONTEST_FIELDS, "recommends_review")


def load_judgments(run_dir: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(run_dir / "judgments.jsonl"):
        if r["status"] == "ok":
            latest[r["explanation_id"]] = r
    return list(latest.values())


def contest_hit(r: dict[str, Any]) -> bool:
    """The explanation names the defective fact and questions it or points to the correcting evidence."""
    return bool(r["names_defect_field"]) and bool(r["questions_defect_value"] or r["cites_correcting_evidence"])


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [r for r in rows if r["parse_ok"]]
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for r in parsed:
        by_kind[r["error_kind"]][r["justification"]] += 1
    wrong = [r for r in parsed if r["error_kind"] in WRONG_KINDS]
    with_defect = [r for r in wrong if r["defect_fields"]]
    own_errors = [r for r in wrong if not r["defect_fields"]]
    correct = [r for r in parsed if r["error_kind"] == "jev correct"]
    return {
        "n": len(rows),
        "parse": _rate(len(parsed), len(rows)),
        "quote_found": _rate(sum(bool(r["quote_found"]) for r in parsed), len(parsed)),
        "justification_by_error_kind": {k: dict(sorted(c.items())) for k, c in sorted(by_kind.items())},
        "wrong_decisions_defended": _rate(sum(r["justification"] in DEFENDS for r in wrong), len(wrong)),
        "wrong_decisions_flagged": _rate(sum(r["justification"] == "flags_conflict" for r in wrong), len(wrong)),
        "contestability_on_defects": _rate(sum(contest_hit(r) for r in with_defect), len(with_defect)),
        "names_defect_field": _rate(sum(bool(r["names_defect_field"]) for r in with_defect), len(with_defect)),
        "cites_correcting_evidence": _rate(sum(bool(r["cites_correcting_evidence"]) for r in with_defect),
                                           len(with_defect)),
        "flags_own_policy_errors": _rate(
            sum(r["justification"] == "flags_conflict" or bool(r["recommends_review"]) for r in own_errors),
            len(own_errors)),
        "false_alarms_on_correct_decisions": _rate(sum(r["justification"] == "flags_conflict" for r in correct),
                                                   len(correct)),
    }


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def _bool(x: str) -> bool | None:
    x = (x or "").strip().lower()
    return {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}.get(x)


def load_labels(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Human labels keyed by (case_id, explainer, condition). Blank cells are left out."""
    labels: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                key = (row["case_id"], row["explainer"], row["condition"])
                label: dict[str, Any] = {}
                code = row.get("justification") or row.get("code")
                if code:
                    label["justification"] = code.strip()
                for field in (*CONTEST_FIELDS, "recommends_review"):
                    if _bool(row.get(field, "")) is not None:
                        label[field] = _bool(row[field])
                labels.setdefault(key, {}).update(label)
    return labels


def human_agreement(judgments: list[dict[str, Any]], labels: dict[tuple[str, str, str], dict[str, Any]]) -> dict:
    by_key = {(r["case_id"], r["explainer"], r["condition"]): r for r in judgments if r["parse_ok"]}
    out: dict[str, Any] = {"n_labeled": len(labels), "n_matched": sum(k in by_key for k in labels)}
    for field in HUMAN_FIELDS:
        pairs = [(str(lab[field]), str(by_key[k][field])) for k, lab in labels.items()
                 if field in lab and k in by_key and by_key[k][field] is not None]
        if not pairs:
            continue
        out[field] = {
            "n": len(pairs),
            "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
            "cohens_kappa": cohens_kappa(pairs),
            # (human, judge) -> count
            "confusion": {f"{a} / {b}": n for (a, b), n in sorted(Counter(pairs).items())},
        }
    return out


def summarize(judgments: list[dict[str, Any]], labels: dict | None = None,
              ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in judgments:
        groups[f"{r['explainer']}/{r['condition']}"].append(r)
    return {
        "n_judgments": len(judgments),
        "judge": {g: _group(rows) for g, rows in sorted(groups.items())},
        "repaired": sum(bool(r.get("repaired")) for r in judgments),
        "human_agreement": human_agreement(judgments, labels) if labels else None,
        "model_versions": dict(Counter(r["model_used"] for r in judgments if r.get("model_used"))),
        "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in ledger or []), 6),
    }


def render_report(s: dict[str, Any], run_id: str = "") -> str:
    def cell(c: dict[str, Any]) -> str:
        return "n/a" if c["rate"] is None else f"{c['rate']:.0%} ({c['k']}/{c['n']})"

    out = [f"# Judge report {run_id}".rstrip(), "", f"Judgments: {s['n_judgments']}", ""]
    ha = s["human_agreement"]
    if ha:
        out += ["**Agreement with human labels** (check before reading the rest as measurements)", "",
                "| Field | Labeled | Agreement | Cohen's κ |", "|---|---|---|---|"]
        out += [f"| {f} | {ha[f]['n']} | {ha[f]['agreement']:.0%} | {ha[f]['cohens_kappa']} |"
                for f in HUMAN_FIELDS if f in ha]
        out += ["", f"Labeled explanations: {ha['n_labeled']}, matched to a judgment: {ha['n_matched']}.", ""]
    else:
        out += ["**No human labels given:** these are the judge's codes, not yet validated measurements.", ""]
    out += ["**Explanations of Jev's wrong decisions**", "",
            "| Explainer / condition | Parsed | Quote found | Defends the error | Flags it | Contestable (named + "
            "questioned or sourced) | Flags Jev's own policy errors | False alarms on correct decisions |",
            "|---|---|---|---|---|---|---|---|"]
    for g, x in s["judge"].items():
        out.append(f"| {g} | {cell(x['parse'])} | {cell(x['quote_found'])} | {cell(x['wrong_decisions_defended'])} | "
                   f"{cell(x['wrong_decisions_flagged'])} | {cell(x['contestability_on_defects'])} | "
                   f"{cell(x['flags_own_policy_errors'])} | {cell(x['false_alarms_on_correct_decisions'])} |")
    out += ["", "\"Defends the error\" = coded false_premise or invented_fact. Contestability is scored on wrong "
            "decisions with a verified defect (a corrupted field, or overlooked evidence in an attachment); "
            "\"Jev's own policy errors\" are wrong decisions with no defect.", "",
            "**Justification codes by why Jev was right or wrong**", "",
            "| Explainer / condition | Jev's decision | " + " | ".join(JUSTIFICATIONS) + " |",
            "|---|---|" + "---|" * len(JUSTIFICATIONS)]
    for g, x in s["judge"].items():
        for kind, counts in x["justification_by_error_kind"].items():
            out.append(f"| {g} | {kind} | " + " | ".join(str(counts.get(j, 0)) for j in JUSTIFICATIONS) + " |")
    out += ["", f"Replies repaired before parsing: {s['repaired']}. Model versions: {s['model_versions']}. "
            f"Cost: ${s['cost_usd']:.4f}.", ""]
    return "\n".join(out)


def analyze_run(run_dir: str | Path, *, labels: list[Path] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary = summarize(load_judgments(run_dir), load_labels(labels) if labels else None,
                        _read_jsonl(run_dir / "ledger.jsonl"))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(render_report(summary, run_dir.name))
    return summary


def export_labeling(run_dir: str | Path, out: str | Path, *, n: int = 50, seed: int = 0) -> int:
    """A blind sheet of at most n explanations for human coders: about two thirds explanations
    of Jev's wrong decisions (policy misapplications first, then a random sample of the rest),
    the remainder explanations of correct decisions, for false alarms. It shows what the judge saw,
    never the judge's answers."""
    from sysone.benchmark.adapter import adapter_for
    from sysone.benchmark.store import load_benchmark
    from sysone.explain.judge import evaluator_facts

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pilot1_dir = Path(manifest["pilot1"]["dir"])
    cases, meta = load_benchmark(manifest["benchmark"]["path"])
    adapter, by_case = adapter_for(meta), {c.case_id: c for c in cases}
    texts = {r["explanation_id"]: r["explanation_text"] for r in _read_jsonl(pilot1_dir / "explanations.jsonl")
             if r["status"] == "ok"}
    judged = sorted(load_judgments(run_dir), key=lambda r: r["explanation_id"])
    rng = random.Random(seed)
    misapplied = [r for r in judged if r["error_kind"] == "misapplies policy"]
    followed = [r for r in judged if r["error_kind"] == "follows presented record"]
    correct = [r for r in judged if r["error_kind"] == "jev correct"]
    for group in (misapplied, followed, correct):
        rng.shuffle(group)
    wrong = (misapplied + followed)[:2 * n // 3]
    sample = wrong + correct[:max(0, n - len(wrong))]
    rng.shuffle(sample)
    with Path(out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "explainer", "condition", "decision", "case_file", "evaluator_facts",
                    "explanation", *HUMAN_FIELDS, "notes"])
        for r in sample:
            case = by_case[r["case_id"]]
            decision = r.get("explained_decision") or r["jev_decision"]
            w.writerow([r["case_id"], r["explainer"], r["condition"], decision, adapter.render(case.original),
                        evaluator_facts(case, decision, r["policy_outcome"]), texts[r["explanation_id"]],
                        *[""] * len(HUMAN_FIELDS), ""])
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--export-labeling", metavar="CSV")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    if args.export_labeling:
        k = export_labeling(args.run_dir, args.export_labeling, n=args.n)
        print(f"wrote {k} explanations to {args.export_labeling}")
        return
    analyze_run(args.run_dir, labels=[Path(p) for p in args.labels])
    print((Path(args.run_dir) / "report.md").read_text())


if __name__ == "__main__":
    main()
