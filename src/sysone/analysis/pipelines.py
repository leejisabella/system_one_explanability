"""Cost, latency and quality of each way to decide a case, from a Pilot 1 run (and optionally its judge run).

The question: is it worth deciding with Jev and explaining afterwards when a decision is
contested, or cheaper and better to let the explainer model decide (and explain) itself?
Every pipeline is costed per case, as one decision call plus any explanation call, from
the calls the runs actually made:

- "Jev decides": one Jev decision.
- "Jev + <explainer> explains (<condition>)": Jev's decision, then the explainer's
  explanation of it, for every case.
- "Jev + <explainer> explains adverse decisions (<condition>)": the explanation only when
  Jev's decision is one the person would contest (refund: deny), the realistic pattern
  for contesting. Not listed for domains with no adverse option.
- "<explainer> decides": one explainer decision.
- "<explainer> decides and explains": its decision, then its explanation of its own
  decision (Pilot 1's own_decision condition; listed when the run has it).

Cost and latency are means over the run's calls for that case (Pilot 0 asks Jev 3 times per
variant; one call is one decision). A pipeline's latency adds its calls, which run one
after the other. Quality, per single decision call (not the majority of 3):
- accuracy against ground truth, and the share of decisions that harm the person in the case;
- with a judge run: of the wrong decisions, the share whose explanation gives grounds to
  contest them (flags the conflict, or names the verified defect and questions it or cites
  the correcting source). A pipeline without an explanation gives none.

Runs from before latency was recorded report it as n/a. The simulator and judge are
evaluation instruments, not part of any pipeline, so their calls aren't counted here.

Usage:  uv run python -m sysone.analysis.pipelines results/runs/<pilot1 run> [--judge results/runs/<judge run>]

Writes pipelines.json and pipelines.md into the Pilot 1 run directory, or with --judge into
the judge run's directory (derived, regenerable).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sysone.analysis.judge import contest_hit, load_judgments
from sysone.analysis.pilot0 import _quantile, _read_jsonl, load_decisions
from sysone.analysis.pilot1 import _latest_ok, explainer_majorities, jev_decisions
from sysone.benchmark.adapter import adapter_for
from sysone.benchmark.store import load_benchmark
from sysone.explain.explainer import OWN_DECISION

JEV = "jev"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _calls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean cost and latency of a set of calls for one case; cached calls aren't provider calls."""
    billed = [r for r in rows if not r.get("cached")]
    lat = [r["latency_s"] for r in billed if r.get("latency_s") is not None]
    return {"cost": _mean([r.get("cost_usd") or 0.0 for r in billed]), "latency": _mean(lat)}


def _summary(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [c["cost"] for c in per_case if c["cost"] is not None]
    lats = [c["latency"] for c in per_case if c["latency"] is not None]
    acc = [c["accuracy"] for c in per_case if c["accuracy"] is not None]
    harm = [c["harmful"] for c in per_case if c["harmful"] is not None]
    wrong = [c for c in per_case if c["wrong_determined"]]
    # caught: True / False for a judged or absent explanation, None for an explanation not judged.
    unjudged = sum(c["caught"] is None for c in wrong)
    return {
        "n_cases": len(per_case),
        "cost_per_case_usd": round(_mean(costs), 8) if costs else None,
        "cost_per_1000_cases_usd": round(1000 * _mean(costs), 4) if costs else None,
        "latency_mean_s": round(_mean(lats), 3) if lats else None,
        "latency_p50_s": _quantile(lats, 0.5),
        "latency_p95_s": _quantile(lats, 0.95),
        "latency_cases": len(lats),
        "decision_accuracy": round(_mean(acc), 4) if acc else None,
        "harmful_decision_rate": round(_mean(harm), 4) if harm else None,
        "wrong_decisions": len(wrong),
        "wrong_decisions_with_grounds": sum(c["caught"] is True for c in wrong),
        # The share is reported only when every explained wrong decision was judged.
        "wrong_decisions_with_grounds_share": (round(sum(c["caught"] is True for c in wrong) / len(wrong), 4)
                                               if wrong and not unjudged else None),
        "wrong_decisions_unjudged": unjudged,
        "explained_share": round(_mean([c["explained"] for c in per_case]), 4) if per_case else None,
    }


def _add(*parts: float | None) -> float | None:
    return None if any(p is None for p in parts) else sum(parts)


def pipelines(pilot1_dir: Path, judge_dir: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((pilot1_dir / "manifest.json").read_text())
    cases, meta = load_benchmark(manifest["benchmark"]["path"])
    adapter = adapter_for(meta)
    skipped = set(manifest.get("skipped_cases_jev_tie") or [])
    cases = [c for c in cases if c.case_id not in skipped]
    explainers = list(manifest["config"]["explainers"])
    conditions = [c for c in manifest["config"]["conditions"] if c != "decision_probabilities"]

    # Calls on each case's original variant: the decision a deployed pipeline would make.
    jev_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in load_decisions(Path(manifest["pilot0"]["dir"])):
        jev_rows[r["variant_id"]].append(r)
    decisions = _latest_ok(_read_jsonl(pilot1_dir / "explainer_decisions.jsonl"),
                           "explainer", "variant_id", "sample_index")
    own_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in decisions:
        own_rows[(r["explainer"], r["variant_id"])].append(r)
    own_majority = explainer_majorities(decisions)
    explanations = {r["explanation_id"]: r
                    for r in _latest_ok(_read_jsonl(pilot1_dir / "explanations.jsonl"), "explanation_id")}
    judged = {}
    if judge_dir:
        judge_manifest = json.loads((judge_dir / "manifest.json").read_text())
        if judge_manifest.get("pilot1", {}).get("run_id") != pilot1_dir.name:
            raise ValueError(f"judge run {judge_dir.name} scored {judge_manifest.get('pilot1')}, not {pilot1_dir.name}")
        judged = {r["explanation_id"]: r for r in load_judgments(judge_dir)}
    jev_majority = {vid: d.label for vid, d in jev_decisions(Path(manifest["pilot0"]["dir"])).items()}

    def harmful_of(case: Any) -> str | None:
        return str(case.harmful_decision) if case.harmful_decision else None

    def decision_quality(rows: list[dict[str, Any]], case: Any) -> tuple[float | None, float | None]:
        labels = [r["decision"] for r in rows if r.get("parse_ok")]
        if not labels:
            return None, None
        truth, harmful = str(case.true_outcome), harmful_of(case)
        return (sum(lab == truth for lab in labels) / len(labels),
                sum(lab == harmful for lab in labels) / len(labels) if harmful else 0.0)

    def grounds(explanation_id: str | None) -> bool | None:
        """Whether an explanation gives grounds to contest a wrong decision; None if not judged."""
        if explanation_id is None:
            return False
        j = judged.get(explanation_id)
        if j is None or not j["parse_ok"]:
            return None
        return j["justification"] == "flags_conflict" or (bool(j["defect_fields"]) and contest_hit(j))

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        vid = case.original.variant_id
        truth_determined = not adapter.undetermined(case.decisive_rule)
        jev_calls = jev_rows.get(vid, [])
        jev = _calls(jev_calls)
        jev_acc, jev_harm = decision_quality(jev_calls, case)
        jev_label = jev_majority.get(vid)
        jev_wrong = truth_determined and jev_label is not None and jev_label != str(case.true_outcome)

        def row(cost, latency, acc, harm, wrong, caught, explained):
            return {"case_id": case.case_id, "cost": cost, "latency": latency, "accuracy": acc, "harmful": harm,
                    "wrong_determined": wrong, "caught": caught if wrong else None, "explained": explained}

        out["Jev decides"].append(row(jev["cost"], jev["latency"], jev_acc, jev_harm, jev_wrong, False, 0))
        for alias in explainers:
            for condition in conditions:
                if condition == OWN_DECISION:
                    continue
                eid = f"{alias}/{condition}/{case.case_id}"
                e = explanations.get(eid)
                if e is None:
                    continue
                ex = _calls([e])
                out[f"Jev + {alias} explains ({condition})"].append(row(
                    _add(jev["cost"], ex["cost"]), _add(jev["latency"], ex["latency"]), jev_acc, jev_harm,
                    jev_wrong, grounds(eid), 1))
                if adapter.adverse_labels:
                    adverse = jev_label in adapter.adverse_labels
                    out[f"Jev + {alias} explains adverse decisions ({condition})"].append(row(
                        _add(jev["cost"], ex["cost"]) if adverse else jev["cost"],
                        _add(jev["latency"], ex["latency"]) if adverse else jev["latency"],
                        jev_acc, jev_harm, jev_wrong, grounds(eid) if adverse else False, int(adverse)))
            calls = own_rows.get((alias, vid), [])
            if not calls:
                continue
            own = _calls(calls)
            own_acc, own_harm = decision_quality(calls, case)
            mine = own_majority.get((alias, vid))
            own_wrong = truth_determined and mine is not None and mine != str(case.true_outcome)
            out[f"{alias} decides"].append(row(own["cost"], own["latency"], own_acc, own_harm, own_wrong, False, 0))
            eid = f"{alias}/{OWN_DECISION}/{case.case_id}"
            if eid in explanations:
                ex = _calls([explanations[eid]])
                out[f"{alias} decides and explains"].append(row(
                    _add(own["cost"], ex["cost"]), _add(own["latency"], ex["latency"]), own_acc, own_harm,
                    own_wrong, grounds(eid), 1))
    return {
        "pilot1_run": pilot1_dir.name,
        "judge_run": judge_dir.name if judge_dir else None,
        "domain": adapter.domain,
        "adverse_labels": list(adapter.adverse_labels),
        "pipelines": {name: _summary(rows) for name, rows in out.items()},
    }


def render_report(s: dict[str, Any]) -> str:
    def money(x: float | None) -> str:
        return "n/a" if x is None else f"${x:.4f}"

    def sec(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}s"

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    out = [f"# Pipelines: cost, latency and quality ({s['pilot1_run']})", "",
           f"Domain: {s['domain']}. Judge run: {s['judge_run'] or 'none (grounds to contest not scored)'}.", "",
           "| Pipeline | Cases | Cost per 1,000 cases | Latency mean / p50 / p95 | Decision accuracy | "
           "Harmful decisions | Wrong decisions | With grounds to contest | Explained |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, p in s["pipelines"].items():
        lat = (f"{sec(p['latency_mean_s'])} / {sec(p['latency_p50_s'])} / {sec(p['latency_p95_s'])}"
               if p["latency_cases"] else "not recorded")
        if p["explained_share"] == 0:
            grounds = "none (no explanation)"
        elif p["wrong_decisions_unjudged"]:
            grounds = "not judged"
        else:
            grounds = (f"{pct(p['wrong_decisions_with_grounds_share'])} "
                       f"({p['wrong_decisions_with_grounds']} of {p['wrong_decisions']})")
        out.append(f"| {name} | {p['n_cases']} | {money(p['cost_per_1000_cases_usd'])} | {lat} | "
                   f"{pct(p['decision_accuracy'])} | {pct(p['harmful_decision_rate'])} | {p['wrong_decisions']} | "
                   f"{grounds} | {pct(p['explained_share'])} |")
    out += ["", "Cost and latency per case add a pipeline's calls (one decision, plus an explanation where it has "
            "one); latency assumes they run one after the other. Accuracy and harmful decisions are per single "
            "decision call. \"With grounds to contest\": of the wrong decisions, the share whose explanation flags "
            "the conflict or names the verified defect and questions it or cites the correcting source (judge codes, "
            "valid once checked against human labels). \"Explains adverse decisions\" explains only decisions "
            f"the person would contest ({', '.join(s['adverse_labels']) or 'none in this domain'}).", ""]
    return "\n".join(out)


def analyze_run(pilot1_dir: str | Path, judge_dir: str | Path | None = None) -> dict[str, Any]:
    """Writes into the judge run's folder when one is given, so a judge never edits the Pilot 1 run."""
    pilot1_dir = Path(pilot1_dir)
    s = pipelines(pilot1_dir, Path(judge_dir) if judge_dir else None)
    out = Path(judge_dir) if judge_dir else pilot1_dir
    (out / "pipelines.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    (out / "pipelines.md").write_text(render_report(s))
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pilot1_dir")
    parser.add_argument("--judge", help="a judge run over this Pilot 1 run, for grounds to contest")
    args = parser.parse_args()
    analyze_run(args.pilot1_dir, args.judge)
    print((Path(args.judge or args.pilot1_dir) / "pipelines.md").read_text())


if __name__ == "__main__":
    main()
