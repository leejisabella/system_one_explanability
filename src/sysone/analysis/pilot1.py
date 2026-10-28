"""Pilot 1 metrics computed from a run's JSONL files, the Pilot 0 run it builds on, and the benchmark.

Faithfulness scores, kept separate and never blended (research_docs/12, "How an
explanation's implied decision is obtained"):
- A-direct: the simulator's prediction from the explanation vs. Jev's actual decision on
  each hidden variant, against the same simulator without the explanation. The gain is
  reported as Normalized Simulatability Gain: (acc_with - acc_without) / (1 - acc_without).
- A-executed: the simulator's basis run against the variant's record. Its agreement
  with A-direct checks whether the simulator followed the explanation.
- Covered rate: the share of hidden variants the simulator says the explanation decides.
- Option B (structured condition only): the explanation's own basis run against each
  variant's record. Deterministic, with no simulator.
Plus Jev-explainer disagreement: the two models' independent majority decisions compared.

Jev's decision on a variant is its Pilot 0 majority; variants where it tied are left out.
Contestability scoring against the verified defects needs the LLM judge and is not here yet.

Saved simulator replies that failed to parse are re-parsed with the current parser, which
repairs decimals with a spelled-out digit ("0. nine"); repaired replies are counted in the
report. --no-repair scores them as unparsed instead.

Usage:  uv run python -m sysone.analysis.pilot1 [--no-repair] results/runs/<run_id>

Writes summary.json and report.md into the run directory. Both are derived files and
can be regenerated at any time without new API calls.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sysone.analysis.pilot0 import _rate, _read_jsonl, load_decisions, majority
from sysone.benchmark.records import Record
from sysone.benchmark.store import load_benchmark
from sysone.decision.jev import LABELS
from sysone.explain.basis import BasisError, execute, parse_basis
from sysone.explain.simulator import parse_simulation
from sysone.types import ChatResponse, Usage

CONTROL = "control"
STRUCTURED = "structured"


@dataclass(frozen=True)
class JevVariantDecision:
    # Majority across Pilot 0 samples; None on a tie or with no parsed sample.
    label: str | None
    # Mean of the parsed samples' probabilities.
    probabilities: dict[str, float] | None
    n_samples: int


def jev_decisions(pilot0_dir: Path) -> dict[str, JevVariantDecision]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_decisions(pilot0_dir):
        if row["parse_ok"]:
            by_variant[row["variant_id"]].append(row)
    out = {}
    for vid, rows in by_variant.items():
        with_probs = [r["probabilities"] for r in rows if r.get("probabilities")]
        probs = (
            {k: round(sum(p[k] for p in with_probs) / len(with_probs), 4) for k in LABELS} if with_probs else None
        )
        out[vid] = JevVariantDecision(majority([r["decision"] for r in rows]), probs, len(rows))
    return out


def _latest_ok(rows: list[dict[str, Any]], *key: str) -> list[dict[str, Any]]:
    latest: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if r["status"] == "ok":
            latest[tuple(r[k] for k in key)] = r
    return list(latest.values())


def _share(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _executed(basis_raw: Any, record: Record) -> str | None:
    try:
        return execute(parse_basis(basis_raw), record).decision
    except BasisError:
        return None


def reparse_simulations(rows: list[dict[str, Any]], *, repair: bool = True) -> list[dict[str, Any]]:
    """Saved simulations, with replies that parse only after repair filled in (repair=True)
    or scored as unparsed (repair=False)."""
    out = []
    for r in rows:
        if r.get("status") == "ok" and not repair and r.get("repaired"):
            r = {**r, "parse_ok": False, "parse_error": "repaired reply excluded (--no-repair)"}
        elif r.get("status") == "ok" and repair and not r["parse_ok"] and r.get("raw_text"):
            saved = ChatResponse(text=r["raw_text"], usage=Usage(0, 0), cost_usd=0.0, provider="",
                                 model_requested="", model_used="", finish_reason=None, raw={})
            sim = parse_simulation(saved, r.get("request_key", ""))
            if sim.repaired:
                r = {**r, "predicted_decision": sim.predicted, "probabilities": sim.probabilities,
                     "basis": sim.basis_raw, "basis_ok": sim.basis is not None, "basis_error": sim.basis_error,
                     "covered": sim.covered, "parse_ok": sim.parse_ok, "parse_error": sim.parse_error,
                     "repaired": True}
        out.append(r)
    return out


def _on_departures(rows: list[dict[str, Any]], control: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A-direct and control on the hidden variants where Jev's decision differs from the policy's,
    the only variants where knowing the policy isn't enough to predict Jev."""
    paired = [r for r in rows if r["parse_ok"] and r["jev_decision"] is not None
              and r.get("policy_outcome") not in (None, r["jev_decision"]) and control.get(r["variant_id"], {}).get("parse_ok")]
    return {
        "a_direct": _rate(sum(r["predicted_decision"] == r["jev_decision"] for r in paired), len(paired)),
        "control": _rate(sum(control[r["variant_id"]]["predicted_decision"] == r["jev_decision"] for r in paired),
                         len(paired)),
    }


def _faithfulness(rows: list[dict[str, Any]], control: dict[str, dict[str, Any]],
                  records: dict[str, Record]) -> dict[str, Any]:
    """Scores for one (explainer, condition) group of simulations with an explanation."""
    parsed = [r for r in rows if r["parse_ok"]]
    # Scored on variants where Jev's decision is determined and both simulations parsed.
    paired = [r for r in parsed if r["jev_decision"] is not None and control.get(r["variant_id"], {}).get("parse_ok")]
    hit_with = sum(r["predicted_decision"] == r["jev_decision"] for r in paired)
    hit_without = sum(control[r["variant_id"]]["predicted_decision"] == r["jev_decision"] for r in paired)
    acc_with, acc_without = _share(hit_with, len(paired)), _share(hit_without, len(paired))
    # From the counts, not the rounded rates.
    n = len(paired)
    nsg = round((hit_with - hit_without) / (n - hit_without), 4) if n and hit_without < n else None

    def p_jev(r: dict[str, Any]) -> float:
        return r["probabilities"][r["jev_decision"]]

    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in paired:
        cell = by_kind[r["variant_kind"]]
        cell[0] += r["predicted_decision"] == r["jev_decision"]
        cell[1] += control[r["variant_id"]]["predicted_decision"] == r["jev_decision"]
        cell[2] += 1

    covered = [r for r in parsed if r["covered"]]
    covered_scored = [r for r in covered if r["jev_decision"] is not None]
    executed = {id(r): _executed(r["basis"], records[r["variant_id"]]) for r in parsed if r["basis_ok"]}
    decidable = [r for r in parsed if executed.get(id(r)) is not None]
    decidable_scored = [r for r in decidable if r["jev_decision"] is not None]
    return {
        "n_simulations": len(rows),
        "simulator_parse": _rate(len(parsed), len(rows)),
        "n_scored": len(paired),
        "a_direct_agreement_with_jev": acc_with,
        "control_agreement_with_jev": acc_without,
        "normalized_simulatability_gain": nsg,
        "mean_p_jev_decision_with": round(sum(map(p_jev, paired)) / len(paired), 4) if paired else None,
        "mean_p_jev_decision_control": (
            round(sum(p_jev(control[r["variant_id"]]) for r in paired) / len(paired), 4) if paired else None
        ),
        "by_variant_kind": {
            kind: {"n": n, "a_direct": _share(w, n), "control": _share(c, n)}
            for kind, (w, c, n) in sorted(by_kind.items())
        },
        "covered": _rate(len(covered), len(parsed)),
        "a_direct_agreement_when_covered": _share(
            sum(r["predicted_decision"] == r["jev_decision"] for r in covered_scored), len(covered_scored)),
        "basis_decidable": _rate(len(decidable), len(parsed)),
        "a_direct_vs_a_executed_agreement": _share(
            sum(r["predicted_decision"] == executed[id(r)] for r in decidable), len(decidable)),
        "a_executed_agreement_with_jev": _share(
            sum(executed[id(r)] == r["jev_decision"] for r in decidable_scored), len(decidable_scored)),
    }


def _option_b(explanations: list[dict[str, Any]], hidden: dict[str, list[str]], records: dict[str, Record],
              jev: dict[str, str | None]) -> dict[str, Any]:
    """Structured explanations run directly against each hidden variant's record."""
    by_explainer: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # agree, decidable, scored variants
    for e in explanations:
        if e["condition"] != STRUCTURED or not e["parse_ok"]:
            continue
        for vid in hidden.get(e["case_id"], []):
            if jev.get(vid) is None:
                continue
            decision = _executed(e["basis"], records[vid])
            cell = by_explainer[e["explainer"]]
            cell[2] += 1
            if decision is not None:
                cell[1] += 1
                cell[0] += decision == jev[vid]
    return {
        alias: {"n_variants": n, "decidable": _share(d, n), "executed_agreement_with_jev": _share(k, d)}
        for alias, (k, d, n) in sorted(by_explainer.items())
    }


def _disagreement(decisions: list[dict[str, Any]], jev: dict[str, str | None],
                  kinds: dict[str, str], policy: dict[str, str]) -> dict[str, Any]:
    """Jev's and each explainer's independent majority decisions, compared."""
    labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in decisions:
        if r["parse_ok"]:
            labels[(r["explainer"], r["variant_id"])].append(r["decision"])
    explainers = sorted({r["explainer"] for r in decisions})
    out = {}
    for alias in explainers:
        cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        unanimous = multi = ties = policy_hits = policy_n = 0
        for (a, vid), ls in labels.items():
            if a != alias:
                continue
            maj = majority(ls)
            if len(ls) >= 2:
                multi += 1
                unanimous += len(set(ls)) == 1
            if maj is None:
                ties += 1
                continue
            policy_n += 1
            policy_hits += maj == policy[vid]
            if jev.get(vid) is None:
                continue
            group = "originals" if kinds[vid] == "original" else "variants"
            for g in (group, "all"):
                cells[g][0] += maj != jev[vid]
                cells[g][1] += 1
        n_parsed = sum(len(ls) for (a, _), ls in labels.items() if a == alias)
        n_rows = sum(r["explainer"] == alias for r in decisions)
        out[alias] = {
            "parse": _rate(n_parsed, n_rows),
            "disagreement_with_jev": {g: _rate(*c) for g, c in sorted(cells.items())},
            "explainer_unanimous_share": _share(unanimous, multi),
            "explainer_tied_variants": ties,
            "explainer_policy_agreement": _share(policy_hits, policy_n),
        }
    return out


def summarize(explainer_decisions: list[dict[str, Any]], explanations: list[dict[str, Any]],
              simulations: list[dict[str, Any]], *, records: dict[str, Record], hidden: dict[str, list[str]],
              kinds: dict[str, str], policy: dict[str, str], jev: dict[str, str | None],
              ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    control = {r["variant_id"]: r for r in simulations if r["condition"] == CONTROL}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in simulations:
        if r["condition"] != CONTROL:
            groups[(r["explainer"], r["condition"])].append(r)
    control_parsed = [r for r in control.values() if r["parse_ok"]]
    control_scored = [r for r in control_parsed if r["jev_decision"] is not None]

    expl_groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for e in explanations:
        expl_groups[(e["explainer"], e["condition"])][0] += bool(e["parse_ok"])
        expl_groups[(e["explainer"], e["condition"])][1] += 1

    cost_by_model: Counter[str] = Counter()
    cost_by_step: Counter[str] = Counter()
    for e in ledger or []:
        cost_by_model[f"{e.get('provider')}/{e.get('model_requested')}"] += e.get("cost_usd", 0.0)
        cost_by_step[(e.get("tags") or {}).get("step", "other")] += e.get("cost_usd", 0.0)
    versions = Counter(r["model_used"] for r in [*explainer_decisions, *explanations, *simulations]
                       if r.get("model_used"))
    return {
        "n_explainer_decisions": len(explainer_decisions),
        "n_explanations": len(explanations),
        "n_simulations": len(simulations),
        "explanation_parse": {f"{a}/{c}": _rate(*cell) for (a, c), cell in sorted(expl_groups.items())},
        "control": {
            "n_simulations": len(control),
            "simulator_parse": _rate(len(control_parsed), len(control)),
            "agreement_with_jev": _rate(
                sum(r["predicted_decision"] == r["jev_decision"] for r in control_scored), len(control_scored)),
        },
        "faithfulness": {f"{a}/{c}": _faithfulness(rows, control, records) for (a, c), rows in sorted(groups.items())},
        "faithfulness_on_policy_departures": {f"{a}/{c}": _on_departures(rows, control)
                                              for (a, c), rows in sorted(groups.items())},
        "simulations_repaired": sum(bool(r.get("repaired")) and bool(r["parse_ok"]) for r in simulations),
        "option_b_structured_executed": _option_b(explanations, hidden, records, jev),
        "jev_explainer_disagreement": _disagreement(explainer_decisions, jev, kinds, policy),
        "model_versions": dict(versions.most_common()),
        "cost": {
            "total_usd": round(sum(cost_by_model.values()), 6),
            "by_model_usd": {m: round(c, 6) for m, c in sorted(cost_by_model.items())},
            "by_step_usd": {m: round(c, 6) for m, c in sorted(cost_by_step.items())},
        },
    }


def render_report(s: dict[str, Any], run_id: str = "") -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    def cell(c: dict[str, Any]) -> str:
        return "n/a" if c["rate"] is None else f"{c['rate']:.0%} ({c['k']}/{c['n']})"

    out = [f"# Pilot 1 report {run_id}".rstrip(), ""]
    out += [f"Explainer decisions: {s['n_explainer_decisions']} · explanations: {s['n_explanations']} · "
            f"simulations: {s['n_simulations']}", ""]
    out += ["**Explanations parsed**", "", "| Explainer / condition | Parsed |", "|---|---|"]
    out += [f"| {g} | {cell(c)} |" for g, c in s["explanation_parse"].items()]
    ctl = s["control"]
    out += ["", f"**Control (no explanation):** simulator parse {cell(ctl['simulator_parse'])}; "
            f"agreement with Jev {cell(ctl['agreement_with_jev'])}", ""]
    out += ["**Faithfulness on hidden variants** (A-direct is the main score; the others are checks, not blended)", "",
            "| Explainer / condition | Scored | A-direct | Control | NSG | Covered | A-direct vs A-executed "
            "| A-executed vs Jev |",
            "|---|---|---|---|---|---|---|---|"]
    for g, f in s["faithfulness"].items():
        out.append(
            f"| {g} | {f['n_scored']} | {pct(f['a_direct_agreement_with_jev'])} | "
            f"{pct(f['control_agreement_with_jev'])} | {f['normalized_simulatability_gain']} | "
            f"{cell(f['covered'])} | {pct(f['a_direct_vs_a_executed_agreement'])} | "
            f"{pct(f['a_executed_agreement_with_jev'])} |")
    out += ["", "NSG = (A-direct − control) / (1 − control). A low A-direct vs A-executed agreement means the "
            "simulator reasoned on its own instead of from the explanation.", ""]
    out += ["**Where Jev departs from the policy** (hidden variants whose policy outcome differs from Jev's "
            "decision; knowing the policy alone predicts these wrong)", "",
            "| Explainer / condition | A-direct | Control (same variants) |", "|---|---|---|"]
    out += [f"| {g} | {cell(d['a_direct'])} | {cell(d['control'])} |"
            for g, d in s["faithfulness_on_policy_departures"].items()]
    out += ["", f"Simulator replies repaired before parsing (spelled-out digits such as \"0. nine\"): "
            f"{s['simulations_repaired']} of {s['n_simulations']}.", ""]
    if s["option_b_structured_executed"]:
        out += ["**Option B: structured explanations run directly against the records**", "",
                "| Explainer | Variants | Decidable | Agreement with Jev |", "|---|---|---|---|"]
        out += [f"| {a} | {b['n_variants']} | {pct(b['decidable'])} | {pct(b['executed_agreement_with_jev'])} |"
                for a, b in s["option_b_structured_executed"].items()]
        out.append("")
    out += ["**Jev–explainer disagreement** (independent majority decisions)", "",
            "| Explainer | Parsed | Originals | Variants | All | Explainer unanimous | Explainer vs policy |",
            "|---|---|---|---|---|---|---|"]
    for a, d in s["jev_explainer_disagreement"].items():
        dis = d["disagreement_with_jev"]
        out.append(
            f"| {a} | {cell(d['parse'])} | {cell(dis['originals']) if 'originals' in dis else 'n/a'} | "
            f"{cell(dis['variants']) if 'variants' in dis else 'n/a'} | {cell(dis['all']) if 'all' in dis else 'n/a'} | "
            f"{pct(d['explainer_unanimous_share'])} | {pct(d['explainer_policy_agreement'])} |")
    out += ["", f"**Model versions:** {s['model_versions']}", ""]
    c = s["cost"]
    out += [f"**Cost:** ${c['total_usd']:.4f} total; by model {c['by_model_usd']}; by step {c['by_step_usd']}", ""]
    out += ["Contestability scoring against the verified defects needs the LLM judge and is not computed yet.", ""]
    return "\n".join(out)


def analyze_run(run_dir: str | Path, *, repair: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cases, _ = load_benchmark(manifest["benchmark"]["path"])
    records = {v.variant_id: v.record for c in cases for v in c.variants}
    hidden = {c.case_id: [v.variant_id for v in c.variants[1:]] for c in cases}
    kinds = {v.variant_id: str(v.kind) for c in cases for v in c.variants}
    policy = {v.variant_id: str(v.policy_outcome) for c in cases for v in c.variants}
    jev = {vid: d.label for vid, d in jev_decisions(Path(manifest["pilot0"]["dir"])).items()}
    summary = summarize(
        _latest_ok(_read_jsonl(run_dir / "explainer_decisions.jsonl"), "explainer", "variant_id", "sample_index"),
        _latest_ok(_read_jsonl(run_dir / "explanations.jsonl"), "explanation_id"),
        _latest_ok(reparse_simulations(_read_jsonl(run_dir / "simulations.jsonl"), repair=repair),
                   "explanation_id", "variant_id"),
        records=records, hidden=hidden, kinds=kinds, policy=policy, jev=jev,
        ledger=_read_jsonl(run_dir / "ledger.jsonl"),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "report.md").write_text(render_report(summary, run_dir.name))
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-repair"]
    if len(args) != 1:
        sys.exit(__doc__)
    analyze_run(args[0], repair="--no-repair" not in sys.argv[1:])
    print((Path(args[0]) / "report.md").read_text())
