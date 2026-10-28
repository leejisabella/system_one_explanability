"""Pilot 0 metrics (P0.1–P0.8) computed from a run's decisions.jsonl and ledger.jsonl.

Decisions are compared as labels (majority across repeats) and, where Jev returned
probabilities, as P(approve): its spread across repeats (the noise floor) and its
shift under each counterfactual edit.

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
# Go / no-go for Experiment 1: enough cases where Jev's wrong decision harms the customer
# (a denial of a refund the policy owes on the true record). Wrong approvals harm the
# store, not the person contesting, so they are reported but don't count toward this.
USER_HARMFUL_MIN_GO = 20
# Share of variants whose repeats all give the same decision (set 2026-10-28, after
# refund_v0 and refund_v1 both came in at 97%).
STABILITY_UNANIMOUS_GO = 0.90
# Rules decided by counting days against the 30- or 90-day window (net of carrier
# delay). Jev's own docs list date arithmetic as unreliable, so these are reported apart.
DATE_WINDOW_RULES = frozenset({"R5_within_window", "R5_outside_window"})


def rule_group(rule: str | None) -> str:
    return "date window" if rule in DATE_WINDOW_RULES else "other rules"


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


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _quantile(xs: list[float], q: float) -> float | None:
    """Nearest-rank quantile, so the result is always one of the observed values."""
    if not xs:
        return None
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def summarize(rows: list[dict[str, Any]], ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    parsed = [r for r in rows if r["parse_ok"]]

    # P0.2: does Jev apply the policy to the record it is shown (per sample)? Jev can
    # answer "undetermined", so variants the policy can't decide are scored too.
    policy_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    policy_by_rule: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    policy_by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in parsed:
        rule = r.get("variant_rule") or "unknown"
        hit = r["decision"] == r["policy_outcome"]
        for cell in (policy_by_type[r["case_type"]], policy_by_rule[rule], policy_by_group[rule_group(rule)]):
            cell[0] += hit
            cell[1] += 1

    variants: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[str]] = defaultdict(list)
    p_approve: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        variants.setdefault(r["variant_id"], r)
    for r in parsed:
        labels[r["variant_id"]].append(r["decision"])
        if r.get("p_approve") is not None:
            p_approve[r["variant_id"]].append(r["p_approve"])
    maj = {vid: majority(labels[vid]) for vid in variants}
    mean_p = {vid: sum(ps) / len(ps) for vid, ps in p_approve.items()}
    originals = {v["case_id"]: vid for vid, v in variants.items() if v["variant_kind"] == "original"}

    # P0.3: is Jev's decision on the original case wrong under ground truth? Ambiguous
    # cases count as right when Jev answers "undetermined", but only cases with a
    # determined true outcome can be contestable.
    truth_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    truth_by_rule: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    contestable, ties = [], []
    # Why each contestable case is wrong. "Follows presented record": Jev applied the
    # policy correctly to a corrupted or incomplete record (a defect the benchmark
    # planted). Otherwise Jev misapplied the policy to what it was shown, split by
    # whether a date-window rule decides the presented record.
    causes: Counter[str] = Counter()
    # Who each wrong decision harms: the customer (denied an owed refund), the store
    # (approved a refund the policy doesn't allow), or neither ("undetermined" given).
    harms: dict[str, list[str]] = defaultdict(list)
    for case_id, vid in sorted(originals.items()):
        v = variants[vid]
        if maj[vid] is None:
            ties.append(case_id)
            continue
        right = maj[vid] == v["true_outcome"]
        for cell in (truth_by_type[v["case_type"]], truth_by_rule[v.get("decisive_rule") or "unknown"]):
            cell[0] += right
            cell[1] += 1
        if not right and v["true_outcome"] != UNDETERMINED:
            contestable.append(case_id)
            harm = ("customer: wrongly denied" if maj[vid] == "deny" and v["true_outcome"] == "approve"
                    else "store: wrongly approved" if maj[vid] == "approve" else "neither: answered undetermined")
            harms[harm].append(case_id)
            if maj[vid] == v["policy_outcome"]:
                causes["follows presented record"] += 1
            else:
                causes[f"misapplies policy: {rule_group(v.get('variant_rule'))}"] += 1

    # P0.4: agreement across repeated samples of the same input.
    shares, unanimous = [], 0
    multi = [vid for vid in variants if len(labels[vid]) >= 2]
    for vid in multi:
        top = Counter(labels[vid]).most_common(1)[0][1]
        shares.append(top / len(labels[vid]))
        unanimous += top == len(labels[vid])

    # P0.4 noise: how far P(approve) moves across repeats of the identical input.
    # The high quantile of the range is a candidate noise floor for later experiments:
    # shifts smaller than it can't be told apart from repeat-to-repeat noise.
    ranges = [round(max(ps) - min(ps), 4) for ps in p_approve.values() if len(ps) >= 2]

    # P0.4 option order: does Jev favor whichever option is listed first?
    ordered = [r for r in parsed if r.get("option_order")]
    orders_by_variant: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    p_first: dict[str, list[float]] = defaultdict(list)
    p_later: dict[str, list[float]] = defaultdict(list)
    for r in ordered:
        orders_by_variant[r["variant_id"]].add(tuple(r["option_order"]))
        if r.get("p_approve") is not None:
            (p_first if r["option_order"][0] == "approve" else p_later)[r["variant_id"]].append(r["p_approve"])
    first_shift = [sum(p_first[v]) / len(p_first[v]) - sum(p_later[v]) / len(p_later[v]) for v in p_first if v in p_later]
    orders_vary = any(len(orders) > 1 for orders in orders_by_variant.values())

    # P0.5 / P0.6: does the decision flip when the policy says it should, and only then?
    flips: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    shifts: dict[tuple[str, str], list[float]] = defaultdict(list)
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
        if vid in mean_p and orig in mean_p:
            shifts[(v["variant_kind"], expectation)].append(abs(mean_p[vid] - mean_p[orig]))

    # P0.7: which Jev versions answered. A locked run should show exactly one.
    versions = Counter(r["model_used"] for r in parsed)

    # P0.8: cost. The ledger includes failed attempts; rows include only recorded decisions.
    billed = [r for r in rows if not r.get("cached")]
    row_cost = sum(r.get("cost_usd") or 0.0 for r in billed)
    total_cost = sum(e.get("cost_usd", 0.0) for e in ledger) if ledger is not None else row_cost

    unanimous_share = round(unanimous / len(multi), 4) if multi else None
    n_cases = len(originals)
    parse_rate = len(parsed) / len(rows) if rows else None
    user_harmful = harms.get("customer: wrongly denied", [])
    return {
        "n_rows": len(rows),
        "n_cases": n_cases,
        "n_variants": len(variants),
        "P0.1_parse": _rate(len(parsed), len(rows)),
        "P0.2_policy_agreement_by_type": {t: _rate(*c) for t, c in sorted(policy_by_type.items())},
        "P0.2_policy_agreement_by_rule": {t: _rate(*c) for t, c in sorted(policy_by_rule.items())},
        "P0.2_policy_agreement_by_rule_group": {t: _rate(*c) for t, c in sorted(policy_by_group.items())},
        "P0.3_truth_agreement_by_type": {t: _rate(*c) for t, c in sorted(truth_by_type.items())},
        "P0.3_truth_agreement_by_rule": {t: _rate(*c) for t, c in sorted(truth_by_rule.items())},
        "P0.3_contestable_cases": contestable,
        "P0.3_contestable_by_cause": dict(sorted(causes.items())),
        "P0.3_contestable_by_harm": {h: sorted(ids) for h, ids in sorted(harms.items())},
        "P0.3_user_harmful_cases": sorted(user_harmful),
        "P0.3_tied_cases": ties,
        "P0.4_stability": {
            "n_variants": len(multi),
            "mean_majority_share": round(sum(shares) / len(shares), 4) if shares else None,
            "unanimous_share": unanimous_share,
        },
        "P0.4_probability_noise": {
            "n_variants": len(ranges),
            "identical_across_repeats_share": round(sum(x == 0 for x in ranges) / len(ranges), 4) if ranges else None,
            "mean_range_p_approve": _mean(ranges),
            "p95_range_p_approve": _quantile(ranges, 0.95),
            "max_range_p_approve": max(ranges) if ranges else None,
            # When repeats list the options in different orders, the range includes any
            # order effect, so it is an upper bound on pure repeat noise.
            "includes_order_effects": orders_vary,
        },
        "P0.4_option_order": {
            "n_rows": len(ordered),
            "orders_vary_across_repeats": orders_vary,
            "first_listed_chosen_share": (
                round(sum(r["decision"] == r["option_order"][0] for r in ordered) / len(ordered), 4) if ordered else None
            ),
            # With balanced rotation and no order effect, each option is chosen while
            # listed first exactly as often as while listed elsewhere: 1/k of the time.
            "no_effect_share": round(1 / len(ordered[0]["option_order"]), 4) if ordered else None,
            "n_variants_with_both": len(first_shift),
            "mean_p_approve_shift_when_approve_first": _mean(first_shift),
        },
        "P0.5_6_flips": [
            {
                "variant_kind": kind,
                "expectation": exp,
                **_rate(*c),
                "mean_abs_shift_p_approve": _mean(shifts.get((kind, exp), [])),
            }
            for (kind, exp), c in sorted(flips.items())
        ],
        "P0.7_model_versions": {
            "model_used_counts": dict(versions.most_common()),
            "n_versions": len(versions),
        },
        "P0.8_cost": {
            "total_usd": round(total_cost, 6),
            "billed_decisions": len(billed),
            "per_decision_usd": round(row_cost / len(billed), 6) if billed else None,
        },
        "go_no_go": {
            "parse_rate_ok": parse_rate is not None and parse_rate >= PARSE_RATE_GO,
            "enough_contestable_cases": len(user_harmful) >= USER_HARMFUL_MIN_GO,
            "stability_ok": unanimous_share is not None and unanimous_share >= STABILITY_UNANIMOUS_GO,
            "single_model_version": len(versions) == 1,
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
    out += ["", "By the rule that decides the presented record. Date-window rules need date arithmetic, "
            "which TypeSafe lists as unreliable for Jev.", "",
            "| Rule | Agreement |", "|---|---|"]
    out += [f"| {t} | {pct(c)} |" for t, c in s["P0.2_policy_agreement_by_rule"].items()]
    out += [f"| **{t}** | {pct(c)} |" for t, c in s["P0.2_policy_agreement_by_rule_group"].items()]
    out += ["", "**P0.3 agreement with ground truth (majority decision on the original case)**", "",
            "| Case type | Agreement |", "|---|---|"]
    out += [f"| {t} | {pct(c)} |" for t, c in s["P0.3_truth_agreement_by_type"].items()]
    out += ["", "| Rule deciding the true record | Agreement |", "|---|---|"]
    out += [f"| {t} | {pct(c)} |" for t, c in s["P0.3_truth_agreement_by_rule"].items()]
    out += ["", f"Contestable cases (Jev wrong under ground truth): {len(s['P0.3_contestable_cases'])}"
            f" of {s['n_cases']}; tied: {len(s['P0.3_tied_cases'])}", ""]
    if s["P0.3_contestable_by_cause"]:
        out += ["| Why Jev is wrong | Cases |", "|---|---|"]
        out += [f"| {cause} | {n} |" for cause, n in s["P0.3_contestable_by_cause"].items()]
        out += ["", "| Who the wrong decision harms | Cases |", "|---|---|"]
        out += [f"| {h} | {len(ids)} |" for h, ids in s["P0.3_contestable_by_harm"].items()]
        out += ["", "\"Follows presented record\" means Jev applied the policy correctly to a corrupted or "
                "incomplete record, a defect planted by the benchmark. The other rows are Jev's own errors.", ""]
    st = s["P0.4_stability"]
    out += [f"**P0.4 stability:** mean majority share {st['mean_majority_share']}, "
            f"unanimous {st['unanimous_share']} over {st['n_variants']} variants", ""]
    nz = s["P0.4_probability_noise"]
    out += ["**P0.4 probability noise:** range of P(approve) across repeats of the same input", "",
            "| Measure | Value |", "|---|---|",
            f"| Variants with 2+ parsed repeats | {nz['n_variants']} |",
            f"| Share with identical P(approve) on every repeat | {nz['identical_across_repeats_share']} |",
            f"| Mean range | {nz['mean_range_p_approve']} |",
            f"| 95th-percentile range (candidate noise floor) | {nz['p95_range_p_approve']} |",
            f"| Largest range | {nz['max_range_p_approve']} |", ""]
    if nz["includes_order_effects"]:
        out += ["Repeats listed the options in different orders, so these ranges include any order effect "
                "and are an upper bound on pure repeat noise. For a pure noise floor, run with "
                "`rotate_option_order: false`.", ""]
    oo = s["P0.4_option_order"]
    if oo["n_rows"]:
        out += [f"**P0.4 option order:** the first-listed option was chosen in {oo['first_listed_chosen_share']} "
                f"of {oo['n_rows']} answers ({oo['no_effect_share']} expected with no order effect, when every "
                f"variant sees each order equally often). P(approve) was {oo['mean_p_approve_shift_when_approve_first']} "
                f"higher on average when approve was listed first ({oo['n_variants_with_both']} variants).", ""]
    out += ["**P0.5 / P0.6 decision flips vs. the original**", "",
            "| Variant kind | Policy expectation | Flip rate | Mean shift in P(approve) |", "|---|---|---|---|"]
    out += [f"| {f['variant_kind']} | {f['expectation']} | {pct(f)} | {f['mean_abs_shift_p_approve']} |"
            for f in s["P0.5_6_flips"]]
    mv = s["P0.7_model_versions"]
    out += ["", f"**P0.7 model versions:** {mv['model_used_counts']}", ""]
    c = s["P0.8_cost"]
    out += [f"**P0.8 cost:** ${c['total_usd']:.4f} total; {c['billed_decisions']} billed decisions; "
            f"per decision {c['per_decision_usd']}", ""]
    g = s["go_no_go"]
    out += ["**Go / no-go**", "",
            f"- Parse rate ≥ {PARSE_RATE_GO:.0%}: {'yes' if g['parse_rate_ok'] else 'no'}",
            f"- At least {USER_HARMFUL_MIN_GO} cases where Jev wrongly denies the customer: "
            f"{'yes' if g['enough_contestable_cases'] else 'no'} ({len(s['P0.3_user_harmful_cases'])})",
            f"- At least {STABILITY_UNANIMOUS_GO:.0%} of variants unanimous across repeats: "
            f"{'yes' if g['stability_ok'] else 'no'} ({s['P0.4_stability']['unanimous_share']})",
            f"- One model version throughout: {'yes' if g['single_model_version'] else 'no'}", ""]
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
