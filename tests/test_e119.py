"""E.119: the 120b-vs-20b comparison is paired on the same items and the decision rule is applied as fixed."""

from sysone.analysis.e119 import compare, decide
from sysone.experiments.e119 import n_calls, pilot_plan


def _plan():
    rows = []
    for i in range(40):
        rows.append({"sample": "s", "item_id": f"i{i}", "label": "a", "jev_label": "a" if i % 4 else "b",
                     "jev_correct": bool(i % 4), "matched_label": "a", "matched_correct": True, "in_subsample": True})
    return rows


def _reviews(plan, *, fix_jev_wrong: bool, alone_right: bool):
    out = []
    for p in plan:
        base = {"status": "ok", "parse_ok": True, "role": "reviewer", "sample": "s", "item_id": p["item_id"], "label": "a"}
        for k in range(5):
            out.append({**base, "condition": "alone", "sample_index": k, "final_label": "a" if alone_right else "b"})
        if p["jev_correct"]:
            out.append({**base, "condition": "correct_label", "sample_index": 0, "final_label": "a"})
        else:
            out.append({**base, "condition": "jev_label", "sample_index": 0, "final_label": "a" if fix_jev_wrong else "b"})
    return out


def test_a_reviewer_that_fixes_jevs_mistakes_passes_the_rule():
    plan = _plan()
    s = compare(plan, _reviews(plan, fix_jev_wrong=False, alone_right=False), _reviews(plan, fix_jev_wrong=True, alone_right=True))
    P = s["pooled"]
    assert P["jev_alone"]["rate"] == 0.75 and P["cascade_20b"]["rate"] == 0.75 and P["cascade_120b"]["rate"] == 1.0
    assert P["cascade_120b_minus_jev"]["diff"] == 0.25 and P["cascade_20b_minus_jev"]["diff"] == 0.0
    assert P["jev_wrong_by_20b_record"]["never"]["override_120b"]["rate"] == 1.0
    assert decide(s, {"comparison": "cascade_120b_minus_jev", "ci_lower_above": 0.0})["stronger_reviewer_fixes_it"]
    same = compare(plan, _reviews(plan, fix_jev_wrong=False, alone_right=True), _reviews(plan, fix_jev_wrong=False, alone_right=True))
    assert not decide(same, {"comparison": "cascade_120b_minus_jev", "ci_lower_above": 0.0})["stronger_reviewer_fixes_it"]


def test_call_count_and_pilot_plan():
    plan = _plan()
    assert n_calls(plan, 5) == 40 * 5 + 10 + 40  # alone x5, Jev's 10 wrong labels, 40 subsample reviews
    assert sum(1 for p in pilot_plan(plan, 7)) == 7
