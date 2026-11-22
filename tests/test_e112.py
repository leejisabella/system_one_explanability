"""E.112 teacher screen: offline checks (no network)."""
from sysone.cascade.reviewer import parse_review
from sysone.experiments.e112 import RIGHT, SOLVED, UNSOLVED, E112Config, _weighted, parse_teacher, select, strata
from sysone.types import ChatResponse, Usage


def _resp(text):
    return ChatResponse(text=text, usage=Usage(10, 10), cost_usd=0.0, provider="fake", model_requested="m",
                        model_used="m", finish_reason="end_turn", raw={})


def _mix():
    rows = []
    for i, (correct, label, prop) in enumerate([(False, "toxic", "non_toxic"), (False, "toxic", "non_toxic"),
                                                (True, "toxic", "toxic")]):
        rows.append({"sample": "p", "item_id": f"i{i}", "format": "review", "condition": "jev_label",
                     "jev_correct": correct, "label": label, "proposed_label": prop})
    rows.append({"sample": "p", "item_id": "i0", "format": "alone", "condition": "alone", "jev_correct": False,
                 "label": "toxic", "proposed_label": None})
    return rows


def _trace(item, final, **kw):
    return {"sample": "p", "item_id": item, "format": "review", "condition": "jev_label", "status": "ok",
            "parse_ok": True, "final_label": final, "label": "toxic", "consistent": True, **kw}


def test_strata_unsolved_means_no_correct_review_try():
    st = strata(_mix(), [_trace("i0", "toxic"), _trace("i1", "non_toxic")])
    assert st == {("p", "i0"): SOLVED, ("p", "i1"): UNSOLVED, ("p", "i2"): RIGHT}


def test_select_is_deterministic_and_capped():
    cfg = E112Config(source_run="a", traces_run="b", samples=["p"], per_stratum={UNSOLVED: 1, SOLVED: 0, RIGHT: 1},
                     teachers={})
    a = select(_mix(), [], cfg)
    assert a == select(_mix(), [], cfg)
    assert sorted(p["stratum"] for p in a) == [RIGHT, UNSOLVED]


def test_written_reasoning_then_json():
    text = 'The comment {quotes} an insult, so the automated model is wrong.\n{"verdict": "override", "final_label": "toxic"}'
    r, written = parse_teacher(_resp(text), "k", ("non_toxic", "toxic"), "non_toxic")
    assert r.parse_ok and r.final_label == "toxic" and r.verdict == "override"
    assert written.startswith("The comment")
    assert not parse_review(_resp(text), "k", ("non_toxic", "toxic"), "non_toxic").parse_ok


def test_weighted_override_uses_mix_weights():
    rows = [{"sample": "p", "stratum": UNSOLVED, "parse_ok": True, "final_label": "toxic", "label": "toxic"},
            {"sample": "p", "stratum": SOLVED, "parse_ok": True, "final_label": "non_toxic", "label": "toxic"}]
    assert _weighted(rows, {"p": {UNSOLVED: 1, SOLVED: 3}}, ["p"]) == 0.25


def test_reasoning_inside_fenced_json_is_recovered():
    text = '```json\n{"reasoning": "It insults a person.", "verdict": "override", "final_label": "toxic"}\n```'
    r, written = parse_teacher(_resp(text), "k", ("non_toxic", "toxic"), "non_toxic")
    assert r.parse_ok and written == "It insults a person."
