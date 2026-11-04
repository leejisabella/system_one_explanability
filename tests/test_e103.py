import pytest

from sysone.analysis.e103 import paired_change
from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import CIVIL_TOXICITY, WILDGUARD_PROMPT
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider
from sysone.experiments.e103 import (
    BLIND_FIRST,
    CALIBRATED,
    OPPOSITE,
    OPPOSITE_PREFIX,
    WARNING,
    WARNING_NOTE,
    E103Config,
    first_alone_answers,
    protocol_kwargs,
    run_e103,
)
from sysone.runs import Run
from test_e102 import _run as run_e102_toy
from test_e102 import make_items


def _reviewer(tmp_path, provider="bedrock"):
    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return Reviewer(GuardedClient(ModelSpec("m", provider, "m", Price(1.0, 1.0)), FakeChatProvider(), guard),
                    CIVIL_TOXICITY)


def test_protocol_prompts(tmp_path):
    r = _reviewer(tmp_path)
    standard = r.build_request("hello", "toxic").messages
    warn = r.build_request("hello", "toxic", **protocol_kwargs(WARNING, jev_error_rate=0.25, own_answer=None)).messages
    calib = r.build_request("hello", "toxic", **protocol_kwargs(CALIBRATED, jev_error_rate=0.25, own_answer=None)).messages
    opp = r.build_request("hello", "toxic", **protocol_kwargs(OPPOSITE, jev_error_rate=0.25, own_answer=None)).messages
    assert len(standard) == len(warn) == len(calib) == len(opp) == 1
    assert WARNING_NOTE in warn[0].content and WARNING_NOTE not in standard[0].content
    assert "wrong about 25% of the time" in calib[0].content
    assert OPPOSITE_PREFIX in opp[0].content and opp[0].content.index(OPPOSITE_PREFIX) > opp[0].content.index("## Task")
    # Apart from the protocol's line, the prompt is the standard one.
    assert warn[0].content.replace(f"\n{WARNING_NOTE}", "") == standard[0].content


def test_blind_first_is_a_three_turn_conversation(tmp_path):
    r = _reviewer(tmp_path)
    msgs = r.build_request("hello", "toxic", **protocol_kwargs(BLIND_FIRST, jev_error_rate=0.2, own_answer="non_toxic")).messages
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0].content == r.build_request("hello").messages[0].content  # the alone prompt, unchanged
    assert msgs[1].content == '{"final_label": "non_toxic"}'
    assert '"toxic"' in msgs[2].content and "hello" not in msgs[2].content and "verdict" in msgs[2].content
    with pytest.raises(ValueError):
        r.build_request("hello", own_answer="toxic")
    with pytest.raises(ValueError):
        protocol_kwargs("shout", jev_error_rate=0.1, own_answer=None)


def test_first_alone_answers_takes_the_earliest_parsed_sample():
    base = {"status": "ok", "parse_ok": True, "condition": "alone", "reviewer": "r", "role": "reviewer", "sample": "s"}
    rows = [{**base, "item_id": "a", "sample_index": 1, "final_label": "x"},
            {**base, "item_id": "a", "sample_index": 0, "final_label": "y"},
            {**base, "item_id": "b", "sample_index": 0, "final_label": "z", "parse_ok": False},
            {**base, "item_id": "b", "sample_index": 2, "final_label": "w"},
            {**base, "item_id": "c", "sample_index": 0, "final_label": "v", "role": "reference"}]
    assert first_alone_answers(rows, "r") == {("s", "a"): "y", ("s", "b"): "w"}


def test_paired_change():
    a = {"i1": True, "i2": False, "i3": False, "i4": True}
    b = {"i1": True, "i2": True, "i3": False, "i5": True}
    d = paired_change(a, b)
    assert d["n"] == 3 and d["diff"] == pytest.approx(1 / 3, abs=1e-4)
    assert paired_change({}, b)["diff"] is None


def test_e103_end_to_end_on_a_toy_e102_run(tmp_path):
    items = make_items(16)
    status, source, _ = run_e102_toy(tmp_path, items)
    assert status == "completed"
    guard = CostGuard(Ledger.for_run(tmp_path, "e3"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    fake = lambda a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(malformed_every=0),
                                   guard, cache=None)
    config = E103Config(source_run="e", samples=["toy"], upstreams=["up"], workers=4)
    run = Run("e3", tmp_path)
    status = run_e103(config, run, fake("reviewer"), {"up": fake("up")}, samples={"toy": items}, source_dir=source.dir)
    assert status == "completed"
    plan = source.writer("plan").rows()
    per_protocol = (sum(not p["jev_correct"] for p in plan) + sum(not p["matched_correct"] for p in plan)
                    + sum(p["in_subsample"] for p in plan))
    rows = run.writer("reviews").rows()
    assert len(rows) == per_protocol * len(config.protocols)
    assert {r["protocol"] for r in rows} == set(config.protocols)
    assert all(r["own_answer"] for r in rows if r["protocol"] == BLIND_FIRST)
    assert len(run.writer("upstream").rows()) == len(plan)
    assert run.manifest()["jev_error_rate_by_sample"]["toy"] == pytest.approx(
        sum(not p["jev_correct"] for p in plan) / len(plan), abs=1e-4)
    # Resuming makes no new requests.
    status = run_e103(config, run, fake("reviewer"), {"up": fake("up")}, samples={"toy": items}, source_dir=source.dir)
    assert status == "completed" and len(run.writer("reviews").rows()) == len(rows)
    assert WILDGUARD_PROMPT.name == items[0].task
