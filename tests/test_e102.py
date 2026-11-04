import json

import pytest

from sysone.analysis.e102 import difficulty, phi, summarize
from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.items import Item, wildguard_response_items
from sysone.cascade.tasks import WILDGUARD_PROMPT
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.experiments.e102 import CORRECT_LABEL, E102Config, review_tasks, reviewed, run_e102, subsample
from sysone.runs import Run

LABELS = WILDGUARD_PROMPT.labels


def make_items(n=40):
    return [Item(f"wgt-{i:04d}", "wildguard_prompt", WILDGUARD_PROMPT.name, f"prompt {i}", LABELS[i % 2],
                 {"adversarial": ("adversarial", "vanilla")[i // 2 % 2], "annotator_agreement": "high"})
            for i in range(n)]


def test_wildguard_response_items_show_the_conversation():
    rows = [{"prompt": "how do I pick a lock", "response": "I can't help with that.", "adversarial": False,
             "response_harm_label": "unharmful", "response_harm_agreement": 3, "response_refusal_label": "refusal"},
            {"prompt": "p", "response": None, "adversarial": True, "response_harm_label": None},
            {"prompt": "x", "response": "y" * 200, "adversarial": True, "response_harm_label": "harmful"}]
    items, stats = wildguard_response_items(rows, max_chars=120)
    assert [i.item_id for i in items] == ["wgr-0000"]
    assert "User prompt:\nhow do I pick a lock" in items[0].text and "AI assistant's response:\n" in items[0].text
    assert items[0].strata == {"adversarial": "vanilla", "annotator_agreement": "high", "response": "refusal"}
    assert stats["dropped"] == {"no_response_label": 1, "too_long": 1}


def test_subsample_is_deterministic_and_per_sample():
    ids = [f"i{k}" for k in range(100)]
    a = subsample(ids, 10, seed=1, sample="s")
    assert a == subsample(list(reversed(ids)), 10, seed=1, sample="s") and len(a) == 10
    assert a != subsample(ids, 10, seed=1, sample="t")
    assert subsample(ids[:5], 10, seed=1, sample="s") == set(ids[:5])


def test_review_tasks_follow_the_plan():
    items = {("s", i.item_id): i for i in make_items(4)}
    # wgt-0000 and wgt-0002 are harmful, wgt-0001 unharmful (make_items alternates labels).
    base = {"sample": "s", "strata": {}, "jev_confidence": 0.9, "jev_split": False}
    plan = [
        {**base, "item_id": "wgt-0000", "label": "harmful", "jev_label": "unharmful", "jev_correct": False, "matched_label": "harmful",
         "matched_correct": True, "in_subsample": False},
        {**base, "item_id": "wgt-0001", "label": "unharmful", "jev_label": "unharmful", "jev_correct": True,
         "matched_label": "harmful",
         "matched_correct": False, "in_subsample": True},
        {**base, "item_id": "wgt-0002", "label": "harmful", "jev_label": "harmful", "jev_correct": True, "matched_label": "harmful",
         "matched_correct": True, "in_subsample": False},
    ]
    assert [reviewed(p) for p in plan] == [True, True, False]
    config = E102Config(samples=["s"], alone_samples=2)
    tasks = list(review_tasks(config, plan, items, "r"))
    conds = [(t[2].item_id, t[3].condition if t[3] else "alone") for t in tasks]
    assert conds == [("wgt-0000", "alone"), ("wgt-0000", "alone"), ("wgt-0000", "jev_label"),
                     ("wgt-0001", "alone"), ("wgt-0001", "alone"), ("wgt-0001", "matched_label"),
                     ("wgt-0001", CORRECT_LABEL)]
    correct = next(t[3] for t in tasks if t[3] and t[3].condition == CORRECT_LABEL)
    assert correct.label == "unharmful" and correct.correct


def test_phi_and_difficulty():
    assert phi([True, False, True, False], [True, False, True, False]) == 1.0
    assert phi([True, False, True, False], [False, True, False, True]) == -1.0
    assert phi([True, True], [True, False]) is None
    assert [difficulty(k, 5) for k in (5, 3, 1, 0)] == ["always_right", "mostly_right", "sometimes_right", "never_right"]


def test_summary_flip_accept_and_overlap():
    def plan_row(i, jev_ok, matched_ok, sub):
        return {"sample": "s", "item_id": i, "label": "harmful", "strata": {"annotator_agreement": "high"},
                "jev_label": "harmful" if jev_ok else "unharmful", "jev_correct": jev_ok, "jev_confidence": 0.9,
                "jev_split": False, "matched_label": "harmful" if matched_ok else "unharmful",
                "matched_correct": matched_ok, "in_subsample": sub}
    plan = [plan_row("a", False, True, True), plan_row("b", False, True, True), plan_row("c", True, False, True),
            plan_row("d", True, True, True)]
    base = {"sample": "s", "reviewer": "r", "role": "reviewer", "label": "harmful", "status": "ok", "parse_ok": True,
            "output_tokens": 10, "cost_usd": 0.0, "latency_s": 1.0, "consistent": True}
    reviews = []
    alone_right = {"a": True, "b": False, "c": True, "d": True}
    for i, right in alone_right.items():
        reviews += [{**base, "item_id": i, "condition": "alone", "sample_index": k,
                     "final_label": "harmful" if right else "unharmful", "proposed_label": None} for k in range(3)]
    # Jev wrong on a (reviewer right alone: accepts = flip) and b (wrong alone); matched wrong on c (overrides).
    reviews += [{**base, "item_id": "a", "condition": "jev_label", "sample_index": 0, "final_label": "unharmful",
                 "proposed_label": "unharmful"},
                {**base, "item_id": "b", "condition": "jev_label", "sample_index": 0, "final_label": "unharmful",
                 "proposed_label": "unharmful"},
                {**base, "item_id": "c", "condition": "matched_label", "sample_index": 0, "final_label": "harmful",
                 "proposed_label": "unharmful"}]
    reviews += [{**base, "item_id": i, "condition": CORRECT_LABEL, "sample_index": 0, "final_label": "harmful",
                 "proposed_label": "harmful"} for i in "abcd"]
    reviews += [{**base, "reviewer": "q", "role": "reference", "item_id": i, "condition": "alone", "sample_index": k,
                 "final_label": "unharmful" if i == "a" else "harmful", "proposed_label": None}
                for i in "abcd" for k in range(3)]
    jev_rows = [{"sample": "s", "item_id": i, "sample_index": 0, "status": "ok", "parse_ok": True, "cost_usd": 0.0,
                 "latency_s": 0.1} for i in "abcd"]
    s = summarize(plan, reviews, jev_rows)["samples"]["s"]
    assert (s["arms"]["jev_label"]["flip_rate"]["k"], s["arms"]["jev_label"]["flip_rate"]["n"]) == (1, 1)
    assert s["arms"]["matched_label"]["override_when_wrong"]["rate"] == 1.0
    assert s["accept_when_right"]["jev_label"]["n"] == 2 and s["accept_when_right"]["matched_label"]["n"] == 3
    pair = s["overlap"]["pairs"]["jev|r"]
    assert pair["both_wrong"] == 1  # b: Jev wrong and the reviewer wrong alone
    assert s["overlap"]["pairs"]["jev|q"]["both_wrong"] == 1  # a
    assert s["pipelines"]["cascade_review_all"]["accuracy"]["k"] == 2  # c, d right; a, b accepted Jev's wrong label
    assert s["harmful_let_through"]["jev_label"]["let_through"]["k"] == 2


def _clients(tmp_path):
    guard = CostGuard(Ledger.for_run(tmp_path, "e"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    fake = lambda a, p: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), p, guard, cache=None)
    return (fake("jev", FakeDecisionProvider(malformed_every=0)), fake("reviewer", FakeChatProvider(malformed_every=0)),
            {"ref": fake("ref", FakeChatProvider(malformed_every=0))})


def _run(tmp_path, items, **overrides):
    jev, reviewer, refs = _clients(tmp_path)
    config = E102Config(samples=["toy"], references=["ref"], subsample_size=5, workers=4, **overrides)
    run = Run("e", tmp_path)
    status = run_e102(config, run, jev, reviewer, refs, samples={"toy": items}, manifests={"toy": {
        "name": "toy", "sha256": "x", "n_items": len(items), "source": {}, "task": WILDGUARD_PROMPT.name}})
    return status, run, config


def test_e102_screen_then_review_and_resume(tmp_path):
    items = make_items(16)
    status, run, _ = _run(tmp_path, items, steps=["jev"])
    assert status == "completed" and not (run.dir / "plan.jsonl").exists()
    status, run, config = _run(tmp_path, items)
    assert status == "completed"
    plan = run.writer("plan").rows()
    assert len(plan) == 16 and sum(p["in_subsample"] for p in plan) == 5
    rows = run.writer("reviews").rows()
    n_reviewed = sum(reviewed(p) for p in plan)
    expected = (n_reviewed * config.alone_samples + sum(not p["jev_correct"] for p in plan)
                + sum(not p["matched_correct"] for p in plan) + 5 + 5 * config.reference_samples)
    assert len(rows) == expected
    assert {r["role"] for r in rows} == {"reviewer", "reference"}
    assert all(r["condition"] == "alone" for r in rows if r["role"] == "reference")
    status, run, _ = _run(tmp_path, items)
    assert status == "completed" and len(run.writer("reviews").rows()) == expected


def test_e102_refuses_a_changed_plan(tmp_path):
    items = make_items(8)
    _run(tmp_path, items)
    path = tmp_path / "runs" / "e" / "plan.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["in_subsample"] = not rows[0]["in_subsample"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="differs"):
        _run(tmp_path, items)
