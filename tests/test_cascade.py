import json

import pytest

from sysone.analysis.e101 import bootstrap_diff, summarize, wilson
from sysone.budget import CostGuard, Ledger, Price
from sysone.budget.guard import RunBudgetExceeded
from sysone.cascade.items import Item, civil_items, freeze_items, load_items, stratified_sample, wildguard_items
from sysone.cascade.proposals import (
    JEV_CONF,
    JEV_LABEL,
    MATCHED_CONF,
    MATCHED_LABEL,
    JevDecision,
    build_proposals,
    combine_jev,
    flip_condition,
)
from sysone.cascade.reviewer import Reviewer, parse_review
from sysone.cascade.tasks import CIVIL_TOXICITY, WILDGUARD_PROMPT
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider, FakeProvider, ok_result
from sysone.decision.jev import REFUND_TASK, JevDecisionModel
from sysone.experiments.e101 import E101Config, run_e101
from sysone.runs import Run

LABELS = WILDGUARD_PROMPT.labels  # ("harmful", "unharmful")


def make_items(n=40):
    return [Item(f"wgt-{i:04d}", "wildguard_prompt", WILDGUARD_PROMPT.name, f"prompt {i}",
                 LABELS[i % 2], {"adversarial": ("adversarial", "vanilla")[i // 2 % 2], "annotator_agreement": "high"})
            for i in range(n)]


def test_stratified_sample_is_deterministic_and_balanced():
    rows = [{"id": i, "cell": i % 3} for i in range(100)]
    a = stratified_sample(rows, lambda r: (r["cell"],), 10, seed=1, row_id=lambda r: r["id"])
    b = stratified_sample(list(reversed(rows)), lambda r: (r["cell"],), 10, seed=1, row_id=lambda r: r["id"])
    assert sorted(r["id"] for r in a) == sorted(r["id"] for r in b)
    assert [sum(r["cell"] == c for r in a) for c in range(3)] == [10, 10, 10]
    small = stratified_sample(rows[:4], lambda r: (r["cell"],), 10, seed=1, row_id=lambda r: r["id"])
    assert len(small) == 4


def test_wildguard_items_drop_unlabeled_and_long():
    rows = [{"prompt": "hi", "adversarial": False, "prompt_harm_label": "unharmful", "prompt_harm_agreement": 3},
            {"prompt": "x" * 50, "adversarial": True, "prompt_harm_label": "harmful", "prompt_harm_agreement": 2},
            {"prompt": "no label", "adversarial": True, "prompt_harm_label": None}]
    items, stats = wildguard_items(rows, n_per_cell=5, seed=0, max_chars=10)
    assert [i.item_id for i in items] == ["wgt-0000"]
    assert stats["dropped"] == {"too_long_or_empty": 1, "no_prompt_label": 1}
    assert items[0].strata == {"adversarial": "vanilla", "annotator_agreement": "high"}


def test_civil_items_exclude_jigsaw_train_and_texts_in_hf_train():
    def row(i, text, tox):
        return {"id": i, "comment_text": text, "toxicity": tox, "identity_attack": 0.0, "identity_any": 0,
                "toxicity_annotator_count": 4}
    rows = [row(100, "train id", 0.9), row(7_000_001, "seen in train", 0.9), row(7_000_002, "fresh", 0.5),
            row(7_000_003, "calm", 0.0)]
    items, stats = civil_items(rows, {"seen in train"}, n_per_cell=5, seed=0, max_chars=100)
    assert {i.item_id: i.label for i in items} == {"cc-7000002": "toxic", "cc-7000003": "non_toxic"}
    assert stats["dropped"] == {"jigsaw_train_id": 1, "text_in_hf_train": 1}
    by_id = {i.item_id: i for i in items}
    assert by_id["cc-7000002"].strata["annotator_agreement"] == "low"
    assert by_id["cc-7000003"].strata["annotator_agreement"] == "high"


def test_freeze_and_load_refuse_edited_items(tmp_path):
    items = make_items(4)
    freeze_items(items, "toy", {"task": WILDGUARD_PROMPT.name}, data_dir=tmp_path / "data", manifest_dir=tmp_path / "m")
    loaded, manifest = load_items("toy", data_dir=tmp_path / "data", manifest_dir=tmp_path / "m")
    assert loaded == sorted(items, key=lambda i: i.item_id) and manifest["n_items"] == 4
    with pytest.raises(FileExistsError):
        freeze_items(items, "toy", {}, data_dir=tmp_path / "data", manifest_dir=tmp_path / "m")
    path = tmp_path / "data" / "cascade" / "toy" / "items.jsonl"
    path.write_text(path.read_text().replace("prompt 1", "prompt one"))
    with pytest.raises(ValueError, match="does not match"):
        load_items("toy", data_dir=tmp_path / "data", manifest_dir=tmp_path / "m")


def test_combine_jev_averages_rotated_samples():
    d = combine_jev([{"harmful": 0.9, "unharmful": 0.1}, {"harmful": 0.3, "unharmful": 0.7}], LABELS)
    assert d == JevDecision("harmful", 0.6, 2, True)
    tie = combine_jev([{"harmful": 0.5, "unharmful": 0.5}], LABELS)
    assert tie.label == "harmful" and not tie.split
    assert combine_jev([], LABELS) is None


def test_matched_proposals_keep_jev_confusion_matrix_on_random_items():
    items = make_items(40)
    # Jev wrong on the first 6 harmful and first 2 unharmful items, with lower confidence when wrong.
    jev = {}
    for i in items:
        idx = int(i.item_id[-4:])
        wrong = (i.label == "harmful" and idx < 12) or (i.label == "unharmful" and idx < 4)
        label = [lab for lab in LABELS if lab != i.label][0] if wrong else i.label
        jev[i.item_id] = JevDecision(label, 0.55 if wrong else 0.95, 3, False)
    del jev["wgt-0039"]  # Jev never answered it
    props = build_proposals(items, jev, LABELS, seed=7, flip_ps=[0.2], dataset="toy")
    assert {p.item_id for p in props} == set(jev)
    by = lambda cond: {p.item_id: p for p in props if p.condition == cond}
    gold = {i.item_id: i.label for i in items}
    for cond in (MATCHED_LABEL, MATCHED_CONF):
        for label in LABELS:
            n_wrong = sum(not p.correct for p in by(cond).values() if gold[p.item_id] == label)
            n_jev = sum(not p.correct for p in by(JEV_LABEL).values() if gold[p.item_id] == label)
            assert n_wrong == n_jev
    assert by(MATCHED_LABEL).keys() == by(JEV_LABEL).keys()
    # Matched errors land on other items than Jev's (with this seed), and confidences come
    # from Jev's pool with the same correctness.
    assert {i for i, p in by(MATCHED_LABEL).items() if not p.correct} != {i for i, p in by(JEV_LABEL).items() if not p.correct}
    for p in by(MATCHED_CONF).values():
        assert p.confidence == (0.95 if p.correct else 0.55)
    assert all(p.confidence is None for p in by(MATCHED_LABEL).values())
    assert all(p.confidence == jev[p.item_id].confidence for p in by(JEV_CONF).values())
    again = build_proposals(items, jev, LABELS, seed=7, flip_ps=[0.2], dataset="toy")
    assert again == props
    assert by(flip_condition(0.2)) and all(p.label == gold[i] for i, p in by(flip_condition(0.2)).items() if p.correct)


def guarded(tmp_path, provider, *, provider_name="fake", run_caps=None, alias="m"):
    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0,
                      run_provider_caps_usd=run_caps)
    return GuardedClient(ModelSpec(alias, provider_name, "model", Price(1.0, 1.0)), provider, guard,
                         sleep=lambda s: None)


def test_review_prompt_is_identical_for_jev_and_random_proposals(tmp_path):
    reviewer = Reviewer(guarded(tmp_path, FakeChatProvider()), WILDGUARD_PROMPT)
    a = reviewer.build_request("hello", "harmful").messages[0].content
    b = reviewer.build_request("hello", "harmful").messages[0].content
    assert a == b and "automated moderation model" in a and "confidence" not in a
    with_conf = reviewer.build_request("hello", "harmful", 0.873).messages[0].content
    assert "confidence in this decision: 87%" in with_conf
    alone = reviewer.build_request("hello").messages[0].content
    assert "automated" not in alone and WILDGUARD_PROMPT.policy in alone


@pytest.mark.parametrize("text,proposal,final,verdict,consistent,ok", [
    ('{"final_label": "toxic"}', None, "toxic", None, True, True),
    ('{"verdict": "accept", "final_label": "toxic"}', "toxic", "toxic", "accept", True, True),
    ('{"verdict": "override", "final_label": "non_toxic"}', "toxic", "non_toxic", "override", True, True),
    ('{"verdict": "accept", "final_label": "non_toxic"}', "toxic", "non_toxic", "accept", False, True),
    ('{"verdict": "maybe", "final_label": "toxic"}', "toxic", None, None, None, False),
    ('{"final_label": "spam"}', None, None, None, None, False),
    ("I can't help with that.", None, None, None, None, False),
])
def test_parse_review(text, proposal, final, verdict, consistent, ok):
    from sysone.clients.base import GuardedClient  # noqa: F401  (keeps the import set small)
    from sysone.types import ChatResponse, Usage

    resp = ChatResponse(text, Usage(1, 1), 0.0, "fake", "m", "m", "stop", {})
    r = parse_review(resp, "k", CIVIL_TOXICITY.labels, proposal)
    assert (r.final_label, r.verdict, r.consistent, r.parse_ok) == (final, verdict, consistent, ok)


def test_refund_jev_requests_keep_the_case_file_key(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeDecisionProvider()), task=REFUND_TASK)
    assert set(jev.build_request("policy", "case").state) == {"refund_policy", "case_file"}
    cc = JevDecisionModel(guarded(tmp_path, FakeDecisionProvider()), task=CIVIL_TOXICITY.jev)
    assert set(cc.build_request("policy", "case").state) == {"comment_policy", "comment"}


def test_run_provider_cap_stops_one_provider(tmp_path):
    provider = FakeProvider([ok_result(input_tokens=1000, output_tokens=0) for _ in range(5)])
    client = guarded(tmp_path, provider, provider_name="bedrock", run_caps={"bedrock": 0.0012})
    from sysone.types import ChatRequest

    request = lambda i: ChatRequest.from_prompt(f"hi {i}", max_output_tokens=500)
    client.complete(request(0))  # worst case about $0.0005 (500 output tokens), actual $0.001
    with pytest.raises(RunBudgetExceeded, match="bedrock cap"):
        client.complete(request(1))  # $0.001 spent + about $0.0005 worst case > $0.0012
    assert client.guard.run_spent_usd("bedrock") == pytest.approx(0.001)


def test_wilson_and_bootstrap():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    d = bootstrap_diff([True] * 8 + [False] * 2, [True] * 2 + [False] * 8, seed=1)
    assert d["diff"] == 0.6 and d["ci95"][0] > 0


def test_flip_rate_counts_only_items_the_reviewer_gets_right_alone():
    strata = {"adversarial": "vanilla", "annotator_agreement": "high"}
    jev_rows = [{"sample": "s", "item_id": i, "label": "harmful", "strata": strata, "status": "ok", "parse_ok": True}
                for i in ("a", "b", "c")]
    proposals = [{"sample": "s", "item_id": i, "condition": JEV_LABEL, "label": "unharmful", "correct": False,
                  "confidence": None, "jev_split": False} for i in ("a", "b", "c")]
    base = {"reviewer": "r", "sample": "s", "label": "harmful", "strata": strata, "status": "ok", "parse_ok": True,
            "output_tokens": 10, "input_tokens": 5}
    reviews = []
    # Alone: right on a and b, wrong on c.
    for i, final in (("a", "harmful"), ("b", "harmful"), ("c", "unharmful")):
        reviews += [{**base, "item_id": i, "condition": "alone", "sample_index": k, "final_label": final,
                     "proposed_label": None, "proposed_correct": None, "consistent": True} for k in range(3)]
    # With Jev's wrong proposal: accepts on a and c, overrides on b.
    for i, final in (("a", "unharmful"), ("b", "harmful"), ("c", "unharmful")):
        reviews.append({**base, "item_id": i, "condition": JEV_LABEL, "sample_index": 0, "final_label": final,
                        "proposed_label": "unharmful", "proposed_correct": False, "consistent": True})
    s = summarize(jev_rows, proposals, reviews)["reviewers"]["r"]["s"]
    c = s["conditions"][JEV_LABEL]
    assert (c["flip_rate"]["k"], c["flip_rate"]["n"]) == (1, 2)
    assert (c["override_when_wrong"]["k"], c["override_when_wrong"]["n"]) == (1, 3)
    assert s["alone_difficulty"] == {"always_right": 2, "never_right": 1}


def _toy_run(tmp_path, items, **overrides):
    from sysone.budget import CostGuard, Ledger

    guard = CostGuard(Ledger.for_run(tmp_path, "e"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    jev = GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(0.0, 0.0)), FakeDecisionProvider(malformed_every=0),
                        guard, cache=None)
    reviewers = {"reviewer": GuardedClient(ModelSpec("reviewer", "fake", "fake-r", Price(0.0, 0.0)),
                                           FakeChatProvider(malformed_every=0), guard, cache=None)}
    config = E101Config(samples=["toy"], workers=4, flip_ps=[0.2], **overrides)
    run = Run("e", tmp_path)
    status = run_e101(config, run, jev, reviewers, samples={"toy": items}, manifests={"toy": {
        "name": "toy", "sha256": "x", "n_items": len(items), "source": {}, "task": WILDGUARD_PROMPT.name}})
    return status, run, config


def test_e101_end_to_end_with_fakes_and_resume(tmp_path):
    items = make_items(12)
    status, run, config = _toy_run(tmp_path, items)
    assert status == "completed"
    reviews = run.writer("reviews").rows()
    per_item = config.alone_samples + len(config.review_conditions)
    assert len(reviews) == len(items) * per_item
    assert len(run.writer("jev_decisions").rows()) == len(items) * config.jev_samples
    assert run.manifest()["model_version"] == "jev-fake-1.0.0"
    status, run, _ = _toy_run(tmp_path, items)
    assert status == "completed" and len(run.writer("reviews").rows()) == len(items) * per_item


def test_e101_refuses_proposals_that_changed(tmp_path):
    items = make_items(6)
    _toy_run(tmp_path, items)
    path = tmp_path / "runs" / "e" / "proposals.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label"] = "other"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="differs"):
        _toy_run(tmp_path, items)


def test_e101_halts_review_when_jev_is_incomplete(tmp_path):
    items = make_items(4)
    status, run, _ = _toy_run(tmp_path, items, steps=["jev"])
    path = tmp_path / "runs" / "e" / "jev_decisions.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[1:]) + "\n")
    status, run, _ = _toy_run(tmp_path, items, steps=["review"])
    assert status.startswith("halted") and "lack some" in status
    assert not run.writer("reviews").rows()
