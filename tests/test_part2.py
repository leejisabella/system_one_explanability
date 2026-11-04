import json
from collections import Counter

import httpx
import pytest

from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.items import Item, wildguard_items, wildguard_response_items
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import CIVIL_TOXICITY, WILDGUARD_PROMPT
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.clients.local import LocalProvider, reasoning_text
from sysone.experiments.e104 import E104Config, run_e104
from sysone.experiments.e105 import E105Config, export_dataset, needs_more, run_e105, targets
from sysone.experiments.e106 import UPSTREAM_WRONG, E106Config, run_e106, upstream_wrong
from sysone.posttrain import formats
from sysone.posttrain.mix import ALONE_GROUP, JEV_RIGHT, JEV_WRONG, MixSettings, build_mix, split_of
from sysone.posttrain.train_config import TrainConfig, check
from sysone.runs import Run
from sysone.types import ChatRequest, Message
from test_e102 import _run as run_e102_toy
from test_e102 import make_items

LABELS = WILDGUARD_PROMPT.labels


def _guard(tmp_path, name="r"):
    return CostGuard(Ledger.for_run(tmp_path, name), per_call_cap_usd=2.0, run_cap_usd=25.0)


def _fake(alias, provider, guard):
    return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0)), provider, guard, cache=None)


# --- training pools -----------------------------------------------------------------------------

def test_training_pools_drop_test_prompts_and_mark_unknown_agreement():
    rows = [{"prompt": "a", "adversarial": False, "prompt_harm_label": "harmful"},
            {"prompt": "in test", "adversarial": True, "prompt_harm_label": "unharmful"},
            {"prompt": "c", "adversarial": True, "prompt_harm_label": "unharmful"}]
    items, stats = wildguard_items(rows, n_per_cell=None, seed=0, max_chars=100, id_prefix="wgtrain-p",
                                   exclude_prompts=frozenset({"in test"}))
    assert [i.item_id for i in items] == ["wgtrain-p-00000", "wgtrain-p-00002"]
    assert stats["dropped"] == {"prompt_in_excluded_split": 1}
    assert items[0].strata["annotator_agreement"] == "unknown"  # WildGuardTrain has no agreement
    resp = [{"prompt": f"p{i}", "response": "r", "adversarial": i % 2 == 0, "response_harm_label": LABELS[i % 2]}
            for i in range(20)]
    picked, stats = wildguard_response_items(resp, max_chars=100, id_prefix="wgtrain-r", n_per_cell=3, seed=1)
    assert len(picked) == 6 and stats["pool"] == 20
    assert picked == wildguard_response_items(list(resp), max_chars=100, id_prefix="wgtrain-r", n_per_cell=3, seed=1)[0]


# --- the self-hosted client ---------------------------------------------------------------------

def test_local_provider_sends_openai_chat_and_reads_reasoning():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={
            "model": "gpt-oss-20b-base", "choices": [{"finish_reason": "stop", "message": {
                "content": '{"final_label": "toxic"}', "reasoning_content": "It insults someone."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40}})

    p = LocalProvider("http://gpu:8000/v1/", http=httpx.Client(transport=httpx.MockTransport(handler)))
    req = ChatRequest(messages=(Message("user", "hi"),), max_output_tokens=256, reasoning_max_tokens=4096,
                      reasoning_effort="medium")
    out = p.send("gpt-oss-20b-base", req, Price(0, 0))
    assert seen["url"] == "http://gpu:8000/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 4352 and seen["body"]["reasoning_effort"] == "medium"
    assert out.text == '{"final_label": "toxic"}' and out.usage.output_tokens == 40
    assert reasoning_text(out.raw) == "It insults someone."
    assert reasoning_text({"choices": [{"message": {"reasoning": "newer vLLM"}}]}) == "newer vLLM"
    assert reasoning_text({}) is None


def test_local_calls_pass_a_zero_project_cap_but_paid_ones_would_not(tmp_path):
    from sysone.budget.guard import ProjectBudgetExceeded

    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0,
                      project_caps_usd={"local": 0.0})
    guard.release(guard.reserve(0.0, provider="local", model="m", tags={}))
    with pytest.raises(ProjectBudgetExceeded):
        guard.reserve(0.001, provider="local", model="m", tags={})


# --- the mix ------------------------------------------------------------------------------------

def _jev_for(items, wrong_every=4):
    other = {LABELS[0]: LABELS[1], LABELS[1]: LABELS[0]}
    return {i.item_id: {"label": other[i.label] if k % wrong_every == 0 else i.label, "confidence": 0.8, "split": False}
            for k, i in enumerate(items)}


def test_mix_groups_are_disjoint_and_sized_by_ratio():
    items = make_items(80)
    jev = _jev_for(items)
    rows = build_mix("toy", items, jev, LABELS, MixSettings(jev_right_per_wrong=2.0, alone_per_wrong=1.0, val_fraction=0.2))
    groups = Counter(r["group"] for r in rows)
    assert groups == {JEV_WRONG: 20, JEV_RIGHT: 40, ALONE_GROUP: 20}
    assert len({r["item_id"] for r in rows}) == len(rows)  # an item is in one group only
    for r in rows:
        assert r["target_label"] == r["label"]  # always the gold label, never Jev's
        if r["group"] == JEV_WRONG:
            assert r["proposed_label"] != r["label"] and r["target_verdict"] == "override"
        elif r["group"] == JEV_RIGHT:
            assert r["proposed_label"] == r["label"] and r["target_verdict"] == "accept"
        else:
            assert r["proposed_label"] is None and r["format"] == "alone"
    assert rows == build_mix("toy", list(reversed(items)), jev, LABELS,
                             MixSettings(jev_right_per_wrong=2.0, alone_per_wrong=1.0, val_fraction=0.2))
    assert {r["split"] for r in rows} == {"train", "val"}
    assert split_of("toy", "x", 0.2, 104) == split_of("toy", "x", 0.2, 104)
    capped = build_mix("toy", items, jev, LABELS, MixSettings(max_jev_wrong=5))
    assert Counter(r["group"] for r in capped)[JEV_WRONG] == 5


# --- training records ---------------------------------------------------------------------------

def test_training_prompts_are_the_evaluation_prompts(tmp_path):
    client = GuardedClient(ModelSpec("m", "local", "m", Price(0, 0)), LocalProvider("http://x"), _guard(tmp_path))
    r = Reviewer(client, CIVIL_TOXICITY)
    assert formats.prompt_messages(CIVIL_TOXICITY.name, "hello", "toxic") == [
        {"role": "user", "content": r.build_request("hello", "toxic").messages[0].content}]
    assert formats.prompt_messages(CIVIL_TOXICITY.name, "hello", None)[0]["content"] == \
        r.build_request("hello").messages[0].content
    assert formats.answer_json("toxic", "non_toxic") == '{"verdict": "override", "final_label": "toxic"}'
    assert formats.answer_json("toxic", "toxic") == '{"verdict": "accept", "final_label": "toxic"}'
    assert formats.answer_json("toxic", None) == '{"final_label": "toxic"}'


def test_trace_choice_prefers_the_reviewer_and_rejected_prefers_deference():
    base = {"status": "ok", "parse_ok": True, "label": "toxic", "proposed_label": "non_toxic", "reasoning": "r",
            "finish_reason": "stop", "consistent": True}
    traces = [{**base, "source": "teacher", "sample_index": 0, "final_label": "toxic"},
              {**base, "source": "reviewer", "sample_index": 2, "final_label": "toxic"},
              {**base, "source": "reviewer", "sample_index": 1, "final_label": "toxic", "reasoning": None},
              {**base, "source": "reviewer", "sample_index": 3, "final_label": "non_toxic", "verdict": "accept"},
              {**base, "source": "reviewer", "sample_index": 4, "final_label": "toxic", "consistent": False}]
    assert formats.choose(traces)["sample_index"] == 2
    assert formats.rejected(traces)["sample_index"] == 3  # deferred to the wrong proposal
    assert formats.choose(traces[3:4]) is None
    assert needs_more([traces[1]], min_samples=2, max_samples=4)            # one correct try: sample again
    assert not needs_more([traces[1], traces[3]], min_samples=2, max_samples=4)  # correct and wrong: done
    assert not needs_more([{**traces[3], "sample_index": k} for k in range(4)], min_samples=2, max_samples=4)


# --- E.104 -> E.105 -> training check, end to end on fake models --------------------------------

def _pipeline(tmp_path):
    items = make_items(60)
    guard = _guard(tmp_path, "e104")
    config4 = E104Config(samples=["toy"], workers=4, mix={"jev_right_per_wrong": 0.2, "alone_per_wrong": 0.2,
                                                          "val_fraction": 0.2})
    run4 = Run("e104", tmp_path)
    manifests = {"toy": {"name": "toy", "sha256": "x", "n_items": 60, "source": {}, "task": WILDGUARD_PROMPT.name}}
    assert run_e104(config4, run4, _fake("jev", FakeDecisionProvider(malformed_every=0), guard),
                    samples={"toy": items}, manifests=manifests) == "completed"
    mix = run4.writer("mix").rows()
    # Resuming rebuilds the same mix and makes no Jev calls.
    n_jev = len(run4.writer("jev_decisions").rows())
    assert run_e104(config4, run4, _fake("jev", FakeDecisionProvider(malformed_every=0), guard),
                    samples={"toy": items}, manifests=manifests) == "completed"
    assert run4.writer("mix").rows() == mix and len(run4.writer("jev_decisions").rows()) == n_jev

    config5 = E105Config(source_run="e104", samples=["toy"], workers=4,
                         datasets=[{"name": "toy_ds", "samples": ["toy"]}])
    run5 = Run("e105", tmp_path)
    g5 = _guard(tmp_path, "e105")
    dirs = {"data_dir": tmp_path / "data", "manifest_dir": tmp_path / "manifests"}
    status = run_e105(config5, run5, _fake("reviewer", FakeChatProvider(malformed_every=0), g5),
                      _fake("teacher", FakeChatProvider(malformed_every=0), g5),
                      samples={"toy": items}, source_dir=run4.dir, **dirs)
    assert status == "completed"
    return items, mix, run5, dirs


def test_e105_traces_and_exports_paired_arms(tmp_path):
    items, mix, run5, dirs = _pipeline(tmp_path)
    traces = run5.writer("traces").rows()
    todo = targets(mix)
    assert len(todo) == len(mix) + sum(r["format"] == "review" for r in mix)
    per_target = Counter((t["sample"], t["item_id"], t["format"], t["condition"]) for t in traces if t["source"] == "reviewer")
    assert set(per_target.values()) <= {2, 3, 4} and set(per_target) == {t[:4] for t in todo}
    folder = dirs["data_dir"] / "toy_ds"
    load = lambda name: [json.loads(x) for x in (folder / name).read_text().splitlines()]
    review = load("sft_review.train.jsonl") + load("sft_review.val.jsonl")
    plain = load("sft_plain.train.jsonl") + load("sft_plain.val.jsonl")
    # Both arms train on exactly the same items and gold labels; only the format differs.
    pair = lambda recs: sorted((r["meta"]["item_id"], r["meta"]["label"]) for r in recs)
    assert review and pair(review) == pair(plain)
    assert all(r["meta"]["format"] == "alone" for r in plain)
    assert {r["meta"]["format"] for r in review} == {"review", "alone"}
    for r in review:
        assert json.loads(r["completion"][0]["content"])["final_label"] == r["meta"]["label"]
        assert r["completion"][0]["thinking"]
    manifest = json.loads((dirs["manifest_dir"] / "toy_ds" / "manifest.json").read_text())
    assert manifest["files"]["sft_review.train.jsonl"]["n"] == len(load("sft_review.train.jsonl"))
    # A re-export with different data under the same name is refused.
    with pytest.raises(ValueError, match="different data"):
        export_dataset("toy_ds", ["toy"], mix, [], {("toy", i.item_id): i for i in items}, source={}, **dirs)

    for arm in ("sft_review", "sft_plain"):
        config = TrainConfig(name=f"t-{arm}", method="sft", dataset="toy_ds", arm=arm, max_length=4096)
        stats = check(config, data_dir=dirs["data_dir"], manifest_dir=dirs["manifest_dir"])
        assert stats["train"]["records"] > 0
    (folder / "sft_plain.train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest hash"):
        check(TrainConfig(name="t", method="sft", dataset="toy_ds", arm="sft_plain"),
              data_dir=dirs["data_dir"], manifest_dir=dirs["manifest_dir"])
    with pytest.raises(ValueError, match="doesn't go with"):
        check(TrainConfig(name="t", method="dpo", dataset="toy_ds", arm="sft_review", init_from="x"),
              data_dir=dirs["data_dir"], manifest_dir=dirs["manifest_dir"])


def test_committed_training_configs_load():
    from sysone.registry import CONFIG_DIR

    for path in sorted((CONFIG_DIR / "posttrain").glob("*.yaml")):
        config = TrainConfig.load(path)
        assert config.method in ("sft", "dpo") and config.arm.startswith(config.method)
    for name, cls in (("e104", E104Config), ("e105", E105Config), ("e106", E106Config)):
        cls.load(CONFIG_DIR / "experiments" / f"{name}.yaml")


# --- E.106 evaluation and its analysis ----------------------------------------------------------

def test_e106_evaluates_each_reviewer_on_the_e102_plan(tmp_path):
    from sysone.analysis.e106 import summarize

    items = make_items(24)
    status, source, _ = run_e102_toy(tmp_path, items)
    assert status == "completed"
    plan = source.writer("plan").rows()
    up_dir = tmp_path / "runs" / "e103"
    up_dir.mkdir(parents=True)
    other = {LABELS[0]: LABELS[1], LABELS[1]: LABELS[0]}
    upstream = [{"reviewer": "up", "status": "ok", "parse_ok": True, "sample": "toy", "item_id": p["item_id"],
                 "label": p["label"], "final_label": other[p["label"]] if k % 3 == 0 else p["label"]}
                for k, p in enumerate(plan)]
    (up_dir / "upstream.jsonl").write_text("".join(json.dumps(r) + "\n" for r in upstream))
    picked = upstream_wrong(plan, upstream, ["up"], 5, 106)
    assert len(picked[("toy", "up")]) == 5 and picked == upstream_wrong(plan, upstream, ["up"], 5, 106)

    config = E106Config(source_run="e", upstream_run="e103", samples=["toy"], reviewers=["base_local", "sft_review"],
                        upstreams=["up"], upstream_wrong_per_sample=5, workers=4,
                        criteria={"controls": {"sft_review": "sft_plain"}})
    guard = _guard(tmp_path, "e106")
    reviewers = {a: _fake(a, FakeChatProvider(malformed_every=0), guard) for a in config.reviewers}
    run = Run("e106", tmp_path)
    assert run_e106(config, run, reviewers, samples={"toy": items}, source_dir=source.dir, upstream_dir=up_dir) == "completed"
    rows = run.writer("reviews").rows()
    alone = Counter((r["reviewer"], r["item_id"]) for r in rows if r["condition"] == "alone")
    assert {n for (a, _), n in alone.items() if a == "base_local"} == {5}
    assert {n for (a, _), n in alone.items() if a == "sft_review"} == {1}
    assert sum(r["condition"] == f"{UPSTREAM_WRONG}:up" for r in rows) == 5 * 2
    # Resuming makes no new requests.
    assert run_e106(config, run, reviewers, samples={"toy": items}, source_dir=source.dir, upstream_dir=up_dir) == "completed"
    assert len(run.writer("reviews").rows()) == len(rows)

    summary = summarize(plan, rows, upstream, reviewers=config.reviewers, difficulty_reviewer="base_local",
                        upstreams=["up"], criteria=config.criteria)
    assert set(summary["reviewers"]) == {"base_local", "sft_review"}
    crit = summary["criteria"]["sft_review"]
    assert set(crit) == {"1_trade_off", "2_beats_cheap_protocols", "3_cascade_beats_best_single",
                         "4_transfers_to_held_out_upstreams", "5_beats_control_arm"}
    assert crit["5_beats_control_arm"]["pass"] is False  # no control arm was evaluated
    assert "vs_base" in summary["reviewers"]["sft_review"]["pooled"]
