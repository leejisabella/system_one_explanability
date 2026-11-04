import dataclasses
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


def test_committed_training_configs_load(tmp_path):
    from sysone.registry import CONFIG_DIR, Registry

    registry = Registry("t", results_dir=tmp_path)
    served = {registry.spec(a).model for a in registry.models if registry.models[a]["provider"] == "local"}
    for path in sorted((CONFIG_DIR / "posttrain").glob("*.yaml")):
        config = TrainConfig.load(path)
        assert config.method in ("sft", "dpo") and config.arm.startswith(config.method)
        # The Qwen reviewer, with thinking on, and a name vLLM serves under an alias.
        assert config.family == "qwen3" and config.base_model == "Qwen/Qwen3-14B" and config.reasoning_effort is None
        assert config.name in served
    for name, cls in (("e104", E104Config), ("e105", E105Config), ("e106", E106Config)):
        cls.load(CONFIG_DIR / "experiments" / f"{name}.yaml")
    e105 = E105Config.load(CONFIG_DIR / "experiments" / "e105.yaml")
    # E.105's traces are the self-hosted reviewer's own; no alias of it is priced above $0.
    assert {registry.spec(a).provider for a in (e105.reviewer, e105.teacher)} == {"local"}
    assert all(registry.spec(a).price.input_per_m == 0 for a in (e105.reviewer, e105.teacher))
    e106 = E106Config.load(CONFIG_DIR / "experiments" / "e106.yaml")
    # Criterion 4 leaves out the upstream model from the reviewer's own family.
    assert set(e106.criteria["held_out_upstreams"]) == set(e106.upstreams) - {"reviewer_qwen"}


def test_family_settings_render_the_reasoning_each_template_expects():
    from sysone.posttrain.hf import render

    qwen = TrainConfig(name="q", method="sft", dataset="d", arm="sft_review")
    assert qwen.template_kwargs() == {"enable_thinking": True} and qwen.lora_target_parameters() == []
    oss = TrainConfig(name="g", method="sft", dataset="d", arm="sft_review", base_model="openai/gpt-oss-20b",
                      family="gpt-oss", reasoning_effort="medium")
    assert oss.template_kwargs() == {"reasoning_effort": "medium"}
    assert "7.mlp.experts.gate_up_proj" in oss.lora_target_parameters()

    class QwenLikeTokenizer:
        """Renders like Qwen3's template: the last assistant turn's reasoning_content as <think>."""

        def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, enable_thinking=True):
            out = "".join(f"<|im_start|>{m['role']}\n" + (f"<think>\n{m['reasoning_content']}\n</think>\n\n"
                                                         if m.get("reasoning_content") else "")
                          + f"{m['content']}<|im_end|>\n" for m in messages)
            return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    prompt = [{"role": "user", "content": "Decide."}]
    completion = [{"role": "assistant", "thinking": "It reads as an insult.", "content": '{"final_label": "toxic"}'}]
    head, body = render(QwenLikeTokenizer(), prompt, completion, template_kwargs=qwen.template_kwargs(),
                        thinking_key=qwen.model_family.thinking_key)
    assert head.endswith("<|im_start|>assistant\n")
    assert body.startswith("<think>\nIt reads as an insult.\n</think>") and '"toxic"' in body
    # The wrong key for this template drops the reasoning, and render refuses to train on that.
    with pytest.raises(ValueError, match="dropped the reasoning"):
        render(QwenLikeTokenizer(), prompt, completion, template_kwargs={}, thinking_key="thinking")


def test_check_refuses_a_reasoning_setting_the_family_does_not_take(tmp_path):
    with pytest.raises(ValueError, match="has no reasoning_effort"):
        check(TrainConfig(name="t", method="sft", dataset="none", arm="sft_review", reasoning_effort="medium"),
              data_dir=tmp_path, manifest_dir=tmp_path)
    with pytest.raises(ValueError, match="set it as at evaluation"):
        check(TrainConfig(name="t", method="sft", dataset="none", arm="sft_review", family="gpt-oss"),
              data_dir=tmp_path, manifest_dir=tmp_path)


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
    assert set(crit["4_transfers_to_held_out_upstreams"]["per_upstream"]) == {"up"}
    # Criterion 4 counts only the listed upstream models; the rest are reviewed but not judged.
    judged = summarize(plan, rows, upstream, reviewers=config.reviewers, difficulty_reviewer="base_local",
                       upstreams=["up"], criteria={**config.criteria, "held_out_upstreams": ["other_family"]})
    c4 = judged["criteria"]["sft_review"]["4_transfers_to_held_out_upstreams"]
    assert c4["per_upstream"] == {} and c4["pass"] is False


# --- criterion 2's bar for the Qwen reviewer: E.103 on an E.106 source run ------------------------

def test_best_override_change_reproduces_the_preregistered_gpt_oss_bars():
    from sysone.analysis.e103 import best_override_change
    from sysone.registry import CONFIG_DIR, REPO_ROOT

    summary = json.loads((REPO_ROOT / "research_analyses" / "E103" / "data" / "summary.json").read_text())
    bars = E106Config.load(CONFIG_DIR / "experiments" / "e106.yaml").criteria["e103_best_override_change"]
    assert best_override_change(summary) == bars


def test_e103_reruns_the_protocols_on_the_base_reviewer_from_an_e106_run(tmp_path):
    from sysone.analysis.e103 import STANDARD, analyze_run, best_override_change
    from sysone.analysis.e106 import criteria_with_bars
    from sysone.experiments.e103 import BLIND_FIRST, E103Config, run_e103

    items = make_items(16)
    status, source, _ = run_e102_toy(tmp_path, items)
    assert status == "completed"
    plan = source.writer("plan").rows()
    guard = _guard(tmp_path, "e106")
    config = E106Config(source_run=source.run_id, upstream_run="none", samples=["toy"],
                        reviewers=["base_local", "sft_review"], upstreams=[], workers=4)
    e106 = Run("e106", tmp_path)
    reviewers = {a: _fake(a, FakeChatProvider(malformed_every=0), guard) for a in config.reviewers}
    (tmp_path / "runs" / "none").mkdir(parents=True)
    assert run_e106(config, e106, reviewers, samples={"toy": items}, source_dir=source.dir,
                    upstream_dir=tmp_path / "runs" / "none") == "completed"

    e103_config = E103Config(source_run="e106", plan_run=source.run_id, samples=["toy"], reviewer="base_local",
                             upstreams=[], steps=["reviews"], workers=4)
    # An E.106 run without the reviewer has no standard prompt to compare with.
    with pytest.raises(ValueError, match="no reviews by 'sft_plain'"):
        run_e103(dataclasses.replace(e103_config, reviewer="sft_plain"), Run("e103-x", tmp_path),
                 _fake("sft_plain", FakeChatProvider(), guard), {}, samples={"toy": items}, source_dir=e106.dir,
                 plan_dir=source.dir)
    e103 = Run("e103-qwen", tmp_path)
    assert run_e103(e103_config, e103, _fake("base_local", FakeChatProvider(malformed_every=0), guard), {},
                    samples={"toy": items}, source_dir=e106.dir, plan_dir=source.dir) == "completed"
    rows = e103.writer("reviews").rows()
    per_protocol = (sum(not p["jev_correct"] for p in plan) + sum(not p["matched_correct"] for p in plan)
                    + sum(p["in_subsample"] for p in plan))
    assert len(rows) == per_protocol * len(e103_config.protocols)
    # Blind-first commits to base_local's own first alone answer from E.106.
    first = {r["item_id"]: r["final_label"] for r in e106.writer("reviews").rows()
             if r["reviewer"] == "base_local" and r["condition"] == "alone" and r["sample_index"] == 0}
    assert all(r["own_answer"] == first[r["item_id"]] for r in rows if r["protocol"] == BLIND_FIRST)
    assert e103.manifest()["plan_run"] == source.run_id

    summary = analyze_run(e103.dir, e106.dir, source.dir, reviewer="base_local")
    standard = summary["samples"]["toy"]["protocols"][STANDARD]
    # The standard prompt is base_local's reviews only, not sft_review's.
    assert standard["n_reviews"] == sum(r["reviewer"] == "base_local" and r["condition"] != "alone"
                                        for r in e106.writer("reviews").rows())
    bars = best_override_change(summary)
    assert set(bars) == {"toy"}
    # E.106's analysis takes criterion 2's bars from that run once it is named.
    assert criteria_with_bars({"e103_bar_run": "e103-qwen"}, tmp_path / "runs")["e103_best_override_change"] == bars
    assert criteria_with_bars({"e103_best_override_change": {"toy": 0.5}}, tmp_path / "runs") == \
        {"e103_best_override_change": {"toy": 0.5}}
    with pytest.raises(ValueError, match="run and analyze E.103"):
        criteria_with_bars({"e103_bar_run": "never-ran"}, tmp_path / "runs")
