import json
from collections import Counter

import pytest

from sysone.analysis.e106 import GROUP_A, GROUP_B, GROUP_C, base_groups, evaluate_round2
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.cascade.tasks import WILDGUARD_PROMPT
from sysone.experiments.e104 import E104Config, run_e104
from sysone.experiments.e105 import E105Config, run_e105
from sysone.experiments.e108 import SAMPLE_OFFSET, E108Config, export_round2, needs_review_sample, run_e108
from sysone.posttrain import formats
from sysone.posttrain.mix import JEV_RIGHT, JEV_WRONG, MATCHED_WRONG
from sysone.posttrain.train_config import TrainConfig, check
from sysone.runs import Run
from test_e102 import make_items
from test_part2 import _fake, _guard


def _round2_pipeline(tmp_path):
    items = make_items(160)
    manifests = {"toy": {"name": "toy", "sha256": "x", "n_items": 160, "source": {}, "task": WILDGUARD_PROMPT.name}}
    run4 = Run("e104", tmp_path)
    config4 = E104Config(samples=["toy"], workers=4, mix={"jev_right_per_wrong": 2.0, "alone_per_wrong": 1.0, "max_jev_wrong": 24,
                                                          "val_fraction": 0.2})
    assert run_e104(config4, run4, _fake("jev", FakeDecisionProvider(malformed_every=0), _guard(tmp_path, "e104")),
                    samples={"toy": items}, manifests=manifests) == "completed"
    run5 = Run("e105", tmp_path)
    g5 = _guard(tmp_path, "e105")
    config5 = E105Config(source_run="e104", samples=["toy"], workers=4, steps=["traces"])
    assert run_e105(config5, run5, _fake("reviewer", FakeChatProvider(malformed_every=0), g5),
                    _fake("teacher", FakeChatProvider(malformed_every=0), g5), samples={"toy": items},
                    source_dir=run4.dir) == "completed"
    run8 = Run("e108", tmp_path)
    provider = FakeChatProvider(malformed_every=0)
    config8 = E108Config(source_run="e104", trace_run="e105", samples=["toy"], dataset="toy_r2", workers=4)
    dirs = {"data_dir": tmp_path / "data", "manifest_dir": tmp_path / "manifests"}
    status = run_e108(config8, run8, _fake("reviewer", provider, _guard(tmp_path, "e108")), samples={"toy": items},
                      source_dir=run4.dir, trace_dir=run5.dir, **dirs)
    assert status == "completed"
    return items, run4, run5, run8, config8, dirs, provider


def test_e108_selects_half_wrong_proposals_with_an_alone_trace_for_every_row(tmp_path):
    items, run4, run5, run8, config8, dirs, _ = _round2_pipeline(tmp_path)
    rows = run8.writer("rows").rows()
    groups = Counter(r["group"] for r in rows)
    assert groups[JEV_WRONG] and groups[JEV_RIGHT] == groups[JEV_WRONG] + groups[MATCHED_WRONG]
    assert groups[MATCHED_WRONG] == round(groups[JEV_WRONG] * config8.matched_wrong_per_wrong)
    traces = [t for t in run5.writer("traces").rows() if t["status"] == "ok"]
    solved_alone = {(t["sample"], t["item_id"]) for t in traces
                    if t["condition"] == "alone" and t["source"] == "reviewer" and formats.correct(t) and formats.usable(t)}
    assert all((r["sample"], r["item_id"]) in solved_alone for r in rows)
    for r in rows:
        assert (r["proposed_label"] == r["label"]) == (r["group"] == JEV_RIGHT)
        assert r["format"] == "review"
    # Items stay in one group, and matched_wrong items come from the mix's alone group (Jev right).
    assert len({(r["sample"], r["item_id"]) for r in rows}) == len(rows)
    mix_group = {(r["sample"], r["item_id"]): r["group"] for r in run4.writer("mix").rows()}
    assert {mix_group[(r["sample"], r["item_id"])] for r in rows if r["group"] == MATCHED_WRONG} == {"alone"}


def test_e108_samples_only_rows_without_a_correct_review_trace(tmp_path):
    items, run4, run5, run8, config8, dirs, provider = _round2_pipeline(tmp_path)
    new = [t for t in run8.writer("traces").rows() if t["status"] == "ok"]
    assert new and all(t["sample_index"] >= SAMPLE_OFFSET for t in new)
    old = [t for t in run5.writer("traces").rows() if t["status"] == "ok"]
    had = {(t["sample"], t["item_id"], t["condition"]) for t in old
           if t["format"] == "review" and t["source"] == "reviewer" and formats.correct(t) and formats.usable(t)}
    assert not any((t["sample"], t["item_id"], t["condition"]) in had for t in new)
    per_row = Counter((t["sample"], t["item_id"], t["condition"]) for t in new)
    group = {(t["sample"], t["item_id"], t["condition"]): t["group"] for t in new}
    for key, n in per_row.items():
        assert n <= (config8.deference_samples if group[key] == JEV_WRONG else config8.review_samples)
    # Resuming makes no new calls.
    calls = provider.calls
    assert run_e108(config8, run8, _fake("reviewer", provider, _guard(tmp_path, "e108b")), samples={"toy": items},
                    source_dir=run4.dir, trace_dir=run5.dir, **dirs) == "completed"
    assert provider.calls == calls


def test_needs_review_sample_counts_only_new_tries():
    config = E108Config(source_run="a", trace_run="b", samples=["toy"], deference_samples=3, review_samples=1)
    wrong = {"group": JEV_WRONG}
    t = lambda k, ok: {"source": "reviewer", "status": "ok", "sample_index": k, "parse_ok": True, "reasoning": "r",
                       "finish_reason": "stop", "final_label": "a" if ok else "b", "label": "a", "consistent": True}
    assert needs_review_sample(wrong, [t(0, False)] * 4, config)
    assert needs_review_sample(wrong, [t(SAMPLE_OFFSET, False), t(SAMPLE_OFFSET + 1, False)], config)
    assert not needs_review_sample(wrong, [t(SAMPLE_OFFSET + k, False) for k in range(3)], config)
    assert not needs_review_sample(wrong, [t(0, True)], config)
    assert not needs_review_sample({"group": JEV_RIGHT}, [t(SAMPLE_OFFSET, False)], config)


def test_e108_arms_share_rows_and_differ_only_in_the_target_reasoning(tmp_path):
    items, run4, run5, run8, config8, dirs, _ = _round2_pipeline(tmp_path)
    folder = dirs["data_dir"] / "toy_r2"
    load = lambda name: [json.loads(x) for x in (folder / name).read_text().splitlines()]
    consist = load("consist_review.train.jsonl") + load("consist_review.val.jsonl")
    balanced = load("balanced_review.train.jsonl") + load("balanced_review.val.jsonl")
    assert consist and [r["prompt"] for r in consist] == [r["prompt"] for r in balanced]
    assert [r["completion"][0]["content"] for r in consist] == [r["completion"][0]["content"] for r in balanced]
    traces = [t for t in run5.writer("traces").rows() + run8.writer("traces").rows() if t["status"] == "ok"]
    by = {(t["sample"], t["item_id"], t["condition"], t["sample_index"]): t for t in traces if t["source"] == "reviewer"}
    for c, b in zip(consist, balanced):
        m = c["meta"]
        assert m["format"] == "review" and m["proposed_label"] is not None
        answer = json.loads(c["completion"][0]["content"])
        assert answer["final_label"] == m["label"]
        assert answer["verdict"] == ("accept" if m["proposed_label"] == m["label"] else "override")
        # Arm 1's reasoning is an alone trace; arm 2's is a review trace with this proposal shown.
        assert c["completion"][0]["thinking"] == by[(m["sample"], m["item_id"], "alone", m["trace_sample_index"])]["reasoning"]
        cond = {JEV_WRONG: "jev_label", JEV_RIGHT: "jev_label", MATCHED_WRONG: "matched_label"}[m["group"]]
        bm = b["meta"]
        assert b["completion"][0]["thinking"] == by[(m["sample"], m["item_id"], cond, bm["trace_sample_index"])]["reasoning"]
        assert c["meta"]["trace_source"] == b["meta"]["trace_source"] == "reviewer"

    grpo = load("grpo_prompts.train.jsonl") + load("grpo_prompts.val.jsonl")
    mix_review = [r for r in run4.writer("mix").rows() if r["format"] == "review"]
    assert len(grpo) == len(mix_review) + sum(r["group"] == MATCHED_WRONG for r in run8.writer("rows").rows())
    assert all(g["prompt"][-1]["role"] == "user" and g["meta"]["proposed_label"] for g in grpo)

    for arm in ("consist_review", "balanced_review"):
        stats = check(TrainConfig(name=f"t-{arm}", method="sft", dataset="toy_r2", arm=arm),
                      data_dir=dirs["data_dir"], manifest_dir=dirs["manifest_dir"])
        assert stats["train"]["records"] == len(load(f"{arm}.train.jsonl"))
    with pytest.raises(ValueError, match="different data"):
        export_round2("toy_r2", [], [], {}, {}, source={}, **dirs)


def test_committed_e108_config_loads():
    from sysone.registry import CONFIG_DIR

    config = E108Config.load(CONFIG_DIR / "experiments" / "e108.yaml")
    assert config.trace_run.startswith("e105") and config.source_run.startswith("e104")
    for arm in ("consist_review", "balanced_review"):
        c = TrainConfig.load(CONFIG_DIR / "posttrain" / f"{arm}.yaml")
        assert c.dataset == config.dataset and c.init_from is None and c.arm == arm


# --- round-2 groups and criteria -----------------------------------------------------------------

def _alone(reviewer, item, k, right):
    return {"reviewer": reviewer, "sample": "s", "item_id": item, "condition": "alone", "sample_index": k,
            "label": "a", "final_label": "a" if right else "b"}


def test_base_groups_split_by_first_try_and_full_record():
    rows = [_alone("base", "first", k, k == 0) for k in range(5)]
    rows += [_alone("base", "later", k, k == 3) for k in range(5)]
    rows += [_alone("base", "never", k, False) for k in range(5)]
    rows += [_alone("base", "short", k, False) for k in range(4)]  # a parse failure: not counted as never
    rows += [_alone("other", "never", 0, True)]
    g = base_groups(rows, "base")
    assert g == {("s", "first"): GROUP_A, ("s", "later"): GROUP_B, ("s", "never"): GROUP_C, ("s", "short"): GROUP_B}


def _flags(override, accept):
    return {"override_jev": override, "accept_right_jev": accept, "cascade": {}, "override_upstream": {},
            "accept_right_upstream": {}}


def test_round2_criteria_score_solvable_items_and_ignore_group_c():
    groups = {**{("s", f"a{i}"): GROUP_A for i in range(40)}, **{("s", f"c{i}"): GROUP_C for i in range(40)}}
    accept = {("s", f"r{i}"): True for i in range(50)}
    base = _flags({**{("s", f"a{i}"): i < 20 for i in range(40)}, **{("s", f"c{i}"): False for i in range(40)}}, accept)
    # Overrides every group A mistake; gets worse on group C, which isn't scored.
    better = _flags({**{("s", f"a{i}"): True for i in range(40)}, **{("s", f"c{i}"): False for i in range(40)}}, accept)
    out = evaluate_round2(better, base, base, groups, {}, ["s"], [], {"max_accept_drop": 0.02})
    assert out["R1_catches_solvable"]["pass"] and out["R2_gives_in_less"]["pass"] and out["R3_beats_control"]["pass"]
    assert out["R4_every_pool"]["pass"] and out["works"]
    assert out["R1_catches_solvable"]["override_change"]["n"] == 40
    # The same gain bought by rejecting Jev's right labels fails R1, so the model doesn't "work".
    contrarian = _flags(better["override_jev"], {k: i >= 10 for i, k in enumerate(accept)})
    out = evaluate_round2(contrarian, base, base, groups, {}, ["s"], [], {"max_accept_drop": 0.02})
    assert not out["R1_catches_solvable"]["pass"] and not out["works"]
