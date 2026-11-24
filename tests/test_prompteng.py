"""Prompt-engineering track (E.115–E.117, research_docs/16): the standard prompt stays frozen, arms
are reviewer-only and WildGuard-only, E.116 picks its items and an arm by the fixed rule."""

import hashlib
import json

import pytest

from sysone.analysis.e116 import select_arm, summarize
from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.items import Item
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import CIVIL_TOXICITY, TASKS, WILDGUARD_POLICY, WILDGUARD_PROMPT, WILDGUARD_RESPONSE
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider
from sysone.experiments.e116 import LET_THROUGH, OVER_FLAGGED, RIGHT, E116Config, pool_counts, run_e116, select
from sysone.prompteng.arms import (
    ARMS,
    CLARIFY,
    CLARIFY_GROUNDED,
    EXAMPLES,
    GROUNDED,
    GROUNDED_PREFIX,
    STANDARD,
    ArmTexts,
    Example,
    arm_kwargs,
    load_texts,
)
from sysone.runs import JsonlWriter, Run

# sha256 of the rendered messages for the fixed text "ITEM TEXT", computed before the prompt-engineering
# options existed (2026-11-23). A change here changes the frozen prompt of E.101–E.114.
STANDARD_HASHES = {
    ("civil_comments_toxicity", "alone"): "2e33476a581f7db3ab2f1d8acc7b98af2cb8e7d087569b072062f89d6a465aa5",
    ("civil_comments_toxicity", "review"): "2c5da774327f020840b91f5585539734a79fdf166630e25b1f2ff799e35cf770",
    ("civil_comments_toxicity", "blind_first"): "37fd2c2d40d5f04379eab5cd977e07836dee4ca0ca6ece0382e0619a47a49281",
    ("wildguard_prompt_harm", "alone"): "de604234db06a9036abca215dcbb155bca0bc8ba709cf39983b7fa1a81de41f2",
    ("wildguard_prompt_harm", "review"): "a5947d71b4ce3c37add917ecb68b115acaa20ddffa6e9decdfb15b54f1cf45dd",
    ("wildguard_prompt_harm", "blind_first"): "1b638f211ace12e4b9c4ddaf5d26b7057b9c8e0e8a4797c47aec98736dd6e82b",
    ("wildguard_response_harm", "alone"): "8906ba4fae13e2163b99598965893bb947752123c66f454bba52b72c8d3fd7cc",
    ("wildguard_response_harm", "review"): "8a1d6f0edd4f1a0edb2c68ec17a49f41adfec6d10c9a57d7cd1c108565e98403",
    ("wildguard_response_harm", "blind_first"): "f4b418d99c820f1aaf0ab13ecf63a225683bbb26653035a2e322c51c001091c3",
}


def _client(tmp_path):
    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec("m", "bedrock", "m", Price(1.0, 1.0)), FakeChatProvider(), guard)


def _h(req) -> str:
    return hashlib.sha256(json.dumps([[m.role, m.content] for m in req.messages]).encode()).hexdigest()


def test_the_standard_prompt_is_byte_identical(tmp_path):
    client = _client(tmp_path)
    for name, task in TASKS.items():
        r = Reviewer(client, task)
        assert _h(r.build_request("ITEM TEXT")) == STANDARD_HASHES[(name, "alone")]
        assert _h(r.build_request("ITEM TEXT", task.labels[0])) == STANDARD_HASHES[(name, "review")]
        assert _h(r.build_request("ITEM TEXT", task.labels[0], own_answer=task.labels[1])) == STANDARD_HASHES[(name, "blind_first")]
        # The standard arm passes no options at all.
        if name != CIVIL_TOXICITY.name:
            assert arm_kwargs(STANDARD, name, ArmTexts({}, {})) == {}


def test_addendum_and_examples_change_only_the_reviewer_prompt(tmp_path):
    r = Reviewer(_client(tmp_path), WILDGUARD_PROMPT)
    std = r.build_request("ITEM TEXT", "harmful").messages[0].content
    add = r.build_request("ITEM TEXT", "harmful", policy_addendum="EXTRA RULE", examples="EX 1").messages[0].content
    assert WILDGUARD_POLICY + "\n\nEXTRA RULE\n\n## Worked examples\nEX 1\n\n## User prompt" in add
    assert add.replace("\n\nEXTRA RULE", "").replace("\n\n## Worked examples\nEX 1", "") == std
    # Jev's policy text is the module constant, untouched by any arm.
    assert WILDGUARD_PROMPT.policy == WILDGUARD_POLICY and "EXTRA RULE" not in WILDGUARD_POLICY


def test_arm_kwargs():
    texts = ArmTexts({WILDGUARD_PROMPT.name: "C1", WILDGUARD_RESPONSE.name: "C2"}, {})
    assert arm_kwargs(CLARIFY, WILDGUARD_RESPONSE.name, texts) == {"policy_addendum": "C2"}
    assert arm_kwargs(GROUNDED, WILDGUARD_PROMPT.name, texts) == {"task_prefix": GROUNDED_PREFIX[WILDGUARD_PROMPT.name]}
    assert arm_kwargs(CLARIFY_GROUNDED, WILDGUARD_PROMPT.name, texts) == {
        "policy_addendum": "C1", "task_prefix": GROUNDED_PREFIX[WILDGUARD_PROMPT.name]}
    with pytest.raises(ValueError, match="WildGuard only"):
        arm_kwargs(CLARIFY, CIVIL_TOXICITY.name, texts)
    with pytest.raises(ValueError):
        arm_kwargs("shout", WILDGUARD_PROMPT.name, texts)


def test_placeholder_texts_are_refused(tmp_path):
    for task in (WILDGUARD_PROMPT.name, WILDGUARD_RESPONSE.name):
        (tmp_path / f"clarify_{task}.md").write_text("TODO: E.115 writes this")
    with pytest.raises(ValueError, match="placeholder"):
        load_texts([STANDARD, CLARIFY], tmp_path)
    (tmp_path / "examples.yaml").write_text("# TODO\n")
    with pytest.raises(ValueError, match="placeholder"):
        load_texts([STANDARD, EXAMPLES], tmp_path)
    assert load_texts([STANDARD, GROUNDED], tmp_path) == ArmTexts({}, {})  # needs no texts


def test_the_repo_clarify_texts_load():
    texts = load_texts([STANDARD, CLARIFY])
    assert set(texts.clarify) == {WILDGUARD_PROMPT.name, WILDGUARD_RESPONSE.name}


def test_examples_must_be_balanced_approved_development_items(tmp_path):
    import yaml
    rows = [{"sample": "wildguard_prompt_train_v1", "item_id": f"x{i}", "label": ("harmful", "unharmful")[i % 2], "why": "w"}
            for i in range(6)]
    write = lambda d: (tmp_path / "examples.yaml").write_text(yaml.safe_dump(d))
    write({"status": "approved", WILDGUARD_PROMPT.name: rows, WILDGUARD_RESPONSE.name: rows[:5]})
    with pytest.raises(ValueError, match="need 6"):
        load_texts([STANDARD, EXAMPLES], tmp_path)
    test_rows = [{**r, "sample": "wildguard_prompt_full_v1"} for r in rows]
    write({"status": "approved", WILDGUARD_PROMPT.name: test_rows, WILDGUARD_RESPONSE.name: rows})
    with pytest.raises(ValueError, match="training pools"):
        load_texts([STANDARD, EXAMPLES], tmp_path)
    write({"status": "draft", WILDGUARD_PROMPT.name: rows, WILDGUARD_RESPONSE.name: rows})
    with pytest.raises(ValueError, match="approved"):
        load_texts([STANDARD, EXAMPLES], tmp_path)
    both = [{**rows[0], "text": "t"}, *rows[1:]]
    write({"status": "approved", WILDGUARD_PROMPT.name: both, WILDGUARD_RESPONSE.name: rows})
    with pytest.raises(ValueError, match="either text"):
        load_texts([STANDARD, EXAMPLES], tmp_path)


def test_handwritten_examples_render_and_are_checked_against_test_items(tmp_path):
    import yaml
    hand = [{"text": f"a harmless question about gardening number {i} with enough words to compare", "label":
             ("harmful", "unharmful")[i % 2], "why": "because"} for i in range(6)]
    (tmp_path / "examples.yaml").write_text(yaml.safe_dump(
        {"status": "approved", WILDGUARD_PROMPT.name: hand, WILDGUARD_RESPONSE.name: hand}))
    texts = load_texts([STANDARD, EXAMPLES], tmp_path)
    kw = arm_kwargs(EXAMPLES, WILDGUARD_PROMPT.name, texts, {})
    assert kw["examples"].startswith("Example 1:\na harmless question") and 'Decision: "harmful". Why: because' in kw["examples"]
    from sysone.prompteng.arms import check_test_overlap, example_keys
    assert example_keys(texts) == set()
    check_test_overlap(texts, [Item("t1", "wg", WILDGUARD_PROMPT.name, "something else entirely about cars", "harmful", {})])
    copy = Item("t2", "wg", WILDGUARD_PROMPT.name, "Intro. " + hand[3]["text"] + " More.", "harmful", {})
    with pytest.raises(ValueError, match="test item t2"):
        check_test_overlap(texts, [copy])


def test_the_repo_draft_examples_are_valid_but_refused_until_approved(tmp_path):
    import yaml
    from sysone.prompteng.arms import TEXT_DIR
    raw = yaml.safe_load((TEXT_DIR / "examples.yaml").read_text())
    if raw.get("status") != "approved":
        with pytest.raises(ValueError, match="approved"):
            load_texts([STANDARD, EXAMPLES])
    (tmp_path / "examples.yaml").write_text(yaml.safe_dump({**raw, "status": "approved"}))
    texts = load_texts([STANDARD, EXAMPLES], tmp_path)
    assert all(len(v) == 6 for v in texts.examples.values())


def _mix(sample, task, n_wrong=6, n_right=10):
    rows = []
    for i in range(n_wrong):
        label = ("unharmful", "harmful")[i % 3 == 0]
        rows.append({"sample": sample, "item_id": f"w{i}", "task": task, "format": "review", "group": "jev_wrong",
                     "label": label, "jev_label": "harmful" if label == "unharmful" else "unharmful", "jev_confidence": 0.6})
    for i in range(n_right):
        label = ("harmful", "unharmful")[i % 2]
        rows.append({"sample": sample, "item_id": f"r{i}", "task": task, "format": "review", "group": "jev_right",
                     "label": label, "jev_label": label, "jev_confidence": 0.9})
    rows.append({"sample": sample, "item_id": "a0", "task": task, "format": "alone", "group": "alone",
                 "label": "harmful", "jev_label": "harmful", "jev_confidence": 1.0})
    return rows


SAMPLES = ["wildguard_prompt_train_v1", "wildguard_response_train_v1"]


def _config(**kw):
    return E116Config(source_run="e104", samples=SAMPLES, right_per_pool=4, bootstrap=200,
                      selection={"max_accept_right_drop": 0.02, "max_let_through_override_drop": 0.03,
                                 "min_parse_rate": 0.95, "min_gain": 0.05}, **kw)


def test_select_takes_every_mistake_and_a_hash_sample_of_right_labels():
    mix = _mix(SAMPLES[0], WILDGUARD_PROMPT.name) + _mix(SAMPLES[1], WILDGUARD_RESPONSE.name)
    picked = select(mix, _config(), exclude={(SAMPLES[0], "w1")})
    p0 = [p for p in picked if p["sample"] == SAMPLES[0]]
    assert sum(p["cell"] != RIGHT for p in p0) == 5 and sum(p["cell"] == RIGHT for p in p0) == 4
    assert "w1" not in {p["item_id"] for p in p0} and "a0" not in {p["item_id"] for p in picked}
    assert {p["cell"] for p in picked if p["label"] == "harmful" and p["cell"] != RIGHT} == {LET_THROUGH}
    assert {p["cell"] for p in picked if p["label"] == "unharmful" and p["cell"] != RIGHT} == {OVER_FLAGGED}
    assert picked == select(mix, _config(), exclude={(SAMPLES[0], "w1")})  # deterministic
    with pytest.raises(ValueError, match="training pools"):
        _config().__class__(source_run="x", samples=["wildguard_prompt_full_v1"]).check()
    with pytest.raises(ValueError, match="civil|WildGuard"):
        _config().__class__(source_run="x", samples=["civil_toxicity_train_v1"]).check()


def test_pool_counts_weights_cells_back_to_the_pool():
    mix = _mix(SAMPLES[0], WILDGUARD_PROMPT.name)
    items = [Item(f"i{i}", "wg", WILDGUARD_PROMPT.name, "t", ("harmful", "unharmful")[i % 2], {}) for i in range(60)]
    counts = pool_counts(mix, {SAMPLES[0]: items}, {SAMPLES[0]: 0.1})
    assert counts[SAMPLES[0]] == {OVER_FLAGGED: 4, LET_THROUGH: 2, "right_harmful": 28, "right_unharmful": 26}


def _rows(arm, *, fix_over_flagged=0, break_right=0, parse=True):
    out = []
    for s in SAMPLES:
        for i in range(40):
            cell = (OVER_FLAGGED, LET_THROUGH, RIGHT, RIGHT)[i % 4]
            label = {OVER_FLAGGED: "unharmful", LET_THROUGH: "harmful", RIGHT: ("harmful", "unharmful")[i % 8 // 4]}[cell]
            jev = label if cell == RIGHT else ("harmful" if label == "unharmful" else "unharmful")
            final = jev
            if cell == OVER_FLAGGED and i // 4 < fix_over_flagged:
                final = label
            if cell == RIGHT and i // 4 < break_right:
                final = "harmful" if jev == "unharmful" else "unharmful"
            out.append({"status": "ok", "arm": arm, "sample": s, "item_id": f"i{i}", "cell": cell, "label": label,
                        "proposed_label": jev, "final_label": final, "parse_ok": parse, "output_tokens": 100})
    return out


def test_selection_rule_picks_the_admissible_arm_with_the_largest_gain():
    counts = {s: {OVER_FLAGGED: 50, LET_THROUGH: 50, "right_harmful": 450, "right_unharmful": 450} for s in SAMPLES}
    rows = (_rows(STANDARD) + _rows(CLARIFY, fix_over_flagged=8) + _rows(GROUNDED, fix_over_flagged=10, break_right=4))
    summary = summarize(rows, counts, reps=300)
    pooled = summary["pooled"]
    assert pooled[STANDARD]["override_wrong"]["rate"] == 0.0
    assert pooled[CLARIFY]["vs_standard"]["override_over_flagged"]["diff"] == 0.8
    assert pooled[CLARIFY]["vs_standard"]["override_let_through"]["diff"] == 0.0
    assert pooled[CLARIFY]["vs_standard"]["cascade_accuracy"]["diff"] > 0
    sel = select_arm(summary, _config().selection)
    assert sel["verdicts"][GROUNDED]["checks"]["accept_right"] is False  # it breaks Jev's right labels
    assert sel["chosen"] == CLARIFY
    # Too small a gain: nothing goes to E.117.
    small = summarize(_rows(STANDARD) + _rows(CLARIFY, fix_over_flagged=1), counts, reps=300)
    assert select_arm(small, _config().selection)["chosen"] is None


def test_dry_run_end_to_end(tmp_path):
    mix = _mix(SAMPLES[0], WILDGUARD_PROMPT.name) + _mix(SAMPLES[1], WILDGUARD_RESPONSE.name)
    items = {(r["sample"], r["item_id"]): Item(r["item_id"], "wg", r["task"], f"text {r['item_id']}", r["label"], {})
             for r in mix}
    config = _config(arms=[STANDARD, GROUNDED], workers=2)
    picked = select(mix, config, set())
    client = _client(tmp_path)
    run = Run("e116-test", tmp_path)
    status = run_e116(config, run, client, items=items, picked=picked, texts=ArmTexts({}, {}),
                      counts={s: {OVER_FLAGGED: 4, LET_THROUGH: 2, "right_harmful": 50, "right_unharmful": 50} for s in SAMPLES})
    assert status == "completed"
    rows = JsonlWriter(run.dir / "reviews.jsonl").rows()
    assert len(rows) == len(picked) * 2 and {r["arm"] for r in rows} == {STANDARD, GROUNDED}
    # Resuming sends nothing new.
    calls = client.provider.calls
    run_e116(config, run, client, items=items, picked=picked, texts=ArmTexts({}, {}), counts={})
    assert client.provider.calls == calls
    assert ARMS[0] == STANDARD
