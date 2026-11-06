import json
import math
from collections import Counter

import pytest

from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.proposals import ALONE, JEV_LABEL, Proposal
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS, WILDGUARD_PROMPT
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider
from sysone.experiments.e101 import review_work
from sysone.experiments.e102 import CORRECT_LABEL
from sysone.interp import steering as st
from sysone.posttrain.mix import split_of
from sysone.runs import Run
from test_e102 import make_items

LABELS = WILDGUARD_PROMPT.labels  # ("harmful", "unharmful")
H, U = LABELS


def _trace(item, k, condition, final, *, source="reviewer", group="jev_wrong", label=H, proposed=U, **kw):
    return {"source": source, "sample": "pool", "item_id": item, "task": WILDGUARD_PROMPT.name, "group": group,
            "condition": condition, "label": label, "proposed_label": None if condition == ALONE else proposed,
            "final_label": final, "verdict": None if condition == ALONE else ("accept" if final == proposed else "override"),
            "reasoning": f"thinking about {item}", "parse_ok": True, "status": "ok", "sample_index": k, **kw}


def _toy_traces():
    return [
        # a: solved alone; one trace gave in, one held -> kept, both traces
        _trace("a", 0, ALONE, H), _trace("a", 0, JEV_LABEL, U), _trace("a", 1, JEV_LABEL, H),
        # b: solved alone, only gave in -> kept
        _trace("b", 0, ALONE, H), _trace("b", 0, JEV_LABEL, U), _trace("b", 1, JEV_LABEL, U),
        # c: solved alone, always held -> left out (no give-in trace)
        _trace("c", 0, ALONE, H), _trace("c", 0, JEV_LABEL, H), _trace("c", 1, JEV_LABEL, H),
        # d: never solved alone -> left out
        _trace("d", 0, ALONE, U), _trace("d", 0, JEV_LABEL, U),
        # e: only the teacher solved it alone; f: a Jev-right item -> neither counts
        _trace("e", 0, ALONE, H, source="teacher"), _trace("e", 0, JEV_LABEL, U),
        _trace("f", 0, ALONE, H, group="jev_right"), _trace("f", 0, JEV_LABEL, H, group="jev_right", proposed=H),
        # a parse failure is not a trace
        _trace("a", 2, JEV_LABEL, None, parse_ok=False),
    ]


# --- selection and splits ----------------------------------------------------------------------

def test_selection_keeps_every_review_trace_of_items_solved_alone_that_gave_in():
    rows, dropped = st.select_traces(_toy_traces(), heldout_fraction=0.2, seed=110)
    assert [(r["item_id"], r["sample_index"], r["gave_in"]) for r in rows] == [
        ("a", 0, True), ("a", 1, False), ("b", 0, True), ("b", 1, True)]
    assert dropped == Counter({"pool/never_gave_in": 1, "pool/not_solved_alone": 2})
    assert {r["trace_id"] for r in rows} == {"pool/a/0", "pool/a/1", "pool/b/0", "pool/b/1"}
    # The split is per item, so an item's traces are on one side.
    for item in "ab":
        assert len({r["split"] for r in rows if r["item_id"] == item}) == 1


def test_hash_split_is_deterministic_and_about_the_fraction():
    ids = [f"x-{i:05d}" for i in range(4000)]
    held = [split_of("pool", i, 0.2, 110) == "val" for i in ids]
    assert 0.18 < sum(held) / len(held) < 0.22
    assert held == [split_of("pool", i, 0.2, 110) == "val" for i in ids]
    assert held != [split_of("pool", i, 0.2, 104) == "val" for i in ids]  # E.104's seed gives another split


def test_alpha_items_are_heldout_wrong_items_and_a_seeded_jev_right_sample():
    selection = [{"sample": "pool", "item_id": i, "task": "t", "label": H, "proposed_label": U,
                  "split": "heldout" if i in "ab" else "fit"} for i in "abc"]
    mix = [{"sample": "pool", "item_id": f"r{k}", "task": "t", "label": H, "strata": {"s": "x"}, "group": "jev_right"}
           for k in range(10)] + [{"sample": "other", "item_id": "z", "task": "t", "label": H, "strata": {},
                                   "group": "jev_right"}]
    rows = st.alpha_items(selection, mix, n_right=4, seed=110)
    wrong = [r for r in rows if r["set"] == st.WRONG_SET]
    right = [r for r in rows if r["set"] == st.JEV_RIGHT_SET]
    assert [r["item_id"] for r in wrong] == ["a", "b"] and {r["condition"] for r in wrong} == {JEV_LABEL}
    assert len(right) == 4 and {r["sample"] for r in right} == {"pool"}  # only the selection's pools
    assert all(r["condition"] == CORRECT_LABEL and r["proposed_label"] == r["label"] for r in right)
    assert rows == st.alpha_items(selection, mix, n_right=4, seed=110)


def test_eval_items_are_group_a_jev_wrong_and_the_jev_right_subsample():
    plan = [{"sample": "pool", "item_id": i, "label": H, "strata": {}, "jev_label": jl, "jev_correct": jl == H,
             "in_subsample": sub} for i, jl, sub in
            [("a", U, True), ("b", U, False), ("c", H, True), ("d", H, False), ("e", U, True)]]
    groups = {("pool", "a"): "A", ("pool", "b"): "A", ("pool", "e"): "C", ("pool", "c"): "A"}
    rows = st.eval_items(plan, groups)
    assert [(r["item_id"], r["set"], r["condition"], r["proposed_label"]) for r in rows] == [
        ("a", "jev_wrong", JEV_LABEL, U), ("b", "jev_wrong", JEV_LABEL, U), ("c", "jev_right", CORRECT_LABEL, H)]


def _runs(tmp_path, traces):
    runs = tmp_path / "runs"
    mix = [{"sample": "pool", "item_id": i, "task": WILDGUARD_PROMPT.name, "label": H, "strata": {}, "group": "jev_right"}
           for i in ("r1", "r2")]
    plan = [{"sample": "test", "item_id": "t1", "label": H, "strata": {}, "jev_label": U, "jev_correct": False,
             "in_subsample": True}]
    evals = [{"reviewer": "base_local", "status": "ok", "parse_ok": True, "sample": "test", "item_id": "t1",
              "condition": ALONE, "sample_index": k, "label": H, "final_label": H} for k in range(5)]
    for name, fname, rows in (("e105", "traces", traces), ("e104", "mix", mix), ("e102", "plan", plan),
                              ("e106", "reviews", evals)):
        (runs / name).mkdir(parents=True)
        (runs / name / f"{fname}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return st.E110Config(e104_run="e104", e105_run="e105", e102_run="e102", e106_run="e106"), runs


def test_build_writes_frozen_lists_and_refuses_a_changed_rebuild(tmp_path):
    config, runs = _runs(tmp_path, _toy_traces())
    run = Run("e110", tmp_path)
    counts = st.build(config, run, runs)
    assert counts["selection"]["total"]["items"] == 2
    assert counts["selection"]["total"]["items_with_both"] == 1  # only a has a held trace; b is never paired
    assert counts["selection"]["total"]["traces_gave_in"] == 3 and counts["selection"]["total"]["traces_held"] == 1
    assert counts["test_items"] == {"jev_wrong/test": 1}
    assert st.E110Config(**run.manifest()["config"]) == config
    assert st.build(config, run, runs) == counts  # a rebuild with the same inputs is fine
    config, runs = _runs(tmp_path / "changed", _toy_traces() + [_trace("c", 2, JEV_LABEL, U)])
    with pytest.raises(ValueError, match="different rows"):
        st.build(config, run, runs)


# --- statistics ---------------------------------------------------------------------------------

def test_auc_counts_ties_as_half():
    assert st.auc([3, 4], [1, 2]) == 1.0
    assert st.auc([1, 2], [3, 4]) == 0.0
    assert st.auc([1, 1], [1, 1]) == 0.5
    assert st.auc([2, 3], [1, 2]) == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert st.auc([], [1]) is None


def test_the_direction_is_paired_within_items_and_one_sided_items_contribute_nothing():
    vecs = [[3.0, 1.0], [1.0, 1.0],      # item a: give-in, held -> diff [2, 0]
            [5.0, 2.0], [7.0, 2.0], [2.0, 2.0],  # item b: two give-ins (mean [6, 2]), held -> diff [4, 0]
            [100.0, 100.0], [90.0, 90.0]]  # item c: only give-ins -> nothing
    gave = [True, False, True, True, False, True, True]
    items = ["a", "a", "b", "b", "b", "c", "c"]
    d, n = st.paired_diff(vecs, gave, items)
    assert (d, n) == ([3.0, 0.0], 2)
    # Without c the direction is the same: a one-sided item can't move it.
    assert st.paired_diff(vecs[:5], gave[:5], items[:5]) == (d, n)
    with pytest.raises(ValueError, match="both"):
        st.paired_diff(vecs[5:], gave[5:], items[5:])
    assert st.project([[1.0, 0.0], [3.0, 2.0]], d) == [1.0, 3.0]  # along the unit direction


def test_gaps_are_paired_within_items_and_the_auc_uses_every_trace():
    scores = [3.0, 1.0, 6.0, 2.0, 50.0]
    gave = [True, False, True, False, True]
    items = ["a", "a", "b", "b", "c"]  # c has only a give-in trace
    g = st.paired_gap(scores, gave, items)
    assert g["gap"] == 3.0 and g["n_items_paired"] == 2  # per-item gaps 2 and 4
    assert g["std_gap"] == pytest.approx(3.0 / math.sqrt(2), abs=1e-4)  # mean over the per-item gaps' sd
    assert g["auc"] == st.auc([3.0, 6.0, 50.0], [1.0, 2.0]) == 1.0 and (g["n_gave_in"], g["n_held"]) == (3, 2)
    assert st.paired_gap([1.0, 2.0], [True, True], ["a", "b"])["gap"] is None


def test_best_layer_and_its_neighbours():
    assert st.best_layer([0.6, 0.8, 0.8, None]) == 1
    assert st.steer_layers(0, 24, 1) == [0, 1] and st.steer_layers(12, 24, 1) == [11, 12, 13]
    assert st.steer_layers(23, 24, 1) == [22, 23]


def test_alpha_rule_takes_the_smallest_alpha_with_the_largest_admissible_drop():
    table = {0.0: {"give_in": 0.70, "accept": 0.97}, 0.5: {"give_in": 0.60, "accept": 0.97},
             1.0: {"give_in": 0.40, "accept": 0.96}, 2.0: {"give_in": 0.40, "accept": 0.955},
             4.0: {"give_in": 0.10, "accept": 0.80}}
    c = st.choose_alpha(table, max_accept_drop=0.02)
    assert c["alpha"] == 1.0 and c["admissible"]  # 4 drops more but costs 17 points of accepting; 2 ties 1
    table[1.0]["accept"] = 0.94
    assert st.choose_alpha(table, max_accept_drop=0.02)["alpha"] == 2.0
    exact = {0.0: {"give_in": 0.7, "accept": 0.97}, 0.5: {"give_in": 0.6, "accept": 0.95}}
    assert st.choose_alpha(exact, max_accept_drop=0.02)["admissible"]  # exactly 2 points is within the limit
    bad = {0.0: {"give_in": 0.7, "accept": 0.97}, 0.5: {"give_in": 0.6, "accept": 0.9}, 1.0: {"give_in": 0.5, "accept": 0.8}}
    assert st.choose_alpha(bad, max_accept_drop=0.02) == {"alpha": 0.5, "admissible": False,
                                                          "candidates": st.choose_alpha(bad, max_accept_drop=0.02)["candidates"]}


def test_every_prompt_gets_the_fixed_date():
    class Tok:  # renders like gpt-oss's template: the date comes from strftime_now
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, reasoning_effort, strftime_now):
            text = f"Current date: {strftime_now('%Y-%m-%d')}|" + "".join(f"<{m['role']}>{m.get('thinking', '')}"
                                                                         f"{m['content']}" for m in messages)
            return text + ("<assistant" if add_generation_prompt else "")

    config = st.E110Config(chat_date="2026-01-02")
    assert st.fixed_date("2026-01-02")("%d.%m.%Y") == "02.01.2026"
    assert st._prompt(Tok(), WILDGUARD_PROMPT.name, "x", None, config).startswith("Current date: 2026-01-02|")
    head, tail = st._render(Tok(), [{"role": "user", "content": "q"}], [{"role": "assistant", "content": "a"}], config)
    assert (head, tail) == ("Current date: 2026-01-02|<user>q<assistant", ">a")
    with pytest.raises(ValueError, match="prefix"):
        st._render(Tok(), [{"role": "user", "content": "q"}], [{"role": "tool", "content": "a"}], config)


# --- harmony positions and replies --------------------------------------------------------------

def _tokens(pieces):
    """Fake tokenization: each piece is a token; offsets as a fast tokenizer reports them."""
    offsets, at = [], 0
    for p in pieces:
        offsets.append((at, at + len(p)))
        at += len(p)
    return "".join(pieces), offsets


def test_positions_find_the_last_prompt_token_the_reasoning_and_the_token_before_the_answer():
    head = ["<|start|>", "user", "<|message|>", "Is it ", "harmful?", "<|end|>", "<|start|>", "assistant"]
    tail = ["<|channel|>", "analysis", "<|message|>", "We ", "need ", "to ", "decide.", "<|end|>", "<|start|>",
            "assistant", "<|channel|>", "final", "<|message|>", '{"verdict": ', '"accept"}', "<|return|>"]
    full, offsets = _tokens(head + tail)
    p = st.positions(full, len("".join(head)), offsets)
    tokens = head + tail
    assert tokens[p["last_prompt"]] == "assistant" and p["last_prompt"] == len(head) - 1
    assert tokens[slice(*p["reasoning"])] == ["We ", "need ", "to ", "decide."]
    assert tokens[p["pre_answer"]] == "<|message|>" and tokens[p["pre_answer"] - 1] == "final"
    # A token straddling the prompt's end can't be split, so it's refused rather than guessed.
    merged = head[:-1] + ["assistant<|channel|>"] + tail[1:]
    full, offsets = _tokens(merged)
    with pytest.raises(ValueError, match="straddles"):
        st.positions(full, len("".join(head)), offsets)


def test_split_generation_reads_harmony_replies():
    text = '<|channel|>analysis<|message|>Hmm.<|end|><|start|>assistant<|channel|>final<|message|>{"a": 1}<|return|>'
    assert st.split_generation(text) == ("Hmm.", '{"a": 1}', True)
    assert st.split_generation("<|channel|>analysis<|message|>Still thinking") == ("Still thinking", "", False)
    assert st.split_generation('<|channel|>final<|message|>{"a": 1}') == (None, '{"a": 1}', False)


def test_review_rows_have_e106s_schema_and_are_parsed_by_the_reviewers_parser(tmp_path):
    item = make_items(1)[0]
    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    client = GuardedClient(ModelSpec("base_local", "fake", "fake", Price(0.0, 0.0)), FakeChatProvider(malformed_every=0),
                           guard, cache=None)
    work = review_work(Run("ref", tmp_path), {("base_local", "toy"): Reviewer(client, TASKS[item.task])}, experiment="e106")
    reference = work(("base_local", "toy", item, Proposal(item.item_id, JEV_LABEL, U, False, None), 0))
    row = {"sample": "toy", "item_id": item.item_id, "label": item.label, "strata": item.strata,
           "condition": JEV_LABEL, "proposed_label": U, "proposed_correct": False}
    reply = '<|channel|>analysis<|message|>It is fine.<|end|><|start|>assistant<|channel|>final<|message|>' \
            '{"verdict": "accept", "final_label": "unharmful"}<|return|>'
    kw = dict(task=item.task, model_used="ck", input_tokens=10, output_tokens=20, reasoning_tokens=5, seconds=1.0,
              request_key="k", steering={"direction": "steer", "alpha": 1.0, "layers": [11, 12, 13]})
    out = st.review_row("e110", "base_local+steer", row, generated=reply, **kw)
    assert set(reference) - {"error"} <= set(out)
    assert (out["final_label"], out["verdict"], out["consistent"], out["parse_ok"]) == ("unharmful", "accept", True, True)
    assert out["reasoning"] == "It is fine." and out["finish_reason"] == "stop" and out["raw_text"] is None
    cut = st.review_row("e110", "base_local+steer", row, generated="<|channel|>analysis<|message|>and so on", **kw)
    assert not cut["parse_ok"] and cut["finish_reason"] == "length" and cut["raw_text"]


# --- measurement 2 and the report ---------------------------------------------------------------

def test_paired_changes_and_the_causal_rule():
    tests = [{"sample": "s", "item_id": f"w{i}", "condition": JEV_LABEL, "set": "jev_wrong"} for i in range(40)] + \
            [{"sample": "s", "item_id": f"r{i}", "condition": CORRECT_LABEL, "set": "jev_right"} for i in range(40)]

    def rows(reviewer, wrong_accepted, right_accepted):
        out = []
        for t in tests:
            n = int(t["item_id"][1:])
            acc = n < (wrong_accepted if t["set"] == "jev_wrong" else right_accepted)
            out.append({"reviewer": reviewer, "status": "ok", "parse_ok": True, "sample": "s", "item_id": t["item_id"],
                        "condition": t["condition"], "proposed_label": H, "final_label": H if acc else U})
        return out

    reviews = rows("base_local+none", 20, 39) + rows("base_local+steer", 5, 39) + rows("base_local+random", 20, 39)
    m2 = st.measurement_2(reviews, tests, base_reviewer="base_local", vllm_rows=[], max_accept_drop=0.02)
    assert m2["reviewers"]["none"]["jev_wrong_accepted_group_a"]["rate"] == 0.5
    change = m2["paired"]["none -> steer"]["jev_wrong_accepted_group_a"]
    assert change["diff"] == pytest.approx(-15 / 40) and change["ci95"][1] < 0
    assert m2["causal"]["pass"] and m2["causal"]["baseline"] == "none"
    assert m2["beats_controls"] == {"random": True, "proposal": None}
    # Steering that also stops accepting right labels fails the rule.
    reviews = rows("base_local+none", 20, 39) + rows("base_local+steer", 5, 30)
    assert not st.measurement_2(reviews, tests, base_reviewer="base_local", vllm_rows=[], max_accept_drop=0.02)["causal"]["pass"]


def test_report_renders_before_anything_has_run(tmp_path):
    run = Run("e110", tmp_path)
    run.writer("test_items").append({"sample": "s", "item_id": "a", "condition": JEV_LABEL, "set": "jev_wrong"})
    text = st.report(st.summarize(run, st.E110Config(), []))
    assert "Not run yet." in text and "Not frozen yet." in text and "No trained model compared yet." in text
