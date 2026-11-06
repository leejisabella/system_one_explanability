import dataclasses
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.tasks import CIVIL_TOXICITY
from sysone.clients.base import GuardedClient, ModelSpec, Provider
from sysone.clients.fake import ok_result
from sysone.posttrain import formats, grpo
from sysone.posttrain.grpo import (CONTRARIAN, GPU_HOURS, NO_SIGNAL, Check, GrpoConfig, PrepassSettings, RewardLog,
                                   StopMonitor, StopRules, balance, check, final_text, group_stats, keep, outcome,
                                   pick_heldout, require_start, run_prepass, score, stop_reason, window)

TASK = CIVIL_TOXICITY.name
LABELS = CIVIL_TOXICITY.labels
A, B = LABELS
OVERRIDE_A = json.dumps({"verdict": "override", "final_label": A})


def _record(item_id, group, split="train", sample="civil_toxicity_train_v1", label=A):
    proposed = label if group == "jev_right" else (B if label == A else A)
    return {"prompt": formats.prompt_messages(TASK, f"ITEM:{item_id}:END", proposed),
            "meta": {"sample": sample, "item_id": item_id, "task": TASK, "group": group,
                     "proposed_label": proposed, "label": label, "split": split}}


# --- the reward ---------------------------------------------------------------------------------

def test_reward_reads_the_final_channel_in_every_decoding():
    harmony = ('<|channel|>analysis<|message|>It says {"final_label": "x"} in the policy.<|end|>'
               f"<|start|>assistant<|channel|>final<|message|>{OVERRIDE_A}<|return|>")
    stripped = f'analysisIt says {{"final_label": "x"}} in the policy.assistantfinal{OVERRIDE_A}'
    constrained = f"<|channel|>analysis<|message|>r<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>{OVERRIDE_A}"
    for text in (harmony, stripped, constrained, OVERRIDE_A, [{"role": "assistant", "content": OVERRIDE_A}]):
        assert json.loads(final_text(text)) == json.loads(OVERRIDE_A)
        assert score(outcome(text, LABELS, B), A) == 1.0
    # Cut off in the reasoning: no final answer, even though the analysis quotes JSON.
    assert final_text('<|channel|>analysis<|message|>{"verdict": "override", "final_label": "a"}') is None
    assert final_text('analysis{"verdict": "override"}') is None
    assert score(outcome("analysis thinking...", LABELS, B), A) == 0.0


def test_reward_needs_the_gold_label_and_a_verdict_that_agrees():
    r = lambda verdict, final, proposed, gold: score(
        outcome(json.dumps({"verdict": verdict, "final_label": final}), LABELS, proposed), gold)
    assert r("override", A, B, A) == 1.0       # catches Jev's mistake
    assert r("accept", A, A, A) == 1.0         # accepts Jev's right label
    assert r("accept", B, B, A) == 0.0         # gives in to the wrong label
    assert r("override", B, A, A) == 0.0       # overrides a right label
    assert r("accept", A, B, A) == 0.0         # right label, but the verdict contradicts it
    assert r("maybe", A, B, A) == 0.0          # unparsed verdict
    assert score(outcome("not json", LABELS, B), A) == 0.0


def test_reward_log_scores_and_records_each_update(tmp_path):
    log = RewardLog(tmp_path / "reward_log.jsonl", group_size=2)
    accept_a = json.dumps({"verdict": "accept", "final_label": A})
    override_b = json.dumps({"verdict": "override", "final_label": B})
    rewards = log(prompts=["p"] * 4, completions=[OVERRIDE_A, OVERRIDE_A, accept_a, override_b],
                  label=[A] * 4, proposed_label=[B, B, A, A], task=[TASK] * 4,
                  group=["jev_wrong", "jev_wrong", "jev_right", "jev_right"],
                  trainer_state=SimpleNamespace(global_step=6), completion_ids=[[1]] * 4)
    assert rewards == [1.0, 1.0, 1.0, 0.0]
    (row,) = log.writer.rows()
    assert row["step"] == 7 and row["mean_reward"] == 0.75 and row["all_same"] == 0.5
    assert row["override_jev_right"] == 0.5  # one of the two Jev-right answers overrode a right label
    assert log.__name__ == "review_reward"
    assert group_stats([1, 1, 0, 0, 1, 0], 2) == {"mean_reward": 0.5, "all_same": pytest.approx(2 / 3)}


# --- the pre-pass -------------------------------------------------------------------------------

def test_keep_and_balance():
    s = PrepassSettings()
    assert [keep({"status": "ok", "correct": c}, s) for c in (0, 1, 7, 8)] == [False, True, True, False]
    assert not keep({"status": "error", "correct": 3}, s)
    rows = ([_record(f"w{i}", "jev_wrong") for i in range(10)] + [_record(f"r{i}", "jev_right") for i in range(4)]
            + [_record(f"m{i}", "matched_wrong") for i in range(5)]
            + [_record(f"x{i}", "jev_wrong", sample="wildguard_prompt_train_v1") for i in range(3)])
    out = balance(rows, s, seed=109)
    count = lambda pool, g: sum(r["meta"]["sample"] == pool and r["meta"]["group"] == g for r in out)
    assert count("civil_toxicity_train_v1", "jev_wrong") == 4 and count("civil_toxicity_train_v1", "jev_right") == 4
    assert count("civil_toxicity_train_v1", "matched_wrong") == 1    # a third of 4
    assert count("wildguard_prompt_train_v1", "jev_wrong") == 0      # no Jev-right prompts to balance them
    assert out == balance(list(reversed(rows)), s, seed=109)         # by hash, not by order
    assert balance(rows, dataclasses.replace(s, balance=False), seed=109) == rows


def test_heldout_comes_from_val_by_hash_and_tops_up_from_train():
    val = [_record(f"v{i}", "jev_right", split="val") for i in range(5)] + [_record("vw", "jev_wrong", split="val")]
    train = [_record(f"t{i}", "jev_right") for i in range(5)] + [_record("tw", "jev_wrong")]
    held, rest = pick_heldout(val, train, 3, seed=109)
    assert len(held) == 3 and all(r["meta"]["split"] == "val" and r["meta"]["group"] == "jev_right" for r in held)
    assert rest == train
    assert held == pick_heldout(list(reversed(val)), train, 3, seed=109)[0]
    held, rest = pick_heldout(val, train, 7, seed=109)
    assert len(held) == 7 and sum(r["meta"]["split"] == "train" for r in held) == 2
    assert len(rest) == 4 and not {r["meta"]["item_id"] for r in held} & {r["meta"]["item_id"] for r in rest}


class _Reviewer(Provider):
    """Answers by item: 'w*' items are right on even sample indexes only; 'r*' always accept (right);
    'c*' always give in (wrong); every reply arrives as a vLLM chat reply's content."""

    name = "local"

    def __init__(self):
        self.calls = 0

    def send(self, model, request, price):
        self.calls += 1
        text = request.messages[-1].content
        proposed = text.split("## Automated model's decision\n")[1].split('"')[1]
        item = re.search(r"ITEM:(\w+):END", text).group(1)
        assert request.temperature == 1.0 and request.max_output_tokens + request.reasoning_max_tokens == 2048
        if item.startswith("r") or item.startswith("c") or (item.startswith("w") and request.sample_index % 2):
            return ok_result(json.dumps({"verdict": "accept", "final_label": proposed}), model=model)
        other = B if proposed == A else A
        return ok_result(json.dumps({"verdict": "override", "final_label": other}), model=model)


def _setup(tmp_path, train, val):
    data, manifests = tmp_path / "data", tmp_path / "manifests"
    folder = data / "toy"
    folder.mkdir(parents=True)
    files = {}
    for split, rows in (("train", train), ("val", val)):
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (folder / f"grpo_prompts.{split}.jsonl").write_text(body)
        files[f"grpo_prompts.{split}.jsonl"] = {"n": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (manifests / "toy").mkdir(parents=True)
    (manifests / "toy" / "manifest.json").write_text(json.dumps({"name": "toy", "files": files}))
    config = GrpoConfig(name="t-grpo", dataset="toy", init_from="checkpoints/start", checkpoints_dir=str(tmp_path / "ck"),
                        stop=StopRules(heldout_prompts=3), prepass=PrepassSettings(workers=4))
    return config, data, manifests


def test_prepass_counts_filters_and_resumes(tmp_path):
    train = ([_record(f"w{i}", "jev_wrong") for i in range(6)] + [_record(f"r{i}", "jev_right") for i in range(6)]
             + [_record(f"c{i}", "jev_wrong") for i in range(2)])
    val = [_record(f"r{i}v", "jev_right", split="val") for i in range(3)]
    config, data, manifests = _setup(tmp_path, train, val)
    assert check(config, data_dir=data, manifest_dir=manifests)["train"]["groups"] == {"jev_right": 6, "jev_wrong": 8}
    provider = _Reviewer()
    guard = CostGuard(Ledger(tmp_path / "ledger.jsonl", "t"), per_call_cap_usd=0.01, run_cap_usd=0.01,
                      project_caps_usd={"local": 0.0})
    client = GuardedClient(ModelSpec("s", "local", "start", Price(0, 0)), provider, guard)
    summary = run_prepass(config, client, data_dir=data)
    assert provider.calls == (14 + 3) * 8
    rows = {r["item_id"]: r for r in grpo.JsonlWriter(config.prepass_dir / "prepass.jsonl").rows()}
    assert rows["w0"]["correct"] == 4 and rows["r0"]["correct"] == 8 and rows["c0"]["correct"] == 0
    assert rows["r0v"]["role"] == "heldout" and rows["r0v"]["accepted"] == 8
    # Only the w* prompts teach (4 of 8); r* are all right, c* all wrong. No Jev-right prompt is left to
    # balance them with, so balancing would empty the set: check the unbalanced filter too.
    assert summary["kept_1_to_7"] == {"civil_toxicity_train_v1/jev_wrong": 6}
    assert summary["train"] == {}
    assert summary["heldout"]["accept_when_right"] == 1.0 and summary["heldout"]["n"] == 3
    assert summary["correct_counts"]["civil_toxicity_train_v1/jev_wrong/0"] == 2
    # Resuming makes no new requests and rebuilds the same files.
    unbalanced = dataclasses.replace(config, prepass=dataclasses.replace(config.prepass, balance=False))
    again = run_prepass(unbalanced, client, data_dir=data)
    assert provider.calls == 17 * 8 and again["train"] == {"civil_toxicity_train_v1/jev_wrong": 6}
    kept = [json.loads(x) for x in (config.prepass_dir / "train.jsonl").read_text().splitlines()]
    assert {r["meta"]["prepass_correct"] for r in kept} == {4} and again["train_mean_correct"] == 0.5
    assert again["wall_hours"] == summary["wall_hours"]


# --- the stop rules -----------------------------------------------------------------------------

def test_stop_rule_needs_two_checks_in_a_row():
    rules = StopRules()
    c = lambda step, acc, mean=0.5, same=0.2: Check(step, acc, mean, same)
    # Contrarian: below 0.90 - 0.03 = 0.87 at two checks in a row.
    assert stop_reason([c(25, 0.86)], 0.90, rules) is None
    assert stop_reason([c(25, 0.86), c(50, 0.88), c(75, 0.86)], 0.90, rules) is None
    assert stop_reason([c(25, 0.88), c(50, 0.86), c(75, 0.85)], 0.90, rules) == CONTRARIAN
    assert stop_reason([c(25, 0.87), c(50, 0.87)], 0.90, rules) is None  # exactly 3 points: not more than
    # No signal: flat against the previous window and over 90% all-same, twice; never at the first check.
    flat = [c(25, 0.9, 0.80, 0.95), c(50, 0.9, 0.81, 0.95)]
    assert stop_reason(flat, 0.90, rules) is None
    assert stop_reason(flat + [c(75, 0.9, 0.805, 0.93)], 0.90, rules) == NO_SIGNAL
    assert stop_reason(flat + [c(75, 0.9, 0.85, 0.93)], 0.90, rules) is None   # reward still moving
    assert stop_reason(flat + [c(75, 0.9, 0.81, 0.85)], 0.90, rules) is None   # groups still informative
    rows = [{"step": s, "mean_reward": m, "all_same": a} for s, m, a in ((24, 0.0, 0.0), (25, 1.0, 1.0), (26, 0.5, 0.5),
                                                                          (26, 0.25, 0.5), (50, 0.75, 0.5))]
    assert window(rows, 25, 50) == (0.5, 0.5)  # the resumed update 26's latest row wins


def test_stop_monitor_checks_every_25_updates_and_resumes(tmp_path):
    rules = StopRules(check_every=25, gpu_hour_cap=15.0, gpus=2)
    log = grpo.JsonlWriter(tmp_path / "reward_log.jsonl")
    for step in range(1, 51):
        log.append({"step": step, "mean_reward": 0.5, "all_same": 0.2})
    seen, now = [], [0.0]
    accept = lambda step: seen.append(step) or 0.80
    monitor = StopMonitor(tmp_path, rules, start_accept=0.90, prepass_hours=0.5, accept_fn=accept, clock=lambda: now[0])
    control = SimpleNamespace(should_training_stop=False)
    for step in range(1, 26):
        monitor.on_step_end(None, SimpleNamespace(global_step=step), control)
    assert seen == [25] and not control.should_training_stop  # one contrarian check isn't enough
    # A resumed run continues the same checks and the same clock.
    monitor = StopMonitor(tmp_path, rules, start_accept=0.90, prepass_hours=0.5, accept_fn=accept, clock=lambda: now[0])
    for step in range(26, 51):
        monitor.on_step_end(None, SimpleNamespace(global_step=step), control)
    assert seen == [25, 50] and control.should_training_stop and monitor.reason == CONTRARIAN
    assert [c.step for c in monitor.checks()] == [25, 50] and monitor.checks()[0].mean_reward == 0.5

    # The GPU-hour cap: (0.5 pre-pass + 7 training hours) x 2 GPUs = 15.
    out = tmp_path / "cap"
    out.mkdir()
    monitor = StopMonitor(out, rules, start_accept=0.90, prepass_hours=0.5, accept_fn=lambda s: 0.9, clock=lambda: now[0])
    control = SimpleNamespace(should_training_stop=False)
    now[0] = 6.9 * 3600
    monitor.on_step_end(None, SimpleNamespace(global_step=1), control)
    assert not control.should_training_stop
    now[0] = 7.0 * 3600
    monitor.on_step_end(None, SimpleNamespace(global_step=2), control)
    assert control.should_training_stop and monitor.reason == GPU_HOURS
    assert json.loads((out / "gpu_time.json").read_text())["train_wall_s"] == pytest.approx(7.0 * 3600)


# --- the config and --check ---------------------------------------------------------------------

def test_committed_config_matches_the_design():
    from sysone.registry import CONFIG_DIR

    config = GrpoConfig.load(CONFIG_DIR / "posttrain" / "grpo_review.yaml")
    assert (config.name, config.method, config.arm) == ("gpt-oss-20b-grpo-review", "grpo", "grpo_prompts")
    assert (config.num_generations, config.temperature, config.max_completion_length, config.beta) == (8, 1.0, 2048, 0.04)
    assert (config.lora.r, config.lora.target_modules, config.learning_rate) == (8, "all-linear", 1e-5)
    assert (config.max_steps, config.prompts_per_update, config.seed) == (300, 16, 109)
    assert config.gradient_accumulation_steps * config.per_device_batch_size == 16 * 8
    assert (config.prepass.samples, config.prepass.keep_min_correct, config.prepass.keep_max_correct) == (8, 1, 7)
    s = config.stop
    assert (s.check_every, s.consecutive, s.contrarian_drop, s.heldout_prompts, s.all_same_share, s.gpu_hour_cap) == \
        (25, 2, 0.03, 200, 0.90, 15.0)
    assert config.vllm_mode == "server"
    with pytest.raises(ValueError, match="placeholder"):
        require_start(config)


def test_check_validates_the_prompts_against_the_manifest(tmp_path):
    train = [_record("w0", "jev_wrong"), _record("r0", "jev_right"), _record("m0", "matched_wrong")]
    val = [_record("r1", "jev_right", split="val")]
    config, data, manifests = _setup(tmp_path, train, val)
    stats = check(config, data_dir=data, manifest_dir=manifests)
    assert stats["train"]["records"] == 3 and stats["val"]["groups"] == {"jev_right": 1} and stats["init_from_set"]
    with pytest.raises(ValueError, match="multiple of per_device_batch_size"):
        check(dataclasses.replace(config, per_device_batch_size=3), data_dir=data, manifest_dir=manifests)
    (data / "toy" / "grpo_prompts.val.jsonl").write_text(json.dumps(_record("r1", "jev_right", split="train")) + "\n")
    with pytest.raises(ValueError, match="manifest hash") as e:
        check(config, data_dir=data, manifest_dir=manifests)
    assert "split 'train' in the val file" in str(e.value)
    with pytest.raises(ValueError, match="is missing"):
        check(dataclasses.replace(config, dataset="other"), data_dir=data, manifest_dir=manifests)


def test_check_cli_needs_no_torch(tmp_path, monkeypatch, capsys):
    import sys

    train = [_record("w0", "jev_wrong")]
    config, data, manifests = _setup(tmp_path, train, [_record("r1", "jev_right", split="val")])
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps({"name": "t", "dataset": "toy", "init_from": "checkpoints/TODO-x"}))
    real = grpo.check
    monkeypatch.setattr(grpo, "check", lambda c: real(c, data_dir=data, manifest_dir=manifests))
    monkeypatch.setattr(sys, "argv", ["grpo", "--config", str(path), "--check"])
    grpo.main()
    out = capsys.readouterr().out
    assert "prompts toy/grpo_prompts OK" in out and "placeholder" in out
    assert "torch" not in sys.modules
    monkeypatch.setattr(sys, "argv", ["grpo", "--config", str(path), "--paths"])
    grpo.main()
    assert 'SERVED_NAME="t"' in capsys.readouterr().out
