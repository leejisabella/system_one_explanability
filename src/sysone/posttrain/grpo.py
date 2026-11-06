"""E.109: GRPO of the reviewer in review mode, with Jev's proposal shown (research_docs/15).

For each prompt the model writes num_generations (8) answers; an answer scores 1 if it parses, its
final label is the gold label and its verdict agrees with that label (accept iff the final label is
the proposed one), else 0. There is no term for agreeing with Jev and no length term.

Steps, each its own command:
  --check    no torch: the config, and the prompt files against their manifest hash (E.108's export).
  --prepass  against a vLLM server already serving the starting model (scripts/serve_part2.sh, under
             prepass.served_name; LOCAL_LLM_BASE_URL), 8 answers per candidate prompt at temperature 1.0
             with the same token budget as training. Writes, under checkpoints/<name>/prepass/:
               prepass.jsonl  correct/accepted counts per prompt (resumable)
               heldout.jsonl  200 Jev-right prompts for the contrarian stop rule, from the val split by hash
               train.jsonl    the prompts that teach (1-7 of 8 correct), balanced per pool (see `balance`)
               summary.json   counts by group and pool, and the starting model's accept-when-right on heldout
  (train)    TRL's GRPOTrainer with a LoRA adapter, generating with vLLM: by default a `trl vllm-serve`
             on a second GPU (vllm_mode: server), or colocated on one GPU (vllm_mode: colocate). Stops early
             by the pre-registered rules (StopMonitor), checked every 25 updates, or at the GPU-hour cap.
  --merge    the adapter merged into checkpoints/<name>-merged, for vLLM to serve.

Usage (scripts/run_e109.sh runs them all in order on the GPU machine):
  uv run python -m sysone.posttrain.grpo --config configs/posttrain/grpo_review.yaml --check
  uv run python -m sysone.posttrain.grpo --config configs/posttrain/grpo_review.yaml --prepass
  uv run --group train python -m sysone.posttrain.grpo --config configs/posttrain/grpo_review.yaml
  uv run --group train python -m sysone.posttrain.grpo --config configs/posttrain/grpo_review.yaml --merge

Prompts are rendered to text here with gpt-oss's chat template at reasoning_effort medium, as
hf.render does: TRL 0.23's GRPOTrainer applies the template itself without passing template kwargs,
so a conversational prompt would get the template's default reasoning level. A text prompt is used
as is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from sysone.budget import CostGuard, Ledger, Price
from sysone.cascade.reviewer import ReviewResult, parse_review
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.posttrain.mix import JEV_RIGHT, JEV_WRONG, MATCHED_WRONG, TRAIN, VAL
from sysone.posttrain.sft import write_info
from sysone.posttrain.train_config import LoraSettings
from sysone.registry import REPO_ROOT
from sysone.runs import JsonlWriter, now_iso
from sysone.types import ChatRequest, ChatResponse, Message, Usage

GROUPS = (JEV_WRONG, JEV_RIGHT, MATCHED_WRONG)
PLACEHOLDER = "TODO"
CONTRARIAN, NO_SIGNAL, GPU_HOURS = "contrarian", "no_signal", "gpu_hours"


@dataclass(frozen=True)
class PrepassSettings:
    samples: int = 8
    keep_min_correct: int = 1
    keep_max_correct: int = 7
    # The served name of the starting model during the pre-pass (scripts/serve_part2.sh <init_from> <name>).
    served_name: str = "gpt-oss-20b-grpo-start"
    workers: int = 64
    # Tokens of the completion budget kept for the final answer; the rest is the reasoning budget.
    answer_tokens: int = 256
    # Per pool, after the filter: Jev-right prompts and Jev-wrong prompts are cut to the smaller of the
    # two (jev_right_per_wrong 1.0: the design's 50:50), and matched_wrong to this share of them (E.108's).
    balance: bool = True
    jev_right_per_wrong: float = 1.0
    matched_wrong_per_wrong: float = 0.3333


@dataclass(frozen=True)
class StopRules:
    check_every: int = 25
    consecutive: int = 2
    # Contrarian: held-out accept-when-right more than this far below the starting model's pre-pass rate.
    contrarian_drop: float = 0.03
    heldout_prompts: int = 200
    heldout_samples: int = 4
    # No signal: mean reward flat (window mean within flat_tolerance of the previous window's) and the
    # share of all-same groups above all_same_share.
    flat_tolerance: float = 0.02
    all_same_share: float = 0.90
    gpu_hour_cap: float = 15.0
    # GPUs held for the whole run (2 in server mode, 1 colocated): GPU hours = wall hours x gpus.
    gpus: int = 2


@dataclass(frozen=True)
class GrpoConfig:
    name: str
    dataset: str
    init_from: str
    method: str = "grpo"
    arm: str = "grpo_prompts"
    base_model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "medium"
    lora: LoraSettings = field(default_factory=LoraSettings)
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    num_generations: int = 8
    temperature: float = 1.0
    max_completion_length: int = 2048
    # None: never truncate a prompt (TRL's default of 512 would cut the policy text).
    max_prompt_length: int | None = None
    beta: float = 0.04
    loss_type: str = "dapo"
    prompts_per_update: int = 16
    # Completions per forward pass; gradient accumulation makes up prompts_per_update x num_generations.
    per_device_batch_size: int = 2
    max_steps: int = 300
    seed: int = 109
    save_steps: int = 25
    vllm_mode: str = "server"
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8001
    vllm_gpu_memory_utilization: float = 0.3   # colocate only
    vllm_enable_sleep_mode: bool = True        # colocate only
    prepass: PrepassSettings = field(default_factory=PrepassSettings)
    stop: StopRules = field(default_factory=StopRules)
    checkpoints_dir: str = "checkpoints"

    @classmethod
    def load(cls, path: str | Path) -> GrpoConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**{**raw, "lora": LoraSettings(**raw.get("lora", {})),
                      "prepass": PrepassSettings(**raw.get("prepass", {})), "stop": StopRules(**raw.get("stop", {}))})

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / self.checkpoints_dir / self.name

    @property
    def merged_dir(self) -> Path:
        return REPO_ROOT / self.checkpoints_dir / f"{self.name}-merged"

    @property
    def prepass_dir(self) -> Path:
        return self.output_dir / "prepass"

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.prompts_per_update * self.num_generations // self.per_device_batch_size

    def files(self, data_dir: Path | None = None) -> dict[str, Path]:
        folder = (data_dir or REPO_ROOT / "data" / "posttrain") / self.dataset
        return {split: folder / f"{self.arm}.{split}.jsonl" for split in (TRAIN, VAL)}


# --- the reward ---------------------------------------------------------------------------------

_FINAL = re.compile(r"<\|channel\|>final.*?<\|message\|>", re.S)
_STOP = re.compile(r"<\|(?:return|end|call)\|>.*", re.S)


def final_text(completion: str | list[dict[str, Any]]) -> str | None:
    """The final-channel text of a gpt-oss completion, or None if it never got there (cut off in its
    reasoning). Takes the harmony markup (analysis, then <|start|>assistant<|channel|>final<|message|>),
    the same with the special tokens stripped, as TRL decodes completions ("analysis...assistantfinal{...}"),
    or a plain reply, as a chat endpoint returns its content."""
    if isinstance(completion, list):  # a conversational completion
        completion = completion[-1]["content"] if completion else ""
    if "<|channel|>" in completion:
        parts = _FINAL.split(completion)
        return _STOP.sub("", parts[-1]) if len(parts) > 1 else None
    if completion.lstrip().startswith("analysis"):
        _, found, tail = completion.rpartition("assistantfinal")
        return tail if found else None
    return completion


def _response(text: str) -> ChatResponse:
    return ChatResponse(text=text, usage=Usage(0, 0), cost_usd=0.0, provider="grpo", model_requested="",
                        model_used="", finish_reason=None, raw={})


def outcome(completion: str | list[dict[str, Any]], labels: tuple[str, ...], proposed_label: str) -> ReviewResult | None:
    """The completion's review, parsed by the evaluation's own rule (Reviewer's parse_review), or None
    if it has no final answer."""
    text = final_text(completion)
    return None if text is None else parse_review(_response(text), "", labels, proposed_label)


def score(r: ReviewResult | None, label: str) -> float:
    """1 if the review parsed, reached the gold label and its verdict agrees with it; else 0."""
    return float(r is not None and r.parse_ok and r.final_label == label and bool(r.consistent))


def group_stats(rewards: list[float], size: int) -> dict[str, float]:
    """Mean reward, and the share of groups (consecutive runs of `size`, as TRL groups them) whose
    answers all scored the same, which carry no learning signal."""
    groups = [rewards[i:i + size] for i in range(0, len(rewards), size)]
    return {"mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "all_same": sum(max(g) == min(g) for g in groups) / len(groups) if groups else 0.0}


class RewardLog:
    """TRL's reward function (reward_funcs=[RewardLog(...)]), which also records each update's mean
    reward, all-same share and override rate on Jev-right prompts to reward_log.jsonl for the stop rules.
    TRL passes the dataset's columns as lists, one entry per completion, and its TrainerState."""

    __name__ = "review_reward"  # TRL names a reward function by __name__

    def __init__(self, path: Path, group_size: int):
        self.writer = JsonlWriter(path)
        self.group_size = group_size

    def __call__(self, prompts: list, completions: list, *, label: list[str], proposed_label: list[str],
                 task: list[str], group: list[str], trainer_state: Any = None, **_: Any) -> list[float]:
        reviews = [outcome(c, TASKS[t].labels, p) for c, t, p in zip(completions, task, proposed_label)]
        rewards = [score(r, g) for r, g in zip(reviews, label)]
        right = [r.final_label != p for r, p, grp in zip(reviews, proposed_label, group)
                 if grp == JEV_RIGHT and r is not None and r.parse_ok]
        self.writer.append({
            # The completions of one generation batch feed the next update (steps_per_generation =
            # gradient accumulation, one iteration), so they belong to update global_step + 1.
            "step": (trainer_state.global_step if trainer_state is not None else 0) + 1,
            **group_stats(rewards, self.group_size), "n": len(rewards),
            "parsed": sum(r is not None and r.parse_ok for r in reviews) / len(reviews) if reviews else 0.0,
            "override_jev_right": sum(right) / len(right) if right else None,
            "ts": now_iso()})
        return rewards


# --- the stop rules -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    step: int
    accept_right: float | None   # held-out accept-when-right at this step
    mean_reward: float           # mean reward over the updates since the previous check
    all_same: float              # share of all-same groups over the same updates


def window(rows: list[dict[str, Any]], lo: int, hi: int) -> tuple[float, float]:
    """Mean reward and all-same share over updates lo < step <= hi; a resumed update's latest row wins."""
    latest = {r["step"]: r for r in rows if lo < r["step"] <= hi}
    if not latest:
        return 0.0, 0.0
    return (sum(r["mean_reward"] for r in latest.values()) / len(latest),
            sum(r["all_same"] for r in latest.values()) / len(latest))


def stop_reason(checks: list[Check], start_accept: float, rules: StopRules) -> str | None:
    """CONTRARIAN or NO_SIGNAL if that rule held at each of the last `consecutive` checks, else None.

    contrarian at a check: held-out accept-when-right < start_accept - contrarian_drop.
    no signal at a check: the window's mean reward is flat, |m_k - m_(k-1)| < flat_tolerance against
    the previous check's window (so never at the first check), and its all-same share > all_same_share.
    """
    def contrarian(i: int) -> bool:
        a = checks[i].accept_right
        return a is not None and a < start_accept - rules.contrarian_drop

    def no_signal(i: int) -> bool:
        return (i > 0 and abs(checks[i].mean_reward - checks[i - 1].mean_reward) < rules.flat_tolerance
                and checks[i].all_same > rules.all_same_share)

    if len(checks) < rules.consecutive:
        return None
    for name, rule in ((CONTRARIAN, contrarian), (NO_SIGNAL, no_signal)):
        if all(rule(i) for i in range(len(checks) - rules.consecutive, len(checks))):
            return name
    return None


class StopMonitor:
    """The stop rules, as TrainerCallback methods (train() mixes this into transformers' TrainerCallback).

    Every check_every updates it measures accept-when-right on the held-out Jev-right prompts with
    `accept_fn(step)` (generation by the trainer's vLLM, with the current weights), reads the reward log,
    and stops training if stop_reason fires. At every update it adds the wall time to gpu_time.json and
    stops at the GPU-hour cap, counting the pre-pass's hours too. Checks are kept in stop_checks.jsonl,
    so a resumed run continues the same sequence of checks and the same clock."""

    def __init__(self, out: Path, rules: StopRules, *, start_accept: float, prepass_hours: float,
                 accept_fn: Callable[[int], float], clock: Callable[[], float] = time.monotonic):
        self.rules, self.start_accept, self.prepass_hours = rules, start_accept, prepass_hours
        self.accept_fn, self.clock = accept_fn, clock
        self.checks_log, self.rewards_log = JsonlWriter(out / "stop_checks.jsonl"), JsonlWriter(out / "reward_log.jsonl")
        self.time_path = out / "gpu_time.json"
        self.train_s = json.loads(self.time_path.read_text())["train_wall_s"] if self.time_path.exists() else 0.0
        self.last = clock()
        self.reason: str | None = None

    def gpu_hours(self) -> float:
        return (self.prepass_hours + self.train_s / 3600) * self.rules.gpus

    def checks(self) -> list[Check]:
        """The checks so far (step 0, the starting model on the training path, is logged but isn't one)."""
        latest = {r["step"]: r for r in self.checks_log.rows() if r["step"] > 0}
        return [Check(r["step"], r["accept_right"], r["mean_reward"], r["all_same"]) for _, r in sorted(latest.items())]

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        step = state.global_step
        if step % self.rules.check_every == 0:
            mean, same = window(self.rewards_log.rows(), step - self.rules.check_every, step)
            accept = self.accept_fn(step)
            self.checks_log.append({"step": step, "accept_right": accept, "mean_reward": mean, "all_same": same,
                                    "start_accept": self.start_accept, "ts": now_iso()})
            self.reason = stop_reason([c for c in self.checks() if c.step <= step], self.start_accept, self.rules)
        now = self.clock()
        self.train_s += now - self.last
        self.last = now
        self.time_path.write_text(json.dumps({"train_wall_s": self.train_s}) + "\n")
        if self.reason is None and self.gpu_hours() >= self.rules.gpu_hour_cap:
            self.reason = GPU_HOURS
        if self.reason is not None:
            print(f"stopping at update {step}: {self.reason}")
            control.should_training_stop = True
        return control


# --- the check and the pre-pass -----------------------------------------------------------------

def _record_problem(record: dict[str, Any], split: str) -> str | None:
    prompt, meta = record.get("prompt"), record.get("meta") or {}
    if not prompt or prompt[-1].get("role") != "user":
        return "the prompt must be a conversation ending with a user turn"
    if any(k not in meta for k in ("sample", "item_id", "task", "group", "proposed_label", "label", "split")):
        return "meta is missing a key"
    if meta["task"] not in TASKS or not {meta["label"], meta["proposed_label"]} <= set(TASKS[meta["task"]].labels):
        return "unknown task, or a label not among the task's labels"
    if meta["group"] not in GROUPS:
        return f"group {meta['group']!r} is not one of {GROUPS}"
    if meta["split"] != split:
        return f"split {meta['split']!r} in the {split} file"
    return None


def check(config: GrpoConfig, *, data_dir: Path | None = None, manifest_dir: Path | None = None) -> dict[str, Any]:
    """Problems that would waste GPU time, found without a GPU. Raises ValueError listing them."""
    problems: list[str] = []
    if config.method != "grpo" or config.arm != "grpo_prompts":
        problems.append("method must be grpo and arm grpo_prompts")
    if (config.prompts_per_update * config.num_generations) % config.per_device_batch_size:
        problems.append("prompts_per_update x num_generations must be a multiple of per_device_batch_size")
    if config.vllm_mode not in ("server", "colocate"):
        problems.append(f"vllm_mode {config.vllm_mode!r} is not server or colocate")
    if config.prepass.answer_tokens >= config.max_completion_length:
        problems.append("prepass.answer_tokens must leave a reasoning budget under max_completion_length")
    manifest_path = (manifest_dir or REPO_ROOT / "benchmarks" / "posttrain") / config.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None:
        problems.append(f"{manifest_path} is missing; export the dataset with E.108 first")
    stats: dict[str, Any] = {"init_from_set": PLACEHOLDER not in config.init_from}
    for split, path in config.files(data_dir).items():
        if not path.exists():
            problems.append(f"{path} is missing")
            continue
        body = path.read_text()
        want = (manifest or {}).get("files", {}).get(path.name, {}).get("sha256")
        if manifest is not None and hashlib.sha256(body.encode()).hexdigest() != want:
            problems.append(f"{path.name} doesn't match its manifest hash; re-export it or fix the dataset name")
        records = [json.loads(line) for line in body.splitlines() if line.strip()]
        bad = [(i, p) for i, r in enumerate(records) if (p := _record_problem(r, split))]
        if bad:
            problems.append(f"{path.name}: {len(bad)} malformed records, first #{bad[0][0]}: {bad[0][1]}")
        if not records:
            problems.append(f"{path.name} has no records")
        ok = [r for i, r in enumerate(records) if i not in {j for j, _ in bad}]
        stats[split] = {"records": len(records), "groups": dict(sorted(Counter(r["meta"]["group"] for r in ok).items())),
                        "estimated_max_prompt_tokens": max((len(r["prompt"][-1]["content"]) // 4 for r in ok), default=0)}
    if problems:
        raise ValueError("GRPO config check failed:\n- " + "\n- ".join(problems))
    return stats


def require_start(config: GrpoConfig) -> Path:
    """The starting checkpoint, which every GPU step needs and --check doesn't."""
    if PLACEHOLDER in config.init_from:
        raise ValueError(f"init_from is still the placeholder {config.init_from!r}: set it to the starting "
                         "checkpoint chosen by the rule in the config, and commit it, before the GPU steps")
    path = REPO_ROOT / config.init_from
    if not path.exists():
        raise ValueError(f"{path} is missing")
    return path


def _key(meta: dict[str, Any]) -> tuple[str, str, str]:
    return meta["sample"], meta["item_id"], meta["group"]


def _rank(seed: int, meta: dict[str, Any], what: str) -> str:
    return hashlib.sha256(f"{seed}:{meta['sample']}:{meta['item_id']}:{what}".encode()).hexdigest()


def pick_heldout(val: list[dict[str, Any]], train: list[dict[str, Any]], n: int,
                 seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(held-out Jev-right prompts, the train prompts left). The first n Jev-right val prompts by hash;
    if val has fewer, the rest come from train by hash and leave the training candidates."""
    right = lambda rows: sorted((r for r in rows if r["meta"]["group"] == JEV_RIGHT), key=lambda r: _rank(seed, r["meta"], "heldout"))
    held = right(val)[:n]
    extra = right(train)[:max(0, n - len(held))]
    taken = {_key(r["meta"]) for r in extra}
    return held + extra, [r for r in train if _key(r["meta"]) not in taken]


def keep(row: dict[str, Any], settings: PrepassSettings) -> bool:
    """A prompt that teaches: some of its answers right and some wrong."""
    return row["status"] == "ok" and settings.keep_min_correct <= row["correct"] <= settings.keep_max_correct


def balance(rows: list[dict[str, Any]], settings: PrepassSettings, seed: int) -> list[dict[str, Any]]:
    """Per pool, by hash: Jev-wrong and Jev-right prompts cut to the same count (x jev_right_per_wrong),
    and matched_wrong to matched_wrong_per_wrong of it. The filter drops far more Jev-right prompts
    (mostly all right) than Jev-wrong ones, and the 50:50 mix is what keeps GRPO from learning to override."""
    if not settings.balance:
        return rows
    out = []
    for pool in sorted({r["meta"]["sample"] for r in rows}):
        by = {g: sorted((r for r in rows if r["meta"]["sample"] == pool and r["meta"]["group"] == g),
                        key=lambda r: _rank(seed, r["meta"], "balance")) for g in GROUPS}
        n_wrong = min(len(by[JEV_WRONG]), int(len(by[JEV_RIGHT]) / settings.jev_right_per_wrong))
        n_right = min(len(by[JEV_RIGHT]), round(n_wrong * settings.jev_right_per_wrong))
        n_matched = min(len(by[MATCHED_WRONG]), round(n_wrong * settings.matched_wrong_per_wrong))
        out += by[JEV_WRONG][:n_wrong] + by[JEV_RIGHT][:n_right] + by[MATCHED_WRONG][:n_matched]
    return out


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{r['meta']['sample']}/{r['meta']['group']}" for r in rows).items()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def prepass_client(config: GrpoConfig) -> GuardedClient:
    """The starting model on the local vLLM server, at $0: a priced call would be refused."""
    from sysone.clients.local import LocalProvider

    guard = CostGuard(Ledger(config.prepass_dir / "ledger.jsonl", config.name), per_call_cap_usd=0.01,
                      run_cap_usd=0.01, project_caps_usd={"local": 0.0})
    spec = ModelSpec("grpo_start", "local", config.prepass.served_name, Price(0.0, 0.0),
                     reasoning_effort=config.reasoning_effort)
    return GuardedClient(spec, LocalProvider(), guard, cache=None)


def run_prepass(config: GrpoConfig, client: GuardedClient, *, data_dir: Path | None = None) -> dict[str, Any]:
    """Sample every candidate and held-out prompt, then write the training set and the summary."""
    from sysone.experiments.e101 import run_parallel

    s = config.prepass
    out = config.prepass_dir
    out.mkdir(parents=True, exist_ok=True)
    files = config.files(data_dir)
    load = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    heldout, candidates = pick_heldout(load(files[VAL]), load(files[TRAIN]), config.stop.heldout_prompts, config.seed)
    _write_jsonl(out / "heldout.jsonl", heldout)
    role = {**{_key(r["meta"]): "train" for r in candidates}, **{_key(r["meta"]): "heldout" for r in heldout}}

    writer = JsonlWriter(out / "prepass.jsonl")
    done = {tuple(r["key"]) for r in writer.rows() if r["status"] == "ok"}
    started = time.monotonic()

    def work(record: dict[str, Any]) -> dict[str, Any]:
        meta = record["meta"]
        base = {"key": list(_key(meta)), "role": role[_key(meta)], **{k: meta[k] for k in ("sample", "item_id", "task", "group", "split")}}
        reviews = []
        try:
            for k in range(s.samples):
                request = ChatRequest(
                    messages=tuple(Message(m["role"], m["content"]) for m in record["prompt"]),
                    max_output_tokens=s.answer_tokens, reasoning_max_tokens=config.max_completion_length - s.answer_tokens,
                    temperature=config.temperature, sample_index=k,
                    tags={"experiment": "e109", "step": "prepass", "item_id": meta["item_id"], "sample_index": str(k)})
                reviews.append(parse_review(client.complete(request), "", TASKS[meta["task"]].labels, meta["proposed_label"]))
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        parsed = [r for r in reviews if r.parse_ok]
        return {**base, "status": "ok", "n": len(reviews), "parsed": len(parsed),
                "correct": int(sum(score(r, meta["label"]) for r in reviews)),
                "accepted": sum(r.final_label == meta["proposed_label"] for r in parsed), "ts": now_iso()}

    todo = [r for r in heldout + candidates if _key(r["meta"]) not in done]
    run_parallel(todo, work, writer, workers=s.workers, max_consecutive_errors=10)
    latest = {tuple(r["key"]): r for r in writer.rows() if r["status"] == "ok"}
    missing = [r for r in heldout + candidates if _key(r["meta"]) not in latest]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts have no pre-pass result yet (errors); run --prepass again")

    taught = [{**r, "meta": {**r["meta"], "prepass_correct": latest[_key(r["meta"])]["correct"]}}
              for r in candidates if keep(latest[_key(r["meta"])], s)]
    train_rows = balance(taught, s, config.seed)
    _write_jsonl(out / "train.jsonl", train_rows)
    held = [latest[_key(r["meta"])] for r in heldout]
    accept = sum(r["accepted"] for r in held) / max(1, sum(r["parsed"] for r in held))
    by_correct = lambda rows: dict(sorted(Counter(f"{latest[_key(r['meta'])]['sample']}/{r['meta']['group']}/"
                                                  f"{latest[_key(r['meta'])]['correct']}" for r in rows).items()))
    previous = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    summary = {
        "finished_at": now_iso(), "served_model": s.served_name, "init_from": config.init_from,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
        # Wall hours of the sampling; a rerun that only rebuilt the files adds nothing.
        "wall_hours": previous.get("wall_hours", 0.0) + (time.monotonic() - started) / 3600 * bool(todo),
        "candidates": _counts(candidates), "kept_1_to_7": _counts(taught), "train": _counts(train_rows),
        "correct_counts": by_correct(candidates),
        "train_mean_correct": sum(r["meta"]["prepass_correct"] for r in train_rows) / max(1, len(train_rows)) / s.samples,
        "heldout": {"n": len(held), "groups": _counts(heldout), "accept_when_right": accept,
                    "parsed": sum(r["parsed"] for r in held), "samples": s.samples * len(held)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


# --- training (GPU) -----------------------------------------------------------------------------

def train(config: GrpoConfig) -> tuple[Path, str | None]:
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from sysone.posttrain.hf import load_model, load_records, lora_config

    source = str(require_start(config))
    out = config.output_dir
    summary_path = config.prepass_dir / "summary.json"
    if not summary_path.exists():
        raise ValueError("run --prepass first")
    summary = json.loads(summary_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(source)
    text = lambda prompt: tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                                        reasoning_effort=config.reasoning_effort)
    rows = [{"prompt": text(r["prompt"]), **{k: r["meta"][k] for k in ("label", "proposed_label", "task", "group")}}
            for r in load_records(config.prepass_dir / "train.jsonl")]
    if not rows:
        raise ValueError("the pre-pass kept no prompts that teach; see prepass/summary.json")
    heldout = load_records(config.prepass_dir / "heldout.jsonl")
    held_text = [text(r["prompt"]) for r in heldout]

    vllm = {"vllm_mode": config.vllm_mode}
    if config.vllm_mode == "server":
        vllm |= {"vllm_server_host": config.vllm_server_host, "vllm_server_port": config.vllm_server_port}
    else:
        vllm |= {"vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
                 "vllm_enable_sleep_mode": config.vllm_enable_sleep_mode}
    args = GRPOConfig(
        output_dir=str(out), learning_rate=config.learning_rate, lr_scheduler_type=config.lr_scheduler_type,
        max_steps=config.max_steps, per_device_train_batch_size=config.per_device_batch_size,
        # One update = prompts_per_update prompts x num_generations answers, all generated in one batch.
        gradient_accumulation_steps=config.gradient_accumulation_steps, num_generations=config.num_generations,
        temperature=config.temperature, max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length, beta=config.beta, loss_type=config.loss_type,
        gradient_checkpointing=True, bf16=True, logging_steps=1, save_strategy="steps", save_steps=config.save_steps,
        seed=config.seed, report_to="none", use_vllm=True, **vllm,
    )
    trainer: Any = None

    def accept_fn(step: int) -> float:
        # The current weights to vLLM first: TRL only syncs them at the next generation.
        llm = trainer.llm if config.vllm_mode == "colocate" else None
        if llm is not None and config.vllm_enable_sleep_mode:
            llm.wake_up()
        trainer._move_model_to_vllm()
        trainer._last_loaded_step = step
        n, budget = config.stop.heldout_samples, config.max_completion_length
        if llm is None:
            ids = trainer.vllm_client.generate(prompts=held_text, n=n, temperature=config.temperature,
                                               max_tokens=budget)["completion_ids"]
        else:
            from vllm import SamplingParams

            outs = llm.generate(held_text, SamplingParams(n=n, temperature=config.temperature, max_tokens=budget), use_tqdm=False)
            ids = [o.token_ids for g in outs for o in g.outputs]
            if config.vllm_enable_sleep_mode:
                llm.sleep(level=1)
        metas = [r["meta"] for r in heldout for _ in range(n)]  # vLLM returns a prompt's n answers together
        reviews = [(outcome(t, TASKS[m["task"]].labels, m["proposed_label"]), m["proposed_label"])
                   for t, m in zip(tokenizer.batch_decode(ids, skip_special_tokens=False), metas)]
        parsed = [r.final_label == p for r, p in reviews if r is not None and r.parse_ok]
        return sum(parsed) / max(1, len(parsed))

    class Stop(StopMonitor, TrainerCallback):
        pass

    stop = Stop(out, config.stop, start_accept=summary["heldout"]["accept_when_right"],
                prepass_hours=summary["wall_hours"], accept_fn=accept_fn)
    trainer = GRPOTrainer(model=load_model(source, train=True), args=args, train_dataset=Dataset.from_list(rows),
                          reward_funcs=[RewardLog(out / "reward_log.jsonl", config.num_generations)],
                          processing_class=tokenizer, peft_config=lora_config(config), callbacks=[stop])
    if not any(r["step"] == 0 for r in stop.checks_log.rows()):
        # The starting model on the training path (text prompts through TRL's vLLM), once. The rule compares
        # with the pre-pass rate (the chat endpoint, as at evaluation); a gap here means the paths differ.
        start = accept_fn(0)
        stop.checks_log.append({"step": 0, "accept_right": start, "mean_reward": None, "all_same": None,
                                "start_accept": stop.start_accept, "ts": now_iso()})
        if abs(start - stop.start_accept) > config.stop.contrarian_drop:
            print(f"WARNING: held-out accept-when-right is {start:.3f} on the training path but "
                  f"{stop.start_accept:.3f} in the pre-pass; check the prompt rendering before trusting the stop rule")
    resume = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out))
    return out, stop.reason


def _paths(config: GrpoConfig) -> str:
    """Shell assignments for scripts/run_e109.sh."""
    vals = {"INIT_FROM": config.init_from, "OUT_DIR": config.output_dir, "MERGED_DIR": config.merged_dir,
            "PREPASS_DIR": config.prepass_dir, "PREPASS_NAME": config.prepass.served_name, "SERVED_NAME": config.name,
            "VLLM_MODE": config.vllm_mode, "TRL_PORT": config.vllm_server_port}
    return "\n".join(f"{k}={json.dumps(str(v))}" for k, v in vals.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="validate the config and prompt files only (no torch)")
    parser.add_argument("--prepass", action="store_true", help="sample the starting model and write the training set")
    parser.add_argument("--merge", action="store_true", help="merge the trained adapter into a bf16 checkpoint")
    parser.add_argument("--paths", action="store_true", help="print the run's paths as shell assignments")
    args = parser.parse_args()
    config = GrpoConfig.load(args.config)
    if args.paths:
        print(_paths(config))
        return
    stats = check(config)
    print(f"{config.name}: prompts {config.dataset}/{config.arm} OK: {stats}")
    if args.check:
        if not stats["init_from_set"]:
            print(f"note: init_from is still the placeholder {config.init_from!r}; the GPU steps will refuse it")
        return
    if args.prepass:
        from sysone.env import load_env

        load_env()
        require_start(config)
        summary = run_prepass(config, prepass_client(config))
        print(json.dumps({k: summary[k] for k in ("train", "heldout", "wall_hours")}, indent=2))
        return
    if args.merge:
        from sysone.posttrain.hf import save_bf16

        out = save_bf16(str(require_start(config)), config.merged_dir, adapter=config.output_dir)
        write_info(out, config, "merge", stats)  # type: ignore[arg-type]
        print(f"merged into {out}")
        return
    out, reason = train(config)
    write_info(out, config, "grpo", {**stats, "stopped": reason})  # type: ignore[arg-type]
    print(f"adapter saved to {out} (stopped early: {reason})")


if __name__ == "__main__":
    main()
