"""Training configs for Part 2 (configs/posttrain/*.yaml) and the checks run before any GPU time.

`check` needs no GPU and no torch: it confirms the dataset files match their committed manifest
(benchmarks/posttrain/<dataset>/manifest.json), that every record has the shape the method needs,
and roughly how many records would be truncated at max_length.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sysone.registry import REPO_ROOT

# LoRA on gpt-oss-20b as in OpenAI's fine-tuning cookbook (2025): every linear layer, plus the
# expert weights of three of its 24 layers, which are parameters rather than linear modules.
GPT_OSS_EXPERT_PARAMETERS = [f"{layer}.mlp.experts.{w}" for layer in (7, 15, 23) for w in ("gate_up_proj", "down_proj")]
METHOD_ARMS = {"sft": ("sft_review", "sft_plain"), "dpo": ("dpo_review",)}
# A rough characters-per-token ratio for English text with gpt-oss's tokenizer, only for the check.
CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"
    target_parameters: list[str] = field(default_factory=lambda: list(GPT_OSS_EXPERT_PARAMETERS))


@dataclass(frozen=True)
class TrainConfig:
    # The run's name: the checkpoint folder and the vLLM --served-model-name.
    name: str
    method: str                       # "sft" or "dpo"
    dataset: str                      # data/posttrain/<dataset>, exported by E.105
    arm: str                          # which files of the dataset: sft_review, sft_plain or dpo_review
    base_model: str = "openai/gpt-oss-20b"
    # DPO starts from a merged SFT checkpoint (checkpoints/<sft name>-merged); None trains the base.
    init_from: str | None = None
    reasoning_effort: str = "medium"  # the reasoning level the prompts are rendered at, as at evaluation
    lora: LoraSettings = field(default_factory=LoraSettings)
    learning_rate: float = 2e-4
    epochs: float = 1.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    warmup_ratio: float = 0.03
    dpo_beta: float = 0.1
    seed: int = 106
    # A smoke test caps the steps (e.g. 5); None trains for `epochs`.
    max_steps: int | None = None
    checkpoints_dir: str = "checkpoints"

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**{**raw, "lora": LoraSettings(**raw.get("lora", {}))})

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / self.checkpoints_dir / self.name

    @property
    def merged_dir(self) -> Path:
        return REPO_ROOT / self.checkpoints_dir / f"{self.name}-merged"

    def files(self, data_dir: Path | None = None) -> dict[str, Path]:
        folder = (data_dir or REPO_ROOT / "data" / "posttrain") / self.dataset
        return {split: folder / f"{self.arm}.{split}.jsonl" for split in ("train", "val")}


def _record_problem(record: dict[str, Any], method: str) -> str | None:
    keys = ("prompt", "completion") if method == "sft" else ("prompt", "chosen", "rejected")
    if any(k not in record for k in keys):
        return f"missing one of {keys}"
    if not record["prompt"] or record["prompt"][-1]["role"] != "user":
        return "the prompt must end with a user turn"
    for k in keys[1:]:
        turn = record[k]
        if len(turn) != 1 or turn[0]["role"] != "assistant" or not turn[0].get("thinking") or not turn[0].get("content"):
            return f"{k} must be one assistant turn with thinking and content"
    return None


def check(config: TrainConfig, *, data_dir: Path | None = None, manifest_dir: Path | None = None) -> dict[str, Any]:
    """Problems that would waste GPU time, found without a GPU. Raises ValueError listing them."""
    problems: list[str] = []
    if config.method not in METHOD_ARMS:
        problems.append(f"method {config.method!r} is not one of {sorted(METHOD_ARMS)}")
    elif config.arm not in METHOD_ARMS[config.method]:
        problems.append(f"arm {config.arm!r} doesn't go with method {config.method} (use {METHOD_ARMS[config.method]})")
    if config.method == "dpo" and config.init_from is None:
        problems.append("DPO should start from a merged SFT checkpoint (init_from); set it, or null it deliberately")
    manifest_path = (manifest_dir or REPO_ROOT / "benchmarks" / "posttrain") / config.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None:
        problems.append(f"{manifest_path} is missing; export the dataset with E.105 first")
    stats: dict[str, Any] = {}
    for split, path in config.files(data_dir).items():
        if not path.exists():
            problems.append(f"{path} is missing")
            continue
        body = path.read_text()
        want = (manifest or {}).get("files", {}).get(path.name, {}).get("sha256")
        if manifest is not None and hashlib.sha256(body.encode()).hexdigest() != want:
            problems.append(f"{path.name} doesn't match its manifest hash; re-export it or fix the dataset name")
        records = [json.loads(line) for line in body.splitlines() if line.strip()]
        bad = [(i, p) for i, r in enumerate(records) if (p := _record_problem(r, config.method if config.method in METHOD_ARMS else "sft"))]
        if bad:
            problems.append(f"{path.name}: {len(bad)} malformed records, first #{bad[0][0]}: {bad[0][1]}")
        if split == "train" and not records:
            problems.append(f"{path.name} has no records")
        malformed = {i for i, _ in bad}
        lengths = [len(json.dumps(r["prompt"])) + max(len(json.dumps(r.get(k, ""))) for k in ("completion", "chosen", "rejected"))
                   for i, r in enumerate(records) if i not in malformed]
        over = sum(n / CHARS_PER_TOKEN > config.max_length for n in lengths)
        stats[split] = {"records": len(records), "estimated_over_max_length": over}
    if problems:
        raise ValueError("training config check failed:\n- " + "\n- ".join(problems))
    return stats
