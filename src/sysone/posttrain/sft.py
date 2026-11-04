"""E.106: LoRA supervised fine-tuning of gpt-oss-20b on E.105's traces (research_docs/14).

Usage (on the GPU machine, with `uv sync --group train`):
  uv run python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml --check   # no GPU needed
  uv run --group train python -m sysone.posttrain.sft --save-base checkpoints/gpt-oss-20b-base
  uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml
  uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml --merge

--check validates the config and dataset without torch. --save-base writes the dequantized bf16
base the evaluation compares against. Training writes the LoRA adapter to checkpoints/<name>;
--merge then writes checkpoints/<name>-merged for vLLM to serve under --served-model-name <name>.
Every step writes train_info.json next to its output, with the config and the dataset's hashes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from sysone.posttrain.train_config import TrainConfig, check
from sysone.runs import git_info, now_iso


def train(config: TrainConfig) -> Path:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from sysone.posttrain.hf import load_model, load_records, lora_config, render

    source = config.init_from or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(source)

    def dataset(path: Path) -> Dataset:
        rows = []
        for r in load_records(path):
            prompt, completion = render(tokenizer, r["prompt"], r["completion"], reasoning_effort=config.reasoning_effort)
            rows.append({"prompt": prompt, "completion": completion})
        return Dataset.from_list(rows)

    files = config.files()
    train_ds = dataset(files["train"])
    val_ds = dataset(files["val"]) if files["val"].stat().st_size else None
    args = SFTConfig(
        output_dir=str(config.output_dir), learning_rate=config.learning_rate, num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1, per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, gradient_checkpointing=True,
        max_length=config.max_length, warmup_ratio=config.warmup_ratio, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1}, bf16=True, logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no", seed=config.seed, report_to="none",
        completion_only_loss=True,
    )
    trainer = SFTTrainer(model=load_model(source, train=True), args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=lora_config(config))
    trainer.train()
    trainer.save_model(str(config.output_dir))
    return config.output_dir


def write_info(out: Path, config: TrainConfig | None, step: str, stats: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = {"step": step, "finished_at": now_iso(), "git": git_info(),
            "config": dataclasses.asdict(config) if config else None, "dataset_check": stats}
    (out / "train_info.json").write_text(json.dumps(info, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--check", action="store_true", help="validate the config and dataset only (no torch)")
    parser.add_argument("--merge", action="store_true", help="merge the trained adapter into a bf16 checkpoint")
    parser.add_argument("--save-base", metavar="DIR", help="save the dequantized bf16 base model to DIR and exit")
    parser.add_argument("--base-model", default="openai/gpt-oss-20b", help="with --save-base")
    args = parser.parse_args()

    if args.save_base:
        from sysone.posttrain.hf import save_bf16

        out = save_bf16(args.base_model, Path(args.save_base))
        write_info(out, None, f"save_base {args.base_model}")
        print(f"saved {out}")
        return
    if not args.config:
        parser.error("--config is required unless --save-base")
    config = TrainConfig.load(args.config)
    stats = check(config)
    print(f"{config.name}: dataset {config.dataset}/{config.arm} OK: {stats}")
    if args.check:
        return
    if args.merge:
        from sysone.posttrain.hf import save_bf16

        out = save_bf16(config.init_from or config.base_model, config.merged_dir, adapter=config.output_dir)
        write_info(out, config, "merge", stats)
        print(f"merged into {out}")
        return
    out = train(config)
    write_info(out, config, "sft", stats)
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
