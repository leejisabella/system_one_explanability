"""E.107: DPO of the SFT reviewer on E.105's preference pairs (research_docs/14).

Each pair is the reviewer's own correct trace (chosen) against its own wrong one (rejected) for
the same prompt; in review, the wrong one is preferably a trace that deferred to a wrong proposal.
Training starts from the merged SFT checkpoint (init_from) with a fresh LoRA adapter; the
reference model is that checkpoint with the adapter switched off.

Usage (on the GPU machine):
  uv run python -m sysone.posttrain.dpo --config configs/posttrain/dpo_review.yaml --check
  uv run --group train python -m sysone.posttrain.dpo --config configs/posttrain/dpo_review.yaml
  uv run --group train python -m sysone.posttrain.dpo --config configs/posttrain/dpo_review.yaml --merge
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sysone.posttrain.sft import write_info
from sysone.posttrain.train_config import TrainConfig, check


def train(config: TrainConfig) -> Path:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from sysone.posttrain.hf import load_model, load_records, lora_config, render

    source = config.init_from or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(source)
    fmt = {"template_kwargs": config.template_kwargs(), "thinking_key": config.model_family.thinking_key}

    def dataset(path: Path) -> Dataset:
        rows = []
        for r in load_records(path):
            prompt, chosen = render(tokenizer, r["prompt"], r["chosen"], **fmt)
            _, rejected = render(tokenizer, r["prompt"], r["rejected"], **fmt)
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return Dataset.from_list(rows)

    files = config.files()
    train_ds = dataset(files["train"])
    val_ds = dataset(files["val"]) if files["val"].stat().st_size else None
    args = DPOConfig(
        output_dir=str(config.output_dir), beta=config.dpo_beta, learning_rate=config.learning_rate,
        num_train_epochs=config.epochs, max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, gradient_checkpointing=True,
        max_length=config.max_length, warmup_ratio=config.warmup_ratio, bf16=True, logging_steps=10,
        save_strategy="epoch", eval_strategy="epoch" if val_ds is not None else "no", seed=config.seed, report_to="none",
    )
    model = load_model(source, train=True, attn_implementation=config.model_family.train_attn_implementation)
    trainer = DPOTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=lora_config(config))
    trainer.train()
    trainer.save_model(str(config.output_dir))
    return config.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="validate the config and dataset only (no torch)")
    parser.add_argument("--merge", action="store_true", help="merge the trained adapter into a bf16 checkpoint")
    args = parser.parse_args()
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
    write_info(out, config, "dpo", stats)
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
