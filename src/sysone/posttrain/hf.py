"""Hugging Face helpers shared by the SFT and DPO scripts. Needs the `train` dependency group (GPU).

Records are rendered to text here with gpt-oss's own chat template, rather than left to TRL, so
the reasoning level is set explicitly and the prompt is exactly what vLLM will render at
evaluation (vLLM uses the same template). The completion is the full rendering minus the
prompt's, so the loss falls on the reviewer's turn only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render(tokenizer: Any, prompt: list[dict], completion: list[dict], *, reasoning_effort: str) -> tuple[str, str]:
    """(prompt text, completion text) for one conversation, with the completion's thinking kept."""
    head = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                         reasoning_effort=reasoning_effort)
    full = tokenizer.apply_chat_template(prompt + completion, tokenize=False, reasoning_effort=reasoning_effort)
    if not full.startswith(head):
        raise ValueError("the chat template doesn't render the prompt as a prefix of the conversation; "
                         "check the tokenizer's template before training")
    return head, full[len(head):]


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_model(path: str, *, train: bool):
    """gpt-oss in bf16. The Hub checkpoint is MXFP4 and is dequantized on load; a saved bf16
    checkpoint (the base saved by --save-base, or a merged one) loads as it is."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Mxfp4Config

    quantized = getattr(AutoConfig.from_pretrained(path), "quantization_config", None) is not None
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=Mxfp4Config(dequantize=True) if quantized else None,
        attn_implementation="eager" if train else None, use_cache=not train)


def save_bf16(model_path: str, out: Path, *, adapter: Path | None = None) -> Path:
    """Save gpt-oss in bf16, with a LoRA adapter merged in if given. With no adapter this is the
    evaluation's base model: the same precision as every trained checkpoint (research_docs/14)."""
    from transformers import AutoTokenizer

    model = load_model(model_path, train=False)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out)
    return out


def lora_config(config: Any):
    from peft import LoraConfig

    lora = config.lora
    return LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, target_modules=lora.target_modules,
                      target_parameters=lora.target_parameters or None, task_type="CAUSAL_LM")
