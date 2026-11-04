"""Hugging Face helpers shared by the SFT and DPO scripts. Needs the `train` dependency group (GPU).

Records are rendered to text here with the reviewer's own chat template, rather than left to TRL,
so reasoning is switched on explicitly and the prompt is exactly what vLLM will render at
evaluation (vLLM uses the same template). The completion is the full rendering minus the
prompt's, so the loss falls on the reviewer's turn only.

The records store the reasoning as "thinking". Each family's template reads it from its own key
(train_config.FAMILIES): Qwen3's from "reasoning_content", which it renders as <think>...</think>
before the answer; gpt-oss's from "thinking", which it renders into the analysis channel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _for_template(turns: list[dict], thinking_key: str) -> list[dict]:
    if thinking_key == "thinking":
        return turns
    return [{(thinking_key if k == "thinking" else k): v for k, v in t.items()} for t in turns]


def render(tokenizer: Any, prompt: list[dict], completion: list[dict], *, template_kwargs: dict[str, Any],
           thinking_key: str) -> tuple[str, str]:
    """(prompt text, completion text) for one conversation, with the completion's thinking kept."""
    head = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **template_kwargs)
    full = tokenizer.apply_chat_template(prompt + _for_template(completion, thinking_key), tokenize=False,
                                         **template_kwargs)
    if not full.startswith(head):
        raise ValueError("the chat template doesn't render the prompt as a prefix of the conversation; "
                         "check the tokenizer's template before training")
    body = full[len(head):]
    # Qwen3's template strips newlines around the reasoning, so compare it stripped.
    if completion[0].get("thinking") and completion[0]["thinking"].strip() not in body:
        raise ValueError("the chat template dropped the reasoning from the completion; check thinking_key")
    return head, body


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_model(path: str, *, train: bool, attn_implementation: str | None = None):
    """The model in bf16. A quantized Hub checkpoint (gpt-oss's MXFP4) is dequantized on load;
    a bf16 one (Qwen3, or any saved or merged checkpoint) loads as it is."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    quantization_config = None
    if getattr(AutoConfig.from_pretrained(path), "quantization_config", None) is not None:
        from transformers import Mxfp4Config

        quantization_config = Mxfp4Config(dequantize=True)
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config,
        attn_implementation=attn_implementation if train else None, use_cache=not train)


def save_bf16(model_path: str, out: Path, *, adapter: Path | None = None) -> Path:
    """Save the model in bf16, with a LoRA adapter merged in if given. With no adapter this is the
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
                      target_parameters=config.lora_target_parameters() or None, task_type="CAUSAL_LM")
