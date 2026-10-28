"""Getting JSON out of chat models, with or without provider-side structured output."""

from __future__ import annotations

import json
import re
from typing import Any

# Providers whose adapters pass ChatRequest.json_schema through. Bedrock's doesn't yet,
# so Bedrock models get the format in the prompt and their text is parsed here.
SCHEMA_PROVIDERS = frozenset({"openrouter", "gemini", "fake"})


def uses_schema(provider: str) -> bool:
    return provider in SCHEMA_PROVIDERS


_DIGIT_WORDS = {w: str(i) for i, w in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"))}
# A decimal whose digit is spelled out, e.g. "0. nine" for 0.9: gpt-oss-120b wrote this in
# 87 of 669 simulator replies in pilot1-v1b-2026-10-28, which made them invalid JSON.
_SPELLED_DECIMAL = re.compile(r"(?<![\w.])(\d)\.\s*(" + "|".join(_DIGIT_WORDS) + r")\b")


def repair_spelled_digits(text: str) -> str:
    """Rewrite decimals with a spelled-out digit ("0. nine") as numbers ("0.9"). Nothing else changes."""
    return _SPELLED_DECIMAL.sub(lambda m: f"{m.group(1)}.{_DIGIT_WORDS[m.group(2)]}", text)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in a model's reply, tolerating code fences and text around it."""
    text = text.strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
