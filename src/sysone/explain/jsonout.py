"""Getting JSON out of chat models, with or without provider-side structured output."""

from __future__ import annotations

import json
from typing import Any

# Providers whose adapters pass ChatRequest.json_schema through. Bedrock's doesn't yet,
# so Bedrock models get the format in the prompt and their text is parsed here.
SCHEMA_PROVIDERS = frozenset({"openrouter", "gemini", "fake"})


def uses_schema(provider: str) -> bool:
    return provider in SCHEMA_PROVIDERS


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
