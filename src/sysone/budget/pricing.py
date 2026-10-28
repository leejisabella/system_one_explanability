"""Price table and conservative worst-case cost estimation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from sysone.types import Request, SystemOneRequest, Usage


class UnknownPrice(KeyError):
    """Raised for a model with no price entry. Unpriced models are never called."""


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_m + usage.output_tokens * self.output_per_m
        ) / 1_000_000


class PriceTable:
    """Standard prices per provider and model, plus any service-tier prices (e.g. Bedrock's
    "flex"), listed under the tier's name in the model's entry."""

    def __init__(self, prices: dict[str, dict[str, Price]], tiers: dict[str, dict[str, dict[str, Price]]] | None = None):
        self._prices = prices
        self._tiers = tiers or {}

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        data = yaml.safe_load(Path(path).read_text()) or {}
        prices: dict[str, dict[str, Price]] = {}
        tiers: dict[str, dict[str, dict[str, Price]]] = {}
        for provider, models in data.items():
            prices[provider], tiers[provider] = {}, {}
            for model, p in (models or {}).items():
                prices[provider][model] = Price(float(p["input"]), float(p["output"]))
                tiers[provider][model] = {
                    tier: Price(float(t["input"]), float(t["output"]))
                    for tier, t in p.items() if isinstance(t, dict)
                }
        return cls(prices, tiers)

    def lookup(self, provider: str, model: str) -> Price:
        try:
            return self._prices[provider][model]
        except KeyError:
            raise UnknownPrice(f"no price for {provider}/{model} in pricing table") from None

    def lookup_tier(self, provider: str, model: str, tier: str) -> Price:
        try:
            return self._tiers[provider][model][tier]
        except KeyError:
            raise UnknownPrice(f"no {tier} price for {provider}/{model} in pricing table") from None


# Real tokenizers average ~4 characters per English token. Dividing by 3
# overestimates input size, which keeps the pre-call worst case conservative.
_CHARS_PER_TOKEN = 3
_PER_MESSAGE_OVERHEAD = 10
_PER_REQUEST_OVERHEAD = 50


def estimate_input_tokens(request: Request) -> int:
    if isinstance(request, SystemOneRequest):
        # State, question names, instructions and criteria are all billed as input.
        chars = len(json.dumps(request.state)) + len(json.dumps(request.questions))
        return math.ceil(chars / _CHARS_PER_TOKEN) + _PER_REQUEST_OVERHEAD
    chars = len(request.system or "") + sum(len(m.content) for m in request.messages)
    if request.json_schema is not None:
        chars += len(str(request.json_schema))
    return (
        math.ceil(chars / _CHARS_PER_TOKEN)
        + _PER_MESSAGE_OVERHEAD * len(request.messages)
        + _PER_REQUEST_OVERHEAD
    )


def worst_case_cost(request: Request, price: Price, reasoning_allowance: int = 0) -> float:
    """Upper bound on what one request can cost, assuming every output token is used.

    reasoning_allowance is the reasoning-token bound assumed when the request sets no
    reasoning_max_tokens, for providers whose models may reason anyway.
    """
    if isinstance(request, SystemOneRequest):
        return price.cost(Usage(input_tokens=estimate_input_tokens(request), output_tokens=request.max_output_tokens))
    reasoning = request.reasoning_max_tokens if request.reasoning_max_tokens is not None else reasoning_allowance
    max_output = request.max_output_tokens + reasoning
    return price.cost(Usage(input_tokens=estimate_input_tokens(request), output_tokens=max_output))
