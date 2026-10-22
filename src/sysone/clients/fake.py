"""Offline provider for tests: returns scripted results or raises scripted errors."""

from __future__ import annotations

from collections.abc import Iterable

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ChatRequest, ProviderResult, Usage


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, script: Iterable[ProviderResult | ProviderError] | None = None):
        self._script = list(script or [])
        self.calls: list[tuple[str, ChatRequest]] = []

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        self.calls.append((model, request))
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, ProviderError):
                raise item
            return item
        return ok_result(f"echo: {request.messages[-1].content}", model=model)


class FakeDecisionProvider(Provider):
    """Stands in for Jev in dry runs: random but repeatable decisions, no network, no cost.

    Decisions depend only on a hash of the prompt and sample index, so they carry no
    signal about the case. A small share of replies are malformed, and replies
    alternate between two "routed" model names, to exercise parsing and routing stats.
    """

    name = "fake"

    def __init__(self, *, malformed_every: int = 50):
        self.malformed_every = malformed_every
        self.calls = 0

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        import hashlib

        self.calls += 1
        digest = hashlib.sha256(f"{request.messages[-1].content}|{request.sample_index}".encode()).digest()
        if self.malformed_every and digest[0] % self.malformed_every == 0:
            text = "I would approve this refund."
        else:
            label = "approve" if digest[1] % 2 else "deny"
            text = f'{{"decision": "{label}", "confidence": {0.5 + digest[2] / 510:.2f}}}'
        return ok_result(text, model=f"fake/jev-{'ab'[digest[3] % 2]}", input_tokens=600, output_tokens=12)


def ok_result(
    text: str = "ok",
    *,
    model: str = "fake-model",
    input_tokens: int = 100,
    output_tokens: int = 20,
    reported_cost_usd: float | None = None,
) -> ProviderResult:
    return ProviderResult(
        text=text,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_used=model,
        finish_reason="stop",
        raw={},
        reported_cost_usd=reported_cost_usd,
    )
