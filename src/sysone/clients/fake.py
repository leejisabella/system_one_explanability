"""Offline providers for tests and dry runs: scripted or hashed results, no network, no cost."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ProviderResult, Request, SystemOneRequest, Usage

FAKE_JEV_VERSION = "jev-fake-1.0.0"


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, script: Iterable[ProviderResult | ProviderError] | None = None):
        self._script = list(script or [])
        self.calls: list[tuple[str, Request]] = []

    def send(self, model: str, request: Request, price: Price) -> ProviderResult:
        self.calls.append((model, request))
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, ProviderError):
                raise item
            return item
        if isinstance(request, SystemOneRequest):
            return FakeDecisionProvider(malformed_every=0).send(model, request, price)
        return ok_result(f"echo: {request.messages[-1].content}", model=model)


class FakeDecisionProvider(Provider):
    """Stands in for Jev in dry runs, answering in TypeSafe's response format.

    Probabilities depend only on a hash of the request and sample index, so they carry
    no signal about the case. They sit on a 0.01 grid, like Jev's. A small share of
    answers are malformed (no probabilities), to exercise parsing. With
    `version_change_after=n`, every call after the n-th reports a different version,
    as happens when jev-latest moves to a new release.
    """

    name = "fake"

    def __init__(
        self,
        *,
        malformed_every: int = 50,
        version: str = FAKE_JEV_VERSION,
        version_change_after: int | None = None,
    ):
        self.malformed_every = malformed_every
        self.version = version
        self.version_change_after = version_change_after
        self.calls = 0

    def send(self, model: str, request: Request, price: Price) -> ProviderResult:
        if not isinstance(request, SystemOneRequest):
            raise TypeError("the fake Jev takes SystemOneRequest")
        self.calls += 1
        payload = json.dumps([request.state, request.questions, request.sample_index], sort_keys=True)
        digest = hashlib.sha256(payload.encode()).digest()
        malformed = bool(self.malformed_every) and digest[0] % self.malformed_every == 0
        answers = {name: _fake_answer(q, digest, malformed) for name, q in request.questions.items()}
        moved = self.version_change_after is not None and self.calls > self.version_change_after
        return typesafe_result(answers, model=f"{self.version}-next" if moved else self.version)


def _fake_answer(question: dict[str, Any], digest: bytes, malformed: bool) -> dict[str, Any]:
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round(digest[2] / 255, 2)}
    if kind != "choice":
        raise NotImplementedError(f"the fake Jev answers choice and noul questions, not {kind}")
    options = list(question["criteria"])
    if malformed:
        return {"type": "choice", "choice": options[0]}
    if len(options) == 1:
        return choice_answer({options[0]: 1.0})
    # Two decimals, like Jev; the last option takes the remainder so the sum is exactly 1.
    weights = [digest[2 + i % 28] + 1 for i in range(len(options))]
    probs = [round(w / sum(weights), 2) for w in weights[:-1]]
    probs.append(round(1 - sum(probs), 2))
    return choice_answer(dict(zip(options, probs)))


def choice_answer(probabilities: dict[str, float], *, confidence: float | None = None) -> dict[str, Any]:
    """A choice answer in TypeSafe's format. Confidence defaults to TypeSafe's formula:
    the top probability relative to an even split."""
    top = max(probabilities, key=probabilities.get)
    n = len(probabilities)
    if confidence is None:
        confidence = 1.0 if n == 1 else round((probabilities[top] - 1 / n) / (1 - 1 / n), 2)
    return {"type": "choice", "choice": top, "confidence": confidence, "probabilities": dict(probabilities)}


def typesafe_result(
    answers: dict[str, Any],
    *,
    model: str = FAKE_JEV_VERSION,
    input_tokens: int = 600,
    output_tokens: int = 0,
    reported_cost_usd: float | None = None,
) -> ProviderResult:
    """A ProviderResult shaped like TypeSafeProvider's."""
    raw = {"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
    return ProviderResult(
        text=json.dumps(answers, sort_keys=True),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_used=model,
        finish_reason=None,
        raw=raw,
        reported_cost_usd=reported_cost_usd,
    )


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
