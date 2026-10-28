"""TypeSafe's native System One API (used for Jev).

POST https://api.typesafe.ai/v1/systemone with a state and typed questions. Each answer
comes back typed: a choice answer has `choice`, `confidence` and `probabilities` (one per
option), a score answer adds `score` and `legend`, and a noul answer has `noul`, the
probability of yes. The response's `model` field names the exact version that answered,
even when the request used an alias such as jev-latest.

Only input tokens are billed; output tokens are currently free. Retries are left to
GuardedClient, so every attempt goes through the cost guard.
"""

from __future__ import annotations

import json
import os

import httpx

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ProviderResult, Request, SystemOneRequest, Usage

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
# 429 is a rate limit and 529 means TypeSafe is overloaded; both are worth retrying.
RETRYABLE_STATUS = {429, 529}


class TypeSafeProvider(Provider):
    name = "typesafe"

    def __init__(self, api_key: str | None = None, *, http: httpx.Client | None = None, timeout_s: float = 120):
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self._http = http or httpx.Client(timeout=timeout_s)

    @staticmethod
    def build_body(model: str, request: SystemOneRequest) -> dict:
        return {"model": model, "state": request.state, "questions": request.questions}

    def send(self, model: str, request: Request, price: Price) -> ProviderResult:
        if not isinstance(request, SystemOneRequest):
            raise TypeError("TypeSafe takes SystemOneRequest (typed questions), not chat messages")
        try:
            resp = self._http.post(
                TYPESAFE_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self.build_body(model, request),
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"typesafe timeout: {e}", retryable=True, billed="unknown") from e
        except httpx.TransportError as e:
            raise ProviderError(f"typesafe transport error: {e}", retryable=True, billed="unknown") from e

        if resp.status_code != 200:
            retryable = resp.status_code in RETRYABLE_STATUS or resp.status_code >= 500
            raise ProviderError(
                f"typesafe HTTP {resp.status_code}: {resp.text[:500]}", retryable=retryable, billed="no"
            )
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"typesafe returned non-JSON: {resp.text[:200]}", retryable=False, billed="unknown") from e

        # The version is required: it is how every result is tied to one model.
        version = data.get("model")
        if not isinstance(version, str) or not version:
            raise ProviderError("typesafe response has no model version", retryable=False, billed="unknown")
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ProviderError("typesafe response has no answers", retryable=False, billed="unknown")

        usage = data.get("usage") or {}
        return ProviderResult(
            text=json.dumps(answers, sort_keys=True),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            model_used=version,
            finish_reason=None,
            raw=data,
        )
