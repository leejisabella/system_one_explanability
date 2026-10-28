"""OpenRouter chat-completions adapter. No model uses it at the moment; kept for later experiments."""

from __future__ import annotations

import os

import httpx

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ChatRequest, ProviderResult, Usage

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(Provider):
    name = "openrouter"
    # Routed models may pick a reasoning effort on their own.
    reasoning_allowance_tokens = 32_768

    def __init__(self, api_key: str | None = None, *, http: httpx.Client | None = None, timeout_s: float = 120):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._http = http or httpx.Client(timeout=timeout_s)

    def build_body(self, model: str, request: ChatRequest, price: Price) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            # OpenRouter refuses to route to any endpoint priced above this ($ per 1M tokens),
            # which keeps dynamically priced models within our worst-case estimate.
            "provider": {"max_price": {"prompt": price.input_per_m, "completion": price.output_per_m}},
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.seed is not None:
            body["seed"] = request.seed
        if request.reasoning_effort is not None:
            body["reasoning"] = {"effort": request.reasoning_effort}
        elif request.reasoning_max_tokens is not None:
            body["reasoning"] = {"max_tokens": request.reasoning_max_tokens}
        if request.service_tier is not None:
            raise ProviderError("service tiers are not wired up for OpenRouter", retryable=False, billed="no")
        if request.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": request.json_schema},
            }
        return body

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        try:
            resp = self._http.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self.build_body(model, request, price),
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"openrouter timeout: {e}", retryable=True, billed="unknown") from e
        except httpx.TransportError as e:
            raise ProviderError(f"openrouter transport error: {e}", retryable=True, billed="unknown") from e

        if resp.status_code != 200:
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise ProviderError(
                f"openrouter HTTP {resp.status_code}: {resp.text[:500]}", retryable=retryable, billed="no"
            )
        data = resp.json()
        if "error" in data:
            raise ProviderError(f"openrouter error: {data['error']}", retryable=False, billed="unknown")

        usage = data.get("usage") or {}
        choice = data["choices"][0]
        return ProviderResult(
            text=choice["message"].get("content") or "",
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            ),
            model_used=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
            raw=data,
            # OpenRouter reports the amount charged, in credits (1 credit = $1).
            reported_cost_usd=usage.get("cost"),
        )
