"""Self-hosted models behind an OpenAI-compatible chat-completions server (vLLM), for Part 2.

A post-trained reviewer can't run on Bedrock's shared gpt-oss endpoint, so Part 2 serves
models on its own GPU (research_docs/14). vLLM exposes /v1/chat/completions; this adapter
sends the harness's ChatRequest there.

- The server's URL comes from LOCAL_LLM_BASE_URL (default http://localhost:8000/v1), and an
  optional LOCAL_LLM_API_KEY is sent as a bearer token (vLLM's --api-key).
- The model ID in configs/models.yaml is the server's --served-model-name, so one server can
  hold the base model and each trained checkpoint under its own name.
- max_tokens caps all generated tokens, reasoning included (as on Bedrock), so the reasoning
  budget is added to it.
- reasoning_effort is sent as is; vLLM applies it through gpt-oss's chat template.
- vLLM returns the reasoning in message.reasoning_content (older releases) or
  message.reasoning (newer); reasoning_text reads either.
- Calls cost nothing per token (pricing.yaml lists local models at $0). GPU hours are paid
  outside the harness and recorded in each run's manifest by hand.
"""

from __future__ import annotations

import os

import httpx

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ChatRequest, ProviderResult, Usage

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class LocalProvider(Provider):
    name = "local"

    def __init__(self, base_url: str | None = None, *, api_key: str | None = None,
                 http: httpx.Client | None = None, timeout_s: float = 600):
        self.base_url = (base_url or os.environ.get("LOCAL_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")
        self._http = http or httpx.Client(timeout=timeout_s)

    def build_body(self, model: str, request: ChatRequest) -> dict:
        if request.service_tier is not None:
            raise ProviderError("a self-hosted server has no service tiers", retryable=False, billed="no")
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        body: dict = {"model": model, "messages": messages,
                      "max_tokens": request.max_output_tokens + (request.reasoning_max_tokens or 0)}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.seed is not None:
            body["seed"] = request.seed
        if request.reasoning_effort is not None:
            body["reasoning_effort"] = request.reasoning_effort
        if request.json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "strict": True, "schema": request.json_schema}}
        return body

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", headers=headers,
                                   json=self.build_body(model, request))
        except httpx.TimeoutException as e:
            raise ProviderError(f"local server timeout: {e}", retryable=True, billed="no") from e
        except httpx.TransportError as e:
            raise ProviderError(f"local server unreachable at {self.base_url}: {e}", retryable=True, billed="no") from e
        if resp.status_code != 200:
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise ProviderError(f"local HTTP {resp.status_code}: {resp.text[:500]}", retryable=retryable, billed="no")
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ProviderResult(
            text=choice["message"].get("content") or "",
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0),
                        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            model_used=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )


def reasoning_text(raw: dict) -> str | None:
    """The reasoning an OpenAI-compatible response carried (vLLM's reasoning_content or reasoning),
    or None."""
    choices = (raw or {}).get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("reasoning_content") or message.get("reasoning")
    return text or None
