"""Google Gemini adapter (google-genai SDK; Gemini API key or Vertex AI)."""

from __future__ import annotations

import os
from typing import Any

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ChatRequest, ProviderResult, Usage


# Output-cap headroom for thinking when a request sets a thinking level but no budget.
# Gemini counts thinking against max_output_tokens, so without it a short reply (such as
# a 64-token decision) can be cut off or come back empty. The worst-case estimate already
# assumes reasoning_allowance_tokens of thinking, which is larger.
THINKING_LEVEL_HEADROOM_TOKENS = 1024


class GeminiProvider(Provider):
    name = "gemini"
    # Gemini thinking models think by default; 32,768 is the largest Gemini 2.5 thinking budget.
    reasoning_allowance_tokens = 32_768

    def __init__(self, *, client: Any = None, vertex_project: str | None = None, vertex_location: str = "global"):
        if client is None:
            from google import genai
            from google.genai import types

            # One attempt per request: retries are handled by GuardedClient so each passes the cost guard.
            http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
            if vertex_project:
                client = genai.Client(
                    vertexai=True, project=vertex_project, location=vertex_location, http_options=http_options
                )
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY is not set (or pass vertex_project)")
                client = genai.Client(api_key=api_key, http_options=http_options)
        self._client = client

    def build_config(self, request: ChatRequest) -> Any:
        from google.genai import types

        thinking = request.reasoning_max_tokens
        if thinking is None and request.reasoning_effort is not None:
            thinking = THINKING_LEVEL_HEADROOM_TOKENS
        kwargs: dict[str, Any] = {
            # Gemini counts thinking tokens against max_output_tokens, so the total cap
            # is visible output plus the reasoning budget (or headroom for a thinking level).
            "max_output_tokens": request.max_output_tokens + (thinking or 0),
        }
        if request.system:
            kwargs["system_instruction"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.reasoning_effort is not None:
            # Gemini 3 models take a thinking level ("minimal", "low", ...); a budget is still
            # added to max_output_tokens above if one is set, since thinking counts against it.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=request.reasoning_effort.upper())
        elif request.reasoning_max_tokens is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=request.reasoning_max_tokens)
        if request.service_tier is not None:
            raise ProviderError("service tiers are not wired up for Gemini", retryable=False, billed="no")
        if request.json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = request.json_schema
        return types.GenerateContentConfig(**kwargs)

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        import httpx
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in request.messages
        ]
        try:
            resp = self._client.models.generate_content(model=model, contents=contents, config=self.build_config(request))
        except errors.ClientError as e:
            raise ProviderError(f"gemini {e.code}: {e}", retryable=e.code == 429, billed="no") from e
        except errors.ServerError as e:
            raise ProviderError(f"gemini {e.code}: {e}", retryable=True, billed="no") from e
        except errors.APIError as e:
            raise ProviderError(f"gemini error: {e}", retryable=False, billed="unknown") from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise ProviderError(f"gemini connection error: {e}", retryable=True, billed="unknown") from e

        meta = resp.usage_metadata
        thoughts = (meta.thoughts_token_count or 0) if meta else 0
        candidates = (meta.candidates_token_count or 0) if meta else 0
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        return ProviderResult(
            text=resp.text or "",
            usage=Usage(
                input_tokens=(meta.prompt_token_count or 0) if meta else 0,
                # candidates_token_count excludes thinking; both are billed as output.
                output_tokens=candidates + thoughts,
                reasoning_tokens=thoughts,
            ),
            model_used=getattr(resp, "model_version", None) or model,
            finish_reason=str(finish) if finish is not None else None,
            raw=resp.model_dump(mode="json", exclude_none=True),
        )
