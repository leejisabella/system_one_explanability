"""Amazon Bedrock adapter (bedrock-runtime): Claude through InvokeModel, other models through Converse.

Claude uses boto3 InvokeModel with an explicit anthropic_version in the request body.
Anthropic's docs note that recent Anthropic SDK releases send an API version that the
ARN-versioned Bedrock models (such as Claude Opus 4.5) reject; boto3 is not affected.

Every other model (gpt-oss, Qwen, Mistral, Llama, Nova) goes through the Converse API:
- maxTokens caps all generated tokens. For gpt-oss it maps to OpenAI's
  max_completion_tokens, which includes reasoning, so the reasoning budget is added to it.
- reasoning_effort is sent in additionalModelRequestFields, for OpenAI models only.
- service_tier ("flex") is sent as serviceTier, and the tier that served the request is
  read back from the response, so the cost guard charges the flex price only when Bedrock
  confirms it.
- Structured output isn't used: callers put the JSON format in the prompt and parse the text.
"""

from __future__ import annotations

import json
from typing import Any

from sysone.budget.pricing import Price
from sysone.clients.base import Provider, ProviderError
from sysone.types import ChatRequest, ProviderResult, Usage

DEFAULT_REGION = "us-east-2"  # Ohio
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Bedrock error codes that mean "try again later" rather than "fix the request".
_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException", "ModelNotReadyException"}
# Inference-profile IDs carry a routing prefix: "us.meta....", "global.anthropic....".
_ROUTING_PREFIXES = {"us", "eu", "apac", "global"}


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, region: str = DEFAULT_REGION, *, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config

            # One attempt per request: retries are handled by GuardedClient so each passes the cost guard.
            # (In botocore's legacy retry mode, max_attempts counts *extra* attempts.)
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(retries={"total_max_attempts": 1, "mode": "standard"}, read_timeout=300),
            )
        self._client = client

    def build_body(self, model: str, request: ChatRequest) -> dict:
        """InvokeModel body for a Claude model."""
        if not _is_anthropic(model):
            raise ProviderError(f"build_body is for Claude models; {model} goes through Converse",
                                retryable=False, billed="no")
        if request.json_schema is not None:
            raise ProviderError("structured output is not wired up for Bedrock yet", retryable=False, billed="no")
        if request.service_tier is not None:
            raise ProviderError("service tiers are only wired up for Converse models", retryable=False, billed="no")
        if request.reasoning_effort is not None:
            raise ProviderError("Claude takes a thinking budget (reasoning_max_tokens), not a reasoning effort",
                                retryable=False, billed="no")
        body: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": request.max_output_tokens + (request.reasoning_max_tokens or 0),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            body["system"] = request.system
        if request.temperature is not None and request.reasoning_max_tokens:
            raise ProviderError("Claude does not accept a temperature with thinking enabled", retryable=False, billed="no")
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.reasoning_max_tokens:
            # Claude uses a fixed thinking budget, which must sit inside max_tokens.
            body["thinking"] = {"type": "enabled", "budget_tokens": request.reasoning_max_tokens}
        return body

    def build_converse(self, model: str, request: ChatRequest) -> dict:
        """Converse arguments for a non-Claude model."""
        if _is_anthropic(model):
            raise ProviderError("Claude models go through InvokeModel", retryable=False, billed="no")
        if request.json_schema is not None:
            raise ProviderError("structured output is not wired up for Bedrock yet", retryable=False, billed="no")
        if request.seed is not None:
            raise ProviderError("Converse has no seed parameter", retryable=False, billed="no")
        family = _family(model)
        if request.reasoning_max_tokens and family != "openai":
            raise ProviderError(f"{model} takes no reasoning budget", retryable=False, billed="no")
        if request.reasoning_effort is not None and family != "openai":
            raise ProviderError(f"reasoning_effort is only wired up for OpenAI models, not {model}",
                                retryable=False, billed="no")
        inference: dict = {"maxTokens": request.max_output_tokens + (request.reasoning_max_tokens or 0)}
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        kwargs: dict = {
            "modelId": model,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in request.messages],
            "inferenceConfig": inference,
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]
        if request.reasoning_effort is not None:
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": request.reasoning_effort}
        if request.service_tier is not None:
            kwargs["serviceTier"] = {"type": request.service_tier}
        return kwargs

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            ConnectionError,
            ParamValidationError,
            ReadTimeoutError,
        )

        anthropic = _is_anthropic(model)
        args = self.build_body(model, request) if anthropic else self.build_converse(model, request)
        try:
            if anthropic:
                resp = self._client.invoke_model(modelId=model, body=json.dumps(args))
            else:
                resp = self._client.converse(**args)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            raise ProviderError(f"bedrock {code}: {e}", retryable=code in _RETRYABLE, billed="no") from e
        except ParamValidationError as e:
            # Rejected by boto3 before anything was sent.
            raise ProviderError(f"bedrock request invalid: {e}", retryable=False, billed="no") from e
        except (ReadTimeoutError, ConnectionError) as e:
            raise ProviderError(f"bedrock connection error: {e}", retryable=True, billed="unknown") from e
        except BotoCoreError as e:
            raise ProviderError(f"bedrock error: {e}", retryable=False, billed="unknown") from e
        return _parse_invoke(model, resp) if anthropic else _parse_converse(model, resp)


def _parse_invoke(model: str, resp: dict) -> ProviderResult:
    data = json.loads(resp["body"].read())
    usage = data.get("usage", {})
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    return ProviderResult(
        text=text,
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            # Claude bills thinking as output; output_tokens already includes it.
            output_tokens=usage.get("output_tokens", 0),
        ),
        model_used=data.get("model", model),
        finish_reason=data.get("stop_reason"),
        raw=data,
    )


def _parse_converse(model: str, resp: dict) -> ProviderResult:
    data = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
    blocks = ((data.get("output") or {}).get("message") or {}).get("content") or []
    # Reasoning arrives in separate reasoningContent blocks; only the answer text is kept.
    text = "".join(block["text"] for block in blocks if "text" in block)
    usage = data.get("usage") or {}
    return ProviderResult(
        text=text,
        usage=Usage(
            input_tokens=usage.get("inputTokens", 0)
            + usage.get("cacheReadInputTokens", 0)
            + usage.get("cacheWriteInputTokens", 0),
            # Includes reasoning tokens, which Bedrock bills as output.
            output_tokens=usage.get("outputTokens", 0),
        ),
        # Converse doesn't name a model version; the requested ID is fixed, not an alias.
        model_used=model,
        finish_reason=data.get("stopReason"),
        raw=data,
        service_tier=(data.get("serviceTier") or {}).get("type"),
    )


def _family(model: str) -> str:
    parts = model.split(".")
    return parts[1] if parts[0] in _ROUTING_PREFIXES and len(parts) > 1 else parts[0]


def _is_anthropic(model: str) -> bool:
    return _family(model) == "anthropic"
