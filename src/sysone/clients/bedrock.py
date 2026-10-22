"""Amazon Bedrock adapter for Claude models (InvokeModel on bedrock-runtime).

Uses boto3 with an explicit anthropic_version in the request body. Anthropic's
docs note that recent Anthropic SDK releases send an API version that the
ARN-versioned Bedrock models (such as Claude Opus 4.5) reject; boto3 is not affected.
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


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, region: str = DEFAULT_REGION, *, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config

            # One attempt per request: retries are handled by GuardedClient so each passes the
            # cost guard. (In botocore's legacy retry mode, max_attempts counts *extra* attempts.)
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(retries={"total_max_attempts": 1, "mode": "standard"}, read_timeout=300),
            )
        self._client = client

    def build_body(self, model: str, request: ChatRequest) -> dict:
        if not _is_anthropic(model):
            # TODO: add a Converse-API path if we pick non-Claude Bedrock explainers.
            raise ProviderError(f"bedrock adapter only supports Claude models, got {model}", retryable=False, billed="no")
        if request.json_schema is not None:
            raise ProviderError("structured output is not wired up for Bedrock yet", retryable=False, billed="no")
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
            # Claude Opus 4.5 uses a fixed thinking budget, which must sit inside max_tokens.
            body["thinking"] = {"type": "enabled", "budget_tokens": request.reasoning_max_tokens}
        return body

    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        from botocore.exceptions import BotoCoreError, ClientError, ConnectionError, ReadTimeoutError

        body = self.build_body(model, request)
        try:
            resp = self._client.invoke_model(modelId=model, body=json.dumps(body))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            raise ProviderError(f"bedrock {code}: {e}", retryable=code in _RETRYABLE, billed="no") from e
        except (ReadTimeoutError, ConnectionError) as e:
            raise ProviderError(f"bedrock connection error: {e}", retryable=True, billed="unknown") from e
        except BotoCoreError as e:
            raise ProviderError(f"bedrock error: {e}", retryable=False, billed="unknown") from e

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


def _is_anthropic(model: str) -> bool:
    # Inference-profile IDs carry a routing prefix: "us.anthropic....", "global.anthropic....".
    return model.startswith("anthropic.") or ".anthropic." in model
