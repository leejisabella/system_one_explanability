"""Request-building and response-parsing checks for each adapter, with no network access."""

import io
import json
from types import SimpleNamespace

import httpx
import pytest

from sysone.budget import Price
from sysone.clients.base import ProviderError
from sysone.clients.bedrock import BedrockProvider
from sysone.clients.gemini import GeminiProvider
from sysone.clients.openrouter import OpenRouterProvider
from sysone.types import ChatRequest

PRICE = Price(1.0, 2.0)


def req(**kw):
    return ChatRequest.from_prompt("hello", max_output_tokens=50, system="be brief", **kw)


def openrouter(handler):
    return OpenRouterProvider(api_key="test", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openrouter_sends_price_ceiling_and_parses_cost():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "typesafe/underlying",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "cost": 0.0003,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    result = openrouter(handler).send("typesafe/jev-router", req(reasoning_max_tokens=10), PRICE)
    body = seen["body"]
    assert body["provider"]["max_price"] == {"prompt": 1.0, "completion": 2.0}
    assert body["max_tokens"] == 50 and body["reasoning"] == {"max_tokens": 10}
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert result.reported_cost_usd == 0.0003
    assert result.model_used == "typesafe/underlying"
    assert result.usage.reasoning_tokens == 3


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False)])
def test_openrouter_http_errors(status, retryable):
    provider = openrouter(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ProviderError) as exc:
        provider.send("m", req(), PRICE)
    assert exc.value.retryable is retryable and exc.value.billed == "no"


def test_openrouter_timeout_is_unknown_billing():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as exc:
        openrouter(handler).send("m", req(), PRICE)
    assert exc.value.billed == "unknown"


class FakeBedrock:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def invoke_model(self, **kwargs):
        self.kwargs = kwargs
        return {"body": io.BytesIO(json.dumps(self.payload).encode())}


def test_bedrock_claude_request_and_usage():
    fake = FakeBedrock({
        "content": [{"type": "text", "text": "case text"}],
        "usage": {"input_tokens": 20, "output_tokens": 9},
        "stop_reason": "end_turn",
    })
    model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    result = BedrockProvider(client=fake).send(model, req(temperature=0.5), PRICE)
    body = json.loads(fake.kwargs["body"])
    assert fake.kwargs["modelId"] == model
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "be brief" and body["max_tokens"] == 50 and body["temperature"] == 0.5
    assert result.text == "case text" and result.usage.output_tokens == 9


def test_bedrock_thinking_budget_fits_inside_max_tokens():
    body = BedrockProvider(client=object()).build_body(
        "us.anthropic.claude-opus-4-5-20251101-v1:0", req(reasoning_max_tokens=2000)
    )
    assert body["max_tokens"] == 2050
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}


def test_bedrock_rejects_non_claude_models():
    with pytest.raises(ProviderError):
        BedrockProvider(client=object()).build_body("meta.llama-x", req())


class FakeGeminiModels:
    def __init__(self):
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        meta = SimpleNamespace(prompt_token_count=15, candidates_token_count=6, thoughts_token_count=4)
        return SimpleNamespace(
            text="explanation",
            usage_metadata=meta,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            model_version="gemini-2.5-flash-001",
            model_dump=lambda **kw: {"text": "explanation"},
        )


def test_gemini_counts_thinking_as_output():
    models = FakeGeminiModels()
    provider = GeminiProvider(client=SimpleNamespace(models=models))
    result = provider.send("gemini-2.5-flash", req(reasoning_max_tokens=100), PRICE)
    config = models.kwargs["config"]
    assert config.max_output_tokens == 150
    assert config.thinking_config.thinking_budget == 100
    assert config.system_instruction == "be brief"
    assert result.usage.output_tokens == 10 and result.usage.reasoning_tokens == 4
    assert result.model_used == "gemini-2.5-flash-001"
