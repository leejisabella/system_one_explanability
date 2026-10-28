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
from sysone.clients.typesafe import TYPESAFE_URL, TypeSafeProvider
from sysone.types import ChatRequest, SystemOneRequest

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


def test_bedrock_claude_body_rejects_converse_models():
    with pytest.raises(ProviderError):
        BedrockProvider(client=object()).build_body("meta.llama-x", req())


class FakeConverse:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        return {**self.payload, "ResponseMetadata": {"HTTPStatusCode": 200}}


CONVERSE_REPLY = {
    "output": {"message": {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
        {"text": '{"decision": "deny"}'},
    ]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42},
    "serviceTier": {"type": "flex"},
}


def test_bedrock_converse_request_and_response():
    fake = FakeConverse(CONVERSE_REPLY)
    request = req(reasoning_max_tokens=200, reasoning_effort="low", service_tier="flex", temperature=0.2)
    result = BedrockProvider(client=fake).send("openai.gpt-oss-120b-1:0", request, PRICE)
    kw = fake.kwargs
    assert kw["modelId"] == "openai.gpt-oss-120b-1:0"
    assert kw["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert kw["system"] == [{"text": "be brief"}]
    # The reasoning budget sits inside maxTokens, which caps reasoning for gpt-oss.
    assert kw["inferenceConfig"] == {"maxTokens": 250, "temperature": 0.2}
    assert kw["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    assert kw["serviceTier"] == {"type": "flex"}
    # Only the answer text is kept; reasoning is billed in outputTokens.
    assert result.text == '{"decision": "deny"}'
    assert result.usage.input_tokens == 30 and result.usage.output_tokens == 12
    assert result.service_tier == "flex" and result.model_used == "openai.gpt-oss-120b-1:0"
    assert "ResponseMetadata" not in result.raw


def test_bedrock_converse_plain_model_sends_no_extras():
    fake = FakeConverse({**CONVERSE_REPLY, "serviceTier": None})
    result = BedrockProvider(client=fake).send("us.meta.llama4-maverick-17b-instruct-v1:0", req(), PRICE)
    assert set(fake.kwargs) == {"modelId", "messages", "system", "inferenceConfig"}
    assert result.service_tier is None


@pytest.mark.parametrize("model,kwargs", [
    ("us.meta.llama4-maverick-17b-instruct-v1:0", {"reasoning_effort": "low"}),  # not an OpenAI model
    ("qwen.qwen3-235b-a22b-2507-v1:0", {"reasoning_max_tokens": 100}),
    ("openai.gpt-oss-120b-1:0", {"seed": 0}),  # Converse has no seed
    ("openai.gpt-oss-120b-1:0", {"json_schema": {"type": "object"}}),
    ("global.anthropic.claude-haiku-5-5", {"service_tier": "flex"}),  # tiers only via Converse
    ("global.anthropic.claude-haiku-5-5", {"reasoning_effort": "low"}),
])
def test_bedrock_refuses_settings_it_would_drop(model, kwargs):
    # Refused before sending, so an unsupported setting is never silently ignored.
    with pytest.raises(ProviderError) as exc:
        BedrockProvider(client=FakeConverse(CONVERSE_REPLY)).send(model, req(**kwargs), PRICE)
    assert exc.value.billed == "no"


def test_bedrock_local_validation_errors_are_not_billed():
    from botocore.exceptions import ParamValidationError

    class Rejecting:
        def converse(self, **kwargs):
            raise ParamValidationError(report="bad field")

    with pytest.raises(ProviderError) as exc:
        BedrockProvider(client=Rejecting()).send("openai.gpt-oss-120b-1:0", req(), PRICE)
    assert exc.value.billed == "no"


def test_bedrock_routes_global_claude_through_invoke_model():
    fake = FakeBedrock({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 5, "output_tokens": 1}})
    BedrockProvider(client=fake).send("global.anthropic.claude-haiku-5-5", req(), PRICE)
    assert fake.kwargs["modelId"] == "global.anthropic.claude-haiku-5-5"


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


def test_gemini_thinking_level_takes_precedence_over_a_budget():
    models = FakeGeminiModels()
    GeminiProvider(client=SimpleNamespace(models=models)).send(
        "gemini-3.1-flash-lite", req(reasoning_effort="minimal"), PRICE)
    thinking = models.kwargs["config"].thinking_config
    assert thinking.thinking_level.value == "MINIMAL" and thinking.thinking_budget is None
    # Headroom so thinking can't use up a short reply's cap.
    assert models.kwargs["config"].max_output_tokens == 50 + 1024


def test_openrouter_sends_reasoning_effort():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "hi"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    openrouter(handler).send("m", req(reasoning_effort="low", reasoning_max_tokens=10), PRICE)
    assert seen["body"]["reasoning"] == {"effort": "low"}


# --- TypeSafe (native Jev API) ---

TS_ANSWERS = {"refund_decision": {"type": "choice", "choice": "deny", "confidence": 0.4,
                                  "probabilities": {"approve": 0.3, "deny": 0.7}}}


def typesafe(handler):
    return TypeSafeProvider(api_key="test", http=httpx.Client(transport=httpx.MockTransport(handler)))


def s1_req():
    return SystemOneRequest(
        state={"refund_policy": "P", "case_file": "C"},
        questions={"refund_decision": {"type": "choice", "instructions": "Decide.",
                                       "criteria": {"approve": "yes", "deny": "no"}}},
        sample_index=4,
        tags={"case_id": "c1"},
    )


def test_typesafe_sends_state_and_questions_and_reads_probabilities():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": TS_ANSWERS,
                                         "usage": {"input_tokens": 321, "output_tokens": 5}})

    result = typesafe(handler).send("jev-latest", s1_req(), PRICE)
    assert seen["url"] == TYPESAFE_URL and seen["auth"] == "Bearer test"
    # Only the API's own fields are sent: no sample index or tags.
    assert seen["body"] == {"model": "jev-latest", "state": {"refund_policy": "P", "case_file": "C"},
                            "questions": s1_req().questions}
    assert result.model_used == "jev-1.14.0"
    assert result.raw["answers"]["refund_decision"]["probabilities"] == {"approve": 0.3, "deny": 0.7}
    assert result.usage.input_tokens == 321 and result.usage.output_tokens == 5
    assert result.reported_cost_usd is None  # cost comes from the price table


@pytest.mark.parametrize("status,retryable", [(429, True), (529, True), (500, True), (400, False), (422, False), (401, False)])
def test_typesafe_http_errors(status, retryable):
    provider = typesafe(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ProviderError) as exc:
        provider.send("jev-latest", s1_req(), PRICE)
    assert exc.value.retryable is retryable and exc.value.billed == "no"


def test_typesafe_timeout_is_unknown_billing():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as exc:
        typesafe(handler).send("jev-latest", s1_req(), PRICE)
    assert exc.value.billed == "unknown" and exc.value.retryable


@pytest.mark.parametrize("payload", [
    {"answers": TS_ANSWERS, "usage": {"input_tokens": 1}},             # no version
    {"model": "", "answers": TS_ANSWERS, "usage": {"input_tokens": 1}},
    {"model": "jev-1.14.0", "usage": {"input_tokens": 1}},             # no answers
])
def test_typesafe_requires_version_and_answers(payload):
    with pytest.raises(ProviderError) as exc:
        typesafe(lambda r: httpx.Response(200, json=payload)).send("jev-latest", s1_req(), PRICE)
    assert exc.value.billed == "unknown" and not exc.value.retryable


def test_typesafe_refuses_chat_requests():
    with pytest.raises(TypeError):
        typesafe(lambda r: httpx.Response(200)).send("jev-latest", req(), PRICE)


def test_typesafe_needs_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        TypeSafeProvider()
