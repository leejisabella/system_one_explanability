"""Provider-neutral request and response types shared by every client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    # Cap on visible output tokens. Every request must set it so worst-case cost is bounded.
    max_output_tokens: int
    system: str | None = None
    temperature: float | None = None
    seed: int | None = None
    # Cap on reasoning/thinking tokens, for providers that bill them separately.
    # Worst-case cost assumes up to max_output_tokens + reasoning_max_tokens output tokens.
    reasoning_max_tokens: int | None = None
    # Reasoning level for models that take one instead of (or as well as) a token budget:
    # Gemini's thinking_level ("minimal", "low", ...), gpt-oss's reasoning_effort ("low", ...).
    # Usually set per model in configs/models.yaml rather than by the caller.
    reasoning_effort: str | None = None
    # Provider service tier, e.g. Bedrock's "flex" (discounted, slower). Set per model in
    # configs/models.yaml; the cost is charged at the tier the provider reports serving.
    service_tier: str | None = None
    # JSON Schema for structured output, where the provider supports it.
    json_schema: dict[str, Any] | None = None
    # Distinguishes repeated samples of the same request, so stability sampling
    # makes real calls instead of hitting the cache.
    sample_index: int = 0
    # Labels for the spending log (experiment, case ID, ...). Not part of the cache key.
    tags: dict[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.reasoning_max_tokens is not None and self.reasoning_max_tokens < 0:
            raise ValueError("reasoning_max_tokens must be non-negative")
        if not self.messages:
            raise ValueError("a request needs at least one message")

    @staticmethod
    def from_prompt(prompt: str, *, max_output_tokens: int, **kwargs: Any) -> ChatRequest:
        return ChatRequest(
            messages=(Message("user", prompt),), max_output_tokens=max_output_tokens, **kwargs
        )


@dataclass(frozen=True)
class SystemOneRequest:
    """One call to a System One decision model (TypeSafe's Jev): typed questions about a state.

    `state` is the content every question refers to: a string, or a JSON object with
    named parts (TypeSafe recommends an object). `questions` maps question names to
    question objects, e.g. {"type": "choice", "instructions": ..., "criteria": {...}}.
    """

    state: Any
    questions: dict[str, dict[str, Any]]
    # Not sent to the provider. Distinguishes repeated samples of the same input, so
    # repeats make real calls instead of hitting the cache.
    sample_index: int = 0
    # The API has no output-length parameter, and output tokens are currently free.
    # This bound only feeds the worst-case cost estimate, in case output is ever billed.
    max_output_tokens: int = 1024
    tags: dict[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("a request needs at least one question")
        for name, question in self.questions.items():
            if "type" not in question:
                raise ValueError(f"question '{name}' has no type")


# Every request type a client can send.
Request = ChatRequest | SystemOneRequest


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    # All billed output tokens, including reasoning/thinking tokens.
    output_tokens: int
    # Informational: the reasoning share of output_tokens.
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class ProviderResult:
    """What a provider adapter returns for one successful request."""

    text: str
    usage: Usage
    model_used: str
    finish_reason: str | None
    raw: dict[str, Any]
    # Cost the provider itself reported (OpenRouter does), else None.
    reported_cost_usd: float | None = None
    # The service tier the provider says served the request, where it reports one.
    service_tier: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    """A completed request. For System One requests, `text` is the answers as JSON and
    `raw` is the provider's full response, including per-option probabilities."""

    text: str
    usage: Usage
    cost_usd: float
    provider: str
    model_requested: str
    # The model that actually served the request. Differs from model_requested for routers.
    model_used: str
    finish_reason: str | None
    raw: dict[str, Any]
    cached: bool = False
    service_tier: str | None = None
