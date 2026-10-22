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


@dataclass(frozen=True)
class ChatResponse:
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
