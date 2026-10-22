"""Jev as the black-box decision model f: policy + case text in, structured decision out.

Jev returns only a label and a confidence, with no explanation, matching the
setting where a separate LLM explains the decision afterwards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sysone.clients.base import GuardedClient
from sysone.clients.cache import ResponseCache
from sysone.types import ChatRequest, ChatResponse

LABELS = ("approve", "deny")

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["decision", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are the automated refund decision engine for Larkspur Home Goods. "
    "Decide each refund request by applying the refund policy to the case file. "
    "Respond with JSON only, with no explanation."
)

USER_TEMPLATE = """REFUND POLICY
{policy}
CASE FILE
{case}

Return a JSON object: {{"decision": "approve" or "deny", "confidence": a number from 0 to 1}}."""


class QueryBudgetExhausted(RuntimeError):
    pass


class QueryBudget:
    """The research query budget B: how many decision queries a method may make.

    Separate from the dollar cost caps. Cached queries count too, because B limits
    the information a method gets about f, not the money spent.
    """

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise QueryBudgetExhausted(f"query budget of {self.limit} used up")
        self.used += 1


@dataclass(frozen=True)
class Decision:
    label: str | None
    confidence: float | None
    parse_ok: bool
    parse_error: str | None
    raw_text: str


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    response: ChatResponse
    request_key: str


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_decision(text: str) -> Decision:
    def fail(reason: str) -> Decision:
        return Decision(None, None, False, reason, text)

    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end < start:
        return fail("no JSON object found")
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as e:
        return fail(f"invalid JSON: {e}")
    label = str(data.get("decision", "")).strip().lower()
    if label not in LABELS:
        return fail(f"decision {data.get('decision')!r} not in {LABELS}")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return fail(f"confidence {confidence!r} not a number in [0, 1]")
    return Decision(label, float(confidence), True, None, text)


class JevDecisionModel:
    def __init__(
        self,
        client: GuardedClient,
        *,
        temperature: float | None = None,
        max_output_tokens: int = 256,
        reasoning_max_tokens: int | None = None,
        structured_output: bool = True,
    ):
        self.client = client
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.structured_output = structured_output

    def build_request(self, policy: str, case_text: str, *, sample_index: int = 0, tags: dict[str, str] | None = None) -> ChatRequest:
        return ChatRequest.from_prompt(
            USER_TEMPLATE.format(policy=policy, case=case_text),
            system=SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
            reasoning_max_tokens=self.reasoning_max_tokens,
            temperature=self.temperature,
            json_schema=DECISION_SCHEMA if self.structured_output else None,
            sample_index=sample_index,
            tags=tags or {},
        )

    def decide(
        self,
        policy: str,
        case_text: str,
        *,
        sample_index: int = 0,
        tags: dict[str, str] | None = None,
        budget: QueryBudget | None = None,
    ) -> DecisionResult:
        if budget is not None:
            budget.spend()
        request = self.build_request(policy, case_text, sample_index=sample_index, tags=tags)
        response = self.client.complete(request)
        key = ResponseCache.key(self.client.spec.provider, self.client.spec.model, request)
        return DecisionResult(parse_decision(response.text), response, key)
