"""The explainer model: its own decisions on cases, and explanations of Jev's decisions.

The explainer plays two separate roles, in separate requests that never share context:

- Independent decisions. It decides each case and variant itself, from the policy and
  the case file, without seeing Jev's answer. The question and options match Jev's, and
  the option order rotates with the sample index in the same way.
- Explanations. Given the policy, the original case file and Jev's decision, it explains
  that decision. It never sees the counterfactual variants the explanation is later
  scored on, and an explanation is never revised once written.

Explanation conditions:
- "free_form": a plain-language explanation.
- "structured": the explanation as a basis (see sysone.explain.basis), plus a short summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sysone.clients.base import GuardedClient
from sysone.decision.jev import CRITERIA, INSTRUCTIONS, LABELS, option_order
from sysone.explain.basis import BASIS_INSTRUCTIONS, BASIS_SCHEMA, Basis, BasisError, parse_basis
from sysone.explain.jsonout import parse_json_object, uses_schema
from sysone.types import ChatRequest, ChatResponse

FREE_FORM = "free_form"
STRUCTURED = "structured"
EXPLAINED_CONDITIONS = (FREE_FORM, STRUCTURED)

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {"decision": {"type": "string", "enum": list(LABELS)}},
}
# Sent with every explainer decision. Without it Llama 4 Maverick often reasons step by step
# and is cut off at the 64-token cap before answering (58% of its decisions in
# pilot1-v1-2026-10-28, concentrated on edited variants).
DECISION_SYSTEM = (
    "You answer with a single JSON object and nothing else. Do not explain or show your reasoning. "
    'Your entire reply must be exactly {"decision": "<option>"}.'
)
STRUCTURED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["basis", "summary"],
    "properties": {"basis": BASIS_SCHEMA, "summary": {"type": "string"}},
}

_EXPLAIN_CONTEXT = (
    "An automated decision system decided the refund request below. You did not make the decision, "
    "and you cannot see how the system works."
)
_FREE_FORM_TASK = (
    "Explain why the system reached this decision: which facts in the case file and which parts of the "
    "refund policy it was based on. Write plain prose, at most 200 words."
)
_STRUCTURED_TASK = (
    "Explain why the system reached this decision by stating the decision rule it applied, as a basis.\n\n"
    f"{BASIS_INSTRUCTIONS}\n\n"
    'Reply with only a JSON object: {"basis": <the basis>, "summary": <one or two sentences in plain prose>}.'
)


@dataclass(frozen=True)
class ExplainerDecision:
    label: str | None
    parse_ok: bool
    parse_error: str | None
    response: ChatResponse
    request_key: str
    option_order: tuple[str, ...]


@dataclass(frozen=True)
class Explanation:
    condition: str
    # The explanation exactly as the simulator will see it. None if it couldn't be parsed.
    text: str | None
    # Structured condition only.
    basis: Basis | None
    summary: str | None
    parse_ok: bool
    parse_error: str | None
    response: ChatResponse
    request_key: str


def _sections(**parts: str) -> str:
    return "\n\n".join(f"## {name.replace('_', ' ').capitalize()}\n{text}" for name, text in parts.items())


class ExplainerModel:
    def __init__(
        self,
        client: GuardedClient,
        *,
        temperature: float | None = None,
        reasoning_max_tokens: int | None = None,
        decision_max_tokens: int = 64,
        explanation_max_tokens: int = 1024,
    ):
        self.client = client
        self.temperature = temperature
        self.reasoning_max_tokens = reasoning_max_tokens
        self.decision_max_tokens = decision_max_tokens
        self.explanation_max_tokens = explanation_max_tokens

    def _request(self, prompt: str, *, max_tokens: int, schema: dict | None, sample_index: int,
                 tags: dict[str, str], system: str | None = None) -> ChatRequest:
        return ChatRequest.from_prompt(
            prompt,
            system=system,
            max_output_tokens=max_tokens,
            temperature=self.temperature,
            reasoning_max_tokens=self.reasoning_max_tokens,
            json_schema=schema if schema is not None and uses_schema(self.client.spec.provider) else None,
            sample_index=sample_index,
            tags=tags,
        )

    def _send(self, request: ChatRequest) -> tuple[ChatResponse, str]:
        response = self.client.complete(request)
        return response, self.client.request_key(request)

    def build_decision_request(self, policy: str, case_text: str, *, sample_index: int = 0,
                               tags: dict[str, str] | None = None) -> ChatRequest:
        order = option_order(sample_index)
        options = "\n".join(f'- "{label}": {CRITERIA[label]}' for label in order)
        prompt = _sections(
            refund_policy=policy,
            case_file=case_text,
            task=f"{INSTRUCTIONS}\n\nOptions:\n{options}\n\n"
                 'Reply with only a JSON object: {"decision": <one of the options>}.',
        )
        return self._request(prompt, max_tokens=self.decision_max_tokens, schema=DECISION_SCHEMA,
                             sample_index=sample_index, tags=tags or {}, system=DECISION_SYSTEM)

    def decide(self, policy: str, case_text: str, *, sample_index: int = 0,
               tags: dict[str, str] | None = None) -> ExplainerDecision:
        request = self.build_decision_request(policy, case_text, sample_index=sample_index, tags=tags)
        response, key = self._send(request)
        obj = parse_json_object(response.text)
        label = obj.get("decision") if obj else None
        error = None if label in LABELS else f"no valid decision in reply: {response.text[:200]!r}"
        return ExplainerDecision(label if error is None else None, error is None, error, response, key,
                                 option_order(sample_index))

    def build_explanation_request(self, condition: str, policy: str, case_text: str, jev_decision: str, *,
                                  tags: dict[str, str] | None = None) -> ChatRequest:
        if condition not in EXPLAINED_CONDITIONS:
            raise ValueError(f"unknown explanation condition {condition!r}")
        task = _FREE_FORM_TASK if condition == FREE_FORM else _STRUCTURED_TASK
        prompt = _sections(
            context=_EXPLAIN_CONTEXT,
            refund_policy=policy,
            case_file=case_text,
            system_decision=jev_decision,
            task=task,
        )
        schema = STRUCTURED_SCHEMA if condition == STRUCTURED else None
        return self._request(prompt, max_tokens=self.explanation_max_tokens, schema=schema, sample_index=0,
                             tags=tags or {})

    def explain(self, condition: str, policy: str, case_text: str, jev_decision: str, *,
                tags: dict[str, str] | None = None) -> Explanation:
        request = self.build_explanation_request(condition, policy, case_text, jev_decision, tags=tags)
        response, key = self._send(request)
        if condition == FREE_FORM:
            text = response.text.strip()
            error = None if text else "empty explanation"
            return Explanation(condition, text or None, None, None, error is None, error, response, key)
        return _parse_structured(response, key)


def _parse_structured(response: ChatResponse, key: str) -> Explanation:
    def fail(reason: str) -> Explanation:
        return Explanation(STRUCTURED, None, None, None, False, reason, response, key)

    obj = parse_json_object(response.text)
    if obj is None:
        return fail(f"reply is not a JSON object: {response.text[:200]!r}")
    try:
        basis = parse_basis(obj.get("basis"))
    except BasisError as e:
        return fail(f"invalid basis: {e}")
    summary = obj.get("summary")
    if not isinstance(summary, str):
        return fail("summary is not a string")
    text = json.dumps({"basis": basis.to_dict(), "summary": summary}, indent=2)
    return Explanation(STRUCTURED, text, basis, summary, True, None, response, key)


def decision_probabilities_text(label: str, probabilities: dict[str, float] | None) -> str:
    """The baseline 'explanation': Jev's own output, with no text from an explainer."""
    if not probabilities:
        return f"The system's output: decision {label}."
    probs = ", ".join(f"{k} {probabilities[k]:.2f}" for k in LABELS)
    return f"The system's output: decision {label}. Probability of each option: {probs}."


def explanation_record(explanation: Explanation) -> dict[str, Any]:
    """Fields of an explanation to save."""
    return {
        "condition": explanation.condition,
        "explanation_text": explanation.text,
        "basis": explanation.basis.to_dict() if explanation.basis else None,
        "summary": explanation.summary,
        "parse_ok": explanation.parse_ok,
        "parse_error": explanation.parse_error,
        "raw_text": explanation.response.text,
    }
