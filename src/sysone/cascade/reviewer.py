"""The reviewer: a reasoning model that decides alone, or accepts or overrides a proposed decision.

Both prompts carry the same policy and content. In review, the reviewer is told the
proposal comes from an automated moderation model and nothing about how accurate that model
is, so the Jev and random conditions read exactly alike. (A warning that the upstream model
is often wrong is one of the cheap alternatives to post-training, for a later experiment.)
The confidence line appears only in the *_conf conditions.

The reply is JSON: {"final_label"} alone; {"verdict", "final_label"} in review. The final
label is the cascade's decision. A verdict that contradicts it (accept with a different
label, or override with the same one) is kept and flagged, not treated as a parse failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sysone.cascade.tasks import CascadeTask
from sysone.clients.base import GuardedClient
from sysone.explain.jsonout import parse_json_object, uses_schema
from sysone.types import ChatRequest, ChatResponse

ACCEPT, OVERRIDE = "accept", "override"

_ALONE_CONTEXT = "Decide whether this {item_noun} is {a} or {b} under the {policy_name} below."
_REVIEW_CONTEXT = (
    "An automated moderation model has decided whether this {item_noun} is {a} or {b} under the "
    "{policy_name} below. You are reviewing its decision before it takes effect."
)
_ALONE_TASK = (
    "Apply the {policy_name} to the {short_noun}.\n\n"
    "- final_label: {quoted}.\n\n"
    'Reply with only a JSON object: {{"final_label": ...}}.'
)
_REVIEW_TASK = (
    "Accept the automated model's decision if it is correct under the {policy_name}; override it if it is not.\n\n"
    '- verdict: "accept" or "override".\n'
    "- final_label: the decision that should take effect, {quoted}. It is the automated model's decision "
    "if you accept, and the other label if you override.\n\n"
    'Reply with only a JSON object: {{"verdict": ..., "final_label": ...}}.'
)


def review_schema(task: CascadeTask, *, review: bool) -> dict[str, Any]:
    props: dict[str, Any] = {"final_label": {"type": "string", "enum": list(task.labels)}}
    if review:
        props = {"verdict": {"type": "string", "enum": [ACCEPT, OVERRIDE]}, **props}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


@dataclass(frozen=True)
class ReviewResult:
    final_label: str | None
    # None when deciding alone.
    verdict: str | None
    # The verdict agrees with the final label (always true alone).
    consistent: bool | None
    parse_ok: bool
    parse_error: str | None
    response: ChatResponse
    request_key: str


class Reviewer:
    def __init__(
        self,
        client: GuardedClient,
        task: CascadeTask,
        *,
        temperature: float | None = None,
        reasoning_max_tokens: int | None = None,
        max_output_tokens: int = 256,
    ):
        self.client = client
        self.task = task
        self.temperature = temperature
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_output_tokens = max_output_tokens

    def build_request(self, text: str, proposal: str | None = None, confidence: float | None = None, *,
                      sample_index: int = 0, tags: dict[str, str] | None = None) -> ChatRequest:
        t = self.task
        a, b = t.labels
        fill = {"item_noun": t.item_noun, "a": f'"{a}"', "b": f'"{b}"', "policy_name": t.policy_name,
                "short_noun": t.item_heading.lower(), "quoted": f'"{a}" or "{b}"'}
        review = proposal is not None
        parts = [("Context", (_REVIEW_CONTEXT if review else _ALONE_CONTEXT).format(**fill)),
                 (t.policy_name.capitalize(), t.policy),
                 (t.item_heading, text)]
        if review:
            decision = f'"{proposal}"'
            if confidence is not None:
                decision += f"\nThe automated model's confidence in this decision: {confidence:.0%}"
            parts.append(("Automated model's decision", decision))
        parts.append(("Task", (_REVIEW_TASK if review else _ALONE_TASK).format(**fill)))
        prompt = "\n\n".join(f"## {name}\n{body}" for name, body in parts)
        return ChatRequest.from_prompt(
            prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            reasoning_max_tokens=self.reasoning_max_tokens,
            json_schema=review_schema(t, review=review) if uses_schema(self.client.spec.provider) else None,
            sample_index=sample_index,
            tags=tags or {},
        )

    def decide(self, text: str, proposal: str | None = None, confidence: float | None = None, *,
               sample_index: int = 0, tags: dict[str, str] | None = None) -> ReviewResult:
        request = self.build_request(text, proposal, confidence, sample_index=sample_index, tags=tags)
        response = self.client.complete(request)
        return parse_review(response, self.client.request_key(request), self.task.labels, proposal)


def parse_review(response: ChatResponse, key: str, labels: tuple[str, ...], proposal: str | None) -> ReviewResult:
    def fail(reason: str) -> ReviewResult:
        return ReviewResult(None, None, None, False, reason, response, key)

    obj = parse_json_object(response.text)
    if obj is None:
        return fail(f"reply is not a JSON object: {response.text[:200]!r}")
    final = obj.get("final_label")
    if final not in labels:
        return fail(f"final_label {final!r} not in {labels}")
    if proposal is None:
        return ReviewResult(final, None, True, True, None, response, key)
    verdict = obj.get("verdict")
    if verdict not in (ACCEPT, OVERRIDE):
        return fail(f"verdict {verdict!r} is not accept or override")
    consistent = (verdict == ACCEPT) == (final == proposal)
    return ReviewResult(final, verdict, consistent, True, None, response, key)
