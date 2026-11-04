"""The reviewer: a reasoning model that decides alone, or accepts or overrides a proposed decision.

Both prompts carry the same policy and content. In review, the reviewer is told the
proposal comes from an automated moderation model and nothing about how accurate that model
is, so the Jev and random conditions read exactly alike. (A warning that the upstream model
is often wrong is one of the cheap alternatives to post-training, for a later experiment.)
The confidence line appears only in the *_conf conditions.

Protocols (E.103) change how a review is asked, never what it shows about the item:
- note: a line under the automated model's decision (a warning, or its error rate);
- task_prefix: an instruction before the task (e.g. argue for the other label first);
- own_answer: blind-first. The reviewer's earlier answer alone comes first, as a prior turn of
  the conversation, then the automated model's decision is shown and the review is asked.
With none of them, the prompt is exactly E.101's and E.102's.

The reply is JSON: {"final_label"} alone; {"verdict", "final_label"} in review. The final
label is the cascade's decision. A verdict that contradicts it (accept with a different
label, or override with the same one) is kept and flagged, not treated as a parse failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sysone.cascade.tasks import CascadeTask
from sysone.clients.base import GuardedClient
from sysone.explain.jsonout import parse_json_object, uses_schema
from sysone.types import ChatRequest, ChatResponse, Message

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
_BLIND_FIRST_CONTEXT = (
    "An automated moderation model has also decided whether this {item_noun} is {a} or {b} under the "
    "{policy_name}. Its decision is below. You are reviewing its decision before it takes effect."
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
                      note: str | None = None, task_prefix: str | None = None, own_answer: str | None = None,
                      sample_index: int = 0, tags: dict[str, str] | None = None) -> ChatRequest:
        t = self.task
        a, b = t.labels
        fill = {"item_noun": t.item_noun, "a": f'"{a}"', "b": f'"{b}"', "policy_name": t.policy_name,
                "short_noun": t.item_heading.lower(), "quoted": f'"{a}" or "{b}"'}
        review = proposal is not None
        if own_answer is not None and not review:
            raise ValueError("blind-first (own_answer) needs a proposal to review")
        render = lambda parts: "\n\n".join(f"## {name}\n{body}" for name, body in parts)
        decision = None
        if review:
            decision = f'"{proposal}"'
            if confidence is not None:
                decision += f"\nThe automated model's confidence in this decision: {confidence:.0%}"
            if note:
                decision += f"\n{note}"
        task_text = (_REVIEW_TASK if review else _ALONE_TASK).format(**fill)
        if task_prefix:
            task_text = f"{task_prefix}\n\n{task_text}"
        if own_answer is not None:
            # Blind-first: the reviewer's answer alone is a prior turn; then the review.
            first = render([("Context", _ALONE_CONTEXT.format(**fill)), (t.policy_name.capitalize(), t.policy),
                            (t.item_heading, text), ("Task", _ALONE_TASK.format(**fill))])
            then = render([("Context", _BLIND_FIRST_CONTEXT.format(**fill)), ("Automated model's decision", decision),
                           ("Task", task_text)])
            messages = (Message("user", first), Message("assistant", json.dumps({"final_label": own_answer})),
                        Message("user", then))
        else:
            parts = [("Context", (_REVIEW_CONTEXT if review else _ALONE_CONTEXT).format(**fill)),
                     (t.policy_name.capitalize(), t.policy),
                     (t.item_heading, text)]
            if review:
                parts.append(("Automated model's decision", decision))
            parts.append(("Task", task_text))
            messages = (Message("user", render(parts)),)
        return ChatRequest(
            messages=messages,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            reasoning_max_tokens=self.reasoning_max_tokens,
            json_schema=review_schema(t, review=review) if uses_schema(self.client.spec.provider) else None,
            sample_index=sample_index,
            tags=tags or {},
        )

    def decide(self, text: str, proposal: str | None = None, confidence: float | None = None, *,
               note: str | None = None, task_prefix: str | None = None, own_answer: str | None = None,
               sample_index: int = 0, tags: dict[str, str] | None = None) -> ReviewResult:
        request = self.build_request(text, proposal, confidence, note=note, task_prefix=task_prefix,
                                     own_answer=own_answer, sample_index=sample_index, tags=tags)
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
