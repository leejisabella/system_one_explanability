"""Jev as the black-box decision model f: one typed choice question about a refund case.

The state holds the refund policy and the case file as named parts, which is how
TypeSafe recommends laying out a decision that compares a record against rules. Jev
answers with the chosen option, a probability for every option, and TypeSafe's
confidence summary. It gives no explanation, matching the setting where a separate
LLM explains the decision afterwards.

The probabilities are the measurement. TypeSafe's `confidence` is derived from them
and is kept only for reference.

Two design choices follow independent audits of Jev:
- There is an explicit "undetermined" option. Without a "can't tell" option, Jev
  answers unanswerable questions confidently, and ambiguous cases need a way out.
- The option order can rotate with the sample index, because Jev sometimes leans
  toward the first-listed option. Across three repeats each option is listed first once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sysone.clients.base import GuardedClient
from sysone.types import ChatResponse, SystemOneRequest

# "undetermined" matches the benchmark's outcome name for cases the policy can't decide.
LABELS = ("approve", "deny", "undetermined")
QUESTION = "refund_decision"
INSTRUCTIONS = "Decide this refund request by applying the refund policy to the case file."
CRITERIA = {
    "approve": "The refund policy allows a refund for this request.",
    "deny": "The refund policy does not allow a refund for this request.",
    "undetermined": (
        "The decision cannot be determined: the case file is missing information "
        "the refund policy needs to decide this request."
    ),
}


def option_order(sample_index: int, *, rotate: bool = True) -> tuple[str, ...]:
    """The order options are listed in for one sample. Rotating by the sample index
    lists each option first equally often across repeats."""
    if not rotate:
        return LABELS
    k = sample_index % len(LABELS)
    return LABELS[k:] + LABELS[:k]
# Probabilities come rounded to two decimals, so their sum can be off by a little.
PROBABILITY_SUM_TOLERANCE = 0.02


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
    # One probability per label, as Jev returned them.
    probabilities: dict[str, float] | None
    # TypeSafe's summary of the distribution: 1 when all probability is on one option.
    confidence: float | None
    parse_ok: bool
    parse_error: str | None
    raw_answer: Any

    @property
    def p_approve(self) -> float | None:
        return None if self.probabilities is None else self.probabilities["approve"]


def _is_unit_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and 0 <= x <= 1


def parse_answer(answer: Any) -> Decision:
    """Validate one choice answer from Jev."""

    def fail(reason: str) -> Decision:
        return Decision(None, None, None, False, reason, answer)

    if not isinstance(answer, dict):
        return fail("answer is not an object")
    if answer.get("type") != "choice":
        return fail(f"answer type {answer.get('type')!r} is not 'choice'")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        return fail("answer has no probabilities")
    if set(probs) != set(LABELS):
        return fail(f"probabilities cover {sorted(probs)}, expected {sorted(LABELS)}")
    if not all(_is_unit_number(p) for p in probs.values()):
        return fail(f"probabilities {probs!r} are not all numbers in [0, 1]")
    if abs(sum(probs.values()) - 1) > PROBABILITY_SUM_TOLERANCE:
        return fail(f"probabilities sum to {sum(probs.values()):.3f}, not 1")
    label = answer.get("choice")
    if label not in LABELS:
        return fail(f"choice {label!r} not in {LABELS}")
    confidence = answer.get("confidence")
    if not _is_unit_number(confidence):
        return fail(f"confidence {confidence!r} not a number in [0, 1]")
    return Decision(label, {k: float(probs[k]) for k in LABELS}, float(confidence), True, None, answer)


def parse_response(response: ChatResponse) -> Decision:
    answers = response.raw.get("answers")
    if not isinstance(answers, dict) or QUESTION not in answers:
        return Decision(None, None, None, False, f"no answer to '{QUESTION}'", answers)
    return parse_answer(answers[QUESTION])


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    response: ChatResponse
    request_key: str
    # The order the options were listed in for this request.
    option_order: tuple[str, ...]


class JevDecisionModel:
    def __init__(self, client: GuardedClient, *, rotate_options: bool = True):
        self.client = client
        self.rotate_options = rotate_options

    def build_request(
        self, policy: str, case_text: str, *, sample_index: int = 0, tags: dict[str, str] | None = None
    ) -> SystemOneRequest:
        order = option_order(sample_index, rotate=self.rotate_options)
        return SystemOneRequest(
            state={"refund_policy": policy, "case_file": case_text},
            questions={QUESTION: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                # The API keeps this order, so it is the order Jev sees.
                "criteria": {label: CRITERIA[label] for label in order},
            }},
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
        key = self.client.request_key(request)
        order = tuple(request.questions[QUESTION]["criteria"])
        return DecisionResult(parse_response(response), response, key, order)
