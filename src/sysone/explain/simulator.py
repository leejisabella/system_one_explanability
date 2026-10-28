"""The simulator: predicts Jev's decision on a hidden variant from a frozen explanation.

The simulator sees the refund policy, the original case file, Jev's decision on it,
the explanation, and the variant's case file. It never sees Jev's decision on the
variant, the true record or the ground truth. In the no-explanation control the
explanation section is left out and nothing else changes, so the gain over the control
is what the explanation adds.

Every answer has the same structure:
- predicted_decision and a probability for each option;
- basis: the rule the simulator took from the explanation (see sysone.explain.basis),
  which is run against the variant's record to give a second, model-free prediction;
- covered: whether the explanation determines this variant at all. "The explanation
  doesn't say" is different from predicting that Jev answers "undetermined".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sysone.clients.base import GuardedClient
from sysone.decision.jev import LABELS, PROBABILITY_SUM_TOLERANCE
from sysone.explain.basis import BASIS_INSTRUCTIONS, BASIS_SCHEMA, Basis, BasisError, parse_basis
from sysone.explain.jsonout import parse_json_object, uses_schema
from sysone.types import ChatRequest, ChatResponse

# Models state probabilities less precisely than Jev's two-decimal output.
SIMULATOR_PROBABILITY_TOLERANCE = max(PROBABILITY_SUM_TOLERANCE, 0.05)

SIMULATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["predicted_decision", "probabilities", "basis", "covered"],
    "properties": {
        "predicted_decision": {"type": "string", "enum": list(LABELS)},
        "probabilities": {
            "type": "object",
            "additionalProperties": False,
            "required": list(LABELS),
            "properties": {label: {"type": "number"} for label in LABELS},
        },
        "basis": BASIS_SCHEMA,
        "covered": {"type": "boolean"},
    },
}

_CONTEXT = (
    "An automated decision system decides refund requests. It is given the refund policy and a case file, "
    "but it may not apply the policy exactly. You will predict what this system decides on a new version "
    "of a case it has already decided. Below are the refund policy, the original case file and the system's "
    "decision on it{explained}, and the new case file."
)
_TASK = (
    "Predict the system's decision on the new case file.\n\n"
    "- predicted_decision: the option you predict the system chooses.\n"
    "- probabilities: your probability for each option; they sum to 1.\n"
    "- basis: the decision rule you are using to predict the system's behavior, as a basis.\n"
    "- covered: true only if the explanation of the system's decision determines its decision on the new "
    "case file. False if no explanation is given, or it does not say what the system would do here.\n\n"
    f"{BASIS_INSTRUCTIONS}\n\n"
    'Reply with only a JSON object: {"predicted_decision": ..., "probabilities": {"approve": ..., '
    '"deny": ..., "undetermined": ...}, "basis": ..., "covered": ...}.'
)


@dataclass(frozen=True)
class Simulation:
    predicted: str | None
    probabilities: dict[str, float] | None
    # The basis exactly as returned, kept even when it can't be run.
    basis_raw: Any
    basis: Basis | None
    basis_error: str | None
    covered: bool | None
    parse_ok: bool
    parse_error: str | None
    response: ChatResponse
    request_key: str


def _is_unit_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and 0 <= x <= 1


class Simulator:
    def __init__(
        self,
        client: GuardedClient,
        *,
        temperature: float | None = None,
        reasoning_max_tokens: int | None = None,
        seed: int | None = None,
        max_output_tokens: int = 1024,
    ):
        self.client = client
        self.temperature = temperature
        self.reasoning_max_tokens = reasoning_max_tokens
        self.seed = seed
        self.max_output_tokens = max_output_tokens

    def build_request(self, policy: str, original_case: str, jev_decision: str, explanation: str | None,
                      variant_case: str, *, tags: dict[str, str] | None = None) -> ChatRequest:
        parts = [
            ("Context", _CONTEXT.format(explained=", an explanation of that decision" if explanation else "")),
            ("Refund policy", policy),
            ("Original case file", original_case),
            ("System decision on the original case file", jev_decision),
        ]
        if explanation is not None:
            parts.append(("Explanation of the system's decision", explanation))
        parts += [("New case file", variant_case), ("Task", _TASK)]
        prompt = "\n\n".join(f"## {name}\n{text}" for name, text in parts)
        return ChatRequest.from_prompt(
            prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            reasoning_max_tokens=self.reasoning_max_tokens,
            seed=self.seed,
            json_schema=SIMULATION_SCHEMA if uses_schema(self.client.spec.provider) else None,
            tags=tags or {},
        )

    def simulate(self, policy: str, original_case: str, jev_decision: str, explanation: str | None,
                 variant_case: str, *, tags: dict[str, str] | None = None) -> Simulation:
        request = self.build_request(policy, original_case, jev_decision, explanation, variant_case, tags=tags)
        response = self.client.complete(request)
        key = self.client.request_key(request)
        return parse_simulation(response, key)


def parse_simulation(response: ChatResponse, key: str) -> Simulation:
    def fail(reason: str, basis_raw: Any = None) -> Simulation:
        return Simulation(None, None, basis_raw, None, None, None, False, reason, response, key)

    obj = parse_json_object(response.text)
    if obj is None:
        return fail(f"reply is not a JSON object: {response.text[:200]!r}")
    basis_raw = obj.get("basis")
    predicted = obj.get("predicted_decision")
    if predicted not in LABELS:
        return fail(f"predicted_decision {predicted!r} not in {LABELS}", basis_raw)
    probs = obj.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(LABELS):
        return fail(f"probabilities {probs!r} do not cover {LABELS}", basis_raw)
    if not all(_is_unit_number(p) for p in probs.values()):
        return fail(f"probabilities {probs!r} are not all numbers in [0, 1]", basis_raw)
    if abs(sum(probs.values()) - 1) > SIMULATOR_PROBABILITY_TOLERANCE:
        return fail(f"probabilities sum to {sum(probs.values()):.3f}, not 1", basis_raw)
    covered = obj.get("covered")
    if not isinstance(covered, bool):
        return fail(f"covered {covered!r} is not true or false", basis_raw)
    # A basis that can't be run doesn't void the prediction; it is scored as undecidable.
    try:
        basis, basis_error = parse_basis(basis_raw), None
    except BasisError as e:
        basis, basis_error = None, str(e)
    return Simulation(predicted, {k: float(probs[k]) for k in LABELS}, basis_raw, basis, basis_error, covered,
                      True, None, response, key)
