"""The structured basis of a decision: ordered rules that can be run against a record.

Pilot 1 uses this form twice. The simulator states the rule it took from an explanation
in this form (its `basis` field), and in the structured-explanation condition the
explainer writes its explanation in this form. Either way the basis is run against
each hidden variant's structured record, which gives a decision that involves no model.

A basis is an ordered list of rules plus a default outcome. The first rule whose
conditions all hold decides; if none holds, the default decides. The basis cannot
decide (the result is None) when a condition needs a fact the record lacks (other than
an `is_missing` check), or when no rule holds and there is no default.

The fields and outcomes a basis may use form a BasisLanguage, one per decision domain.
REFUND_BASIS is the refund benchmark's, and the default everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sysone.benchmark.records import Condition, Record
from sysone.benchmark.refund_policy import PolicyParams, counted_days

OUTCOMES = ("approve", "deny", "undetermined")

FIELDS = {
    "manager_preapproval": "whether a store manager pre-approved the refund in writing (true or false)",
    "final_sale": "whether the item is marked final sale (true or false)",
    "has_receipt": "whether a receipt or other proof of purchase is on file (true or false)",
    "condition": 'item condition: "unopened", "opened" or "defective"',
    "days_since_purchase": "days from the purchase date to the refund request",
    "days_net_of_carrier_delay": "days from the purchase date to the refund request, minus documented carrier delay",
    "price_usd": "price paid, in dollars",
}
BOOL_FIELDS = frozenset({"manager_preapproval", "final_sale", "has_receipt"})
NUMBER_FIELDS = frozenset({"days_since_purchase", "days_net_of_carrier_delay", "price_usd"})

CONDITION_VALUES = tuple(str(c) for c in Condition)

OPERATORS = ("equals", "not_equals", "at_most", "more_than", "is_missing")
ORDER_OPERATORS = frozenset({"at_most", "more_than"})

_VALUE_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]}


class BasisError(ValueError):
    pass


class BasisLanguage:
    """The fields and outcomes a basis may use, and how to read a field from a record.

    Fields are true/false (bool_fields), numbers (number_fields), or one of a fixed set
    of strings (enum_fields). `get(record, field)` returns None when the fact is missing.
    """

    def __init__(
        self,
        fields: dict[str, str],
        outcomes: tuple[str, ...],
        get: Callable[[Any, str], Any],
        *,
        bool_fields: frozenset[str] = frozenset(),
        number_fields: frozenset[str] = frozenset(),
        enum_fields: dict[str, tuple[str, ...]] | None = None,
    ):
        self.fields = fields
        self.outcomes = outcomes
        self.get = get
        self.bool_fields = bool_fields
        self.number_fields = number_fields
        self.enum_fields = enum_fields or {}
        unknown = (bool_fields | number_fields | set(self.enum_fields)) - set(fields)
        if unknown:
            raise ValueError(f"typed fields {sorted(unknown)} are not in the field list")
        # Shown to any model asked to write a basis. Providers without structured output
        # (Bedrock) get only this text, so it spells out the whole format.
        self.instructions = f"""A basis is an ordered list of rules and a default outcome, as JSON:
{{"rules": [{{"conditions": [{{"field": ..., "operator": ..., "value": ...}}, ...], "outcome": ...}}, ...],
 "default_outcome": ...}}
Rules are checked in order. The first rule whose conditions all hold decides the outcome. If no rule holds, the default outcome decides; use null if there is none.
Fields:
{chr(10).join(f'- "{name}": {desc}' for name, desc in fields.items())}
Operators: "equals", "not_equals", "at_most" and "more_than" (numbers only), and "is_missing" (the fact is not in the case file; its value is null).
Outcomes: {", ".join(f'"{o}"' for o in outcomes)}."""
        outcome_schema = {"type": "string", "enum": list(outcomes)}
        self.schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["rules", "default_outcome"],
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["conditions", "outcome"],
                        "properties": {
                            "conditions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["field", "operator", "value"],
                                    "properties": {
                                        "field": {"type": "string", "enum": list(fields)},
                                        "operator": {"type": "string", "enum": list(OPERATORS)},
                                        "value": _VALUE_SCHEMA,
                                    },
                                },
                            },
                            "outcome": outcome_schema,
                        },
                    },
                },
                "default_outcome": {"anyOf": [outcome_schema, {"type": "null"}]},
            },
        }

    def parse_clause(self, obj: Any) -> Clause:
        if not isinstance(obj, dict):
            raise BasisError("a condition is not an object")
        field, op, value = obj.get("field"), obj.get("operator"), obj.get("value")
        if field not in self.fields:
            raise BasisError(f"unknown field {field!r}")
        if op not in OPERATORS:
            raise BasisError(f"unknown operator {op!r}")
        if op == "is_missing":
            return Clause(field, op, None)
        if op in ORDER_OPERATORS and field not in self.number_fields:
            raise BasisError(f"{op} needs a number field, not {field}")
        if field in self.bool_fields and not isinstance(value, bool):
            raise BasisError(f"{field} takes true or false, not {value!r}")
        if field in self.number_fields and not _is_number(value):
            raise BasisError(f"{field} takes a number, not {value!r}")
        if field in self.enum_fields and value not in self.enum_fields[field]:
            *rest, last = self.enum_fields[field]
            raise BasisError(f"{field} takes {', '.join(rest)} or {last}, not {value!r}")
        return Clause(field, op, value)


@dataclass(frozen=True)
class Clause:
    field: str
    operator: str
    value: Any


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Clause, ...]
    outcome: str


@dataclass(frozen=True)
class Basis:
    rules: tuple[Rule, ...]
    default_outcome: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [
                {"conditions": [vars(c) for c in r.conditions], "outcome": r.outcome} for r in self.rules
            ],
            "default_outcome": self.default_outcome,
        }


@dataclass(frozen=True)
class Executed:
    decision: str | None
    # Index of the rule that decided, or None for the default (or no decision).
    rule_index: int | None
    # Why the basis could not decide, when it couldn't.
    note: str | None = None


def _is_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def parse_basis(obj: Any, lang: BasisLanguage | None = None) -> Basis:
    """Validate a basis from a model's JSON. Raises BasisError if it can't be run."""
    lang = lang or REFUND_BASIS
    outcomes = lang.outcomes
    if not isinstance(obj, dict):
        raise BasisError("basis is not an object")
    rules = obj.get("rules")
    if not isinstance(rules, list):
        raise BasisError("basis has no list of rules")
    parsed = []
    for r in rules:
        if not isinstance(r, dict) or not isinstance(r.get("conditions"), list):
            raise BasisError("a rule has no list of conditions")
        if r.get("outcome") not in outcomes:
            raise BasisError(f"rule outcome {r.get('outcome')!r} not in {outcomes}")
        parsed.append(Rule(tuple(lang.parse_clause(c) for c in r["conditions"]), r["outcome"]))
    default = obj.get("default_outcome")
    if default is not None and default not in outcomes:
        raise BasisError(f"default outcome {default!r} not in {outcomes}")
    return Basis(tuple(parsed), default)


def field_value(record: Record, field: str) -> Any:
    """A refund field's value for a record, or None if the fact is missing."""
    if field == "condition":
        return str(record.condition) if record.condition else None
    if field == "days_since_purchase":
        return (record.request_date - record.purchase_date).days if record.purchase_date else None
    if field == "days_net_of_carrier_delay":
        return counted_days(record) if record.purchase_date else None
    return getattr(record, field)


class _Undecidable(Exception):
    pass


def _holds(clause: Clause, record: Any, lang: BasisLanguage) -> bool:
    actual = lang.get(record, clause.field)
    if clause.operator == "is_missing":
        return actual is None
    if actual is None:
        raise _Undecidable(f"{clause.field} is missing from the record")
    if clause.operator == "equals":
        return actual == clause.value
    if clause.operator == "not_equals":
        return actual != clause.value
    if clause.operator == "at_most":
        return actual <= clause.value
    return actual > clause.value


def execute(basis: Basis, record: Any, lang: BasisLanguage | None = None) -> Executed:
    lang = lang or REFUND_BASIS
    try:
        for i, rule in enumerate(basis.rules):
            if all(_holds(c, record, lang) for c in rule.conditions):
                return Executed(rule.outcome, i)
    except _Undecidable as e:
        return Executed(None, None, str(e))
    if basis.default_outcome is None:
        return Executed(None, None, "no rule holds and there is no default outcome")
    return Executed(basis.default_outcome, None)


REFUND_BASIS = BasisLanguage(
    FIELDS, OUTCOMES, field_value,
    bool_fields=BOOL_FIELDS, number_fields=NUMBER_FIELDS, enum_fields={"condition": CONDITION_VALUES},
)
BASIS_INSTRUCTIONS = REFUND_BASIS.instructions
BASIS_SCHEMA = REFUND_BASIS.schema


def policy_basis(params: PolicyParams = PolicyParams()) -> Basis:
    """The refund policy itself as a basis. Running it agrees with refund_policy.evaluate
    on every record, which is what the tests check."""

    def c(field: str, op: str, value: Any = None) -> Clause:
        return Clause(field, op, value)

    days = "days_net_of_carrier_delay"
    return Basis(
        rules=(
            Rule((c("manager_preapproval", "equals", True),), "approve"),
            Rule((c("final_sale", "equals", True),), "deny"),
            Rule((c("has_receipt", "is_missing"),), "undetermined"),
            Rule((c("has_receipt", "equals", False),), "deny"),
            Rule((c("condition", "is_missing"),), "undetermined"),
            Rule((c("condition", "equals", "opened"),), "deny"),
            Rule((c(days, "is_missing"),), "undetermined"),
            Rule((c("condition", "equals", "defective"), c(days, "more_than", params.defect_window_days)), "deny"),
            Rule((c("condition", "equals", "defective"),), "approve"),
            Rule((c(days, "more_than", params.window_days),), "deny"),
        ),
        default_outcome="approve",
    )
