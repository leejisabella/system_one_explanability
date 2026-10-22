"""Benchmark cases: true and presented records, ground truth, defects, evidence and variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sysone.benchmark.records import Outcome, Record


class CaseType(StrEnum):
    CORRECT = "correct"
    INCORRECT_INPUT = "incorrect_input"  # Type I
    IGNORED_EVIDENCE = "ignored_evidence"  # Type II
    IRRELEVANT_CUE = "irrelevant_cue"  # Type III
    AMBIGUOUS = "ambiguous"
    HARD_NEGATIVE = "hard_negative"


class VariantKind(StrEnum):
    ORIGINAL = "original"
    # Changes a policy-relevant fact so the policy outcome flips.
    DECISIVE_EDIT = "decisive_edit"
    # Replaces a corrupted field with its true value.
    CORRECTION = "correction"
    # Changes the customer's name or message register; policy outcome unchanged.
    CUE_EDIT = "cue_edit"
    # Moves overriding evidence from an attachment into the case fields; outcome unchanged.
    EXCEPTION_SALIENT = "exception_salient"
    # Removes the overriding evidence; outcome flips.
    EXCEPTION_REMOVED = "exception_removed"
    # Fixes a discrepancy that doesn't affect the outcome.
    DISCREPANCY_FIXED = "discrepancy_fixed"
    # Fills in a missing fact.
    RESOLUTION = "resolution"


@dataclass(frozen=True)
class Defect:
    # "I" incorrect input, "II" neglected overriding evidence, "III" irrelevant cue,
    # "irrelevant_discrepancy" a real error that doesn't change the outcome.
    type: str
    field: str
    presented_value: Any
    true_value: Any
    description: str


@dataclass(frozen=True)
class EvidenceItem:
    """An independent source that can confirm a fact, available only under some access conditions."""

    source: str
    field: str
    value: Any
    cost: int


@dataclass(frozen=True)
class Variant:
    variant_id: str
    kind: VariantKind
    record: Record
    # Where overriding evidence (carrier delay, manager pre-approval) appears in the case text.
    exception_in: str  # "field" | "attachment"
    description: str
    # The policy applied to this variant's record.
    policy_outcome: Outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "kind": str(self.kind),
            "record": self.record.to_dict(),
            "exception_in": self.exception_in,
            "description": self.description,
            "policy_outcome": str(self.policy_outcome),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Variant:
        return cls(
            variant_id=d["variant_id"],
            kind=VariantKind(d["kind"]),
            record=Record.from_dict(d["record"]),
            exception_in=d["exception_in"],
            description=d["description"],
            policy_outcome=Outcome(d["policy_outcome"]),
        )


@dataclass(frozen=True)
class Case:
    case_id: str
    case_type: CaseType
    policy_id: str
    true_record: Record
    presented_record: Record
    true_outcome: Outcome
    # The policy applied to the presented record.
    presented_outcome: Outcome
    decisive_rule: str
    defects: tuple[Defect, ...]
    evidence: tuple[EvidenceItem, ...]
    # The first variant is always the original presentation of the case.
    variants: tuple[Variant, ...]
    # Stretch-goal annotations: the decision that would harm the customer under
    # ground truth (deny a refund that is owed), and how badly.
    harmful_decision: Outcome | None
    harm_severity: str | None

    @property
    def original(self) -> Variant:
        return self.variants[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_type": str(self.case_type),
            "policy_id": self.policy_id,
            "true_record": self.true_record.to_dict(),
            "presented_record": self.presented_record.to_dict(),
            "true_outcome": str(self.true_outcome),
            "presented_outcome": str(self.presented_outcome),
            "decisive_rule": self.decisive_rule,
            "defects": [vars(d) for d in self.defects],
            "evidence": [vars(e) for e in self.evidence],
            "variants": [v.to_dict() for v in self.variants],
            "harmful_decision": str(self.harmful_decision) if self.harmful_decision else None,
            "harm_severity": self.harm_severity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        return cls(
            case_id=d["case_id"],
            case_type=CaseType(d["case_type"]),
            policy_id=d["policy_id"],
            true_record=Record.from_dict(d["true_record"]),
            presented_record=Record.from_dict(d["presented_record"]),
            true_outcome=Outcome(d["true_outcome"]),
            presented_outcome=Outcome(d["presented_outcome"]),
            decisive_rule=d["decisive_rule"],
            defects=tuple(Defect(**x) for x in d["defects"]),
            evidence=tuple(EvidenceItem(**x) for x in d["evidence"]),
            variants=tuple(Variant.from_dict(v) for v in d["variants"]),
            harmful_decision=Outcome(d["harmful_decision"]) if d["harmful_decision"] else None,
            harm_severity=d["harm_severity"],
        )
