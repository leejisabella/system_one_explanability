"""Refund-eligibility policy family: stated text and the programmatic ground truth.

The rules are applied in order; the first rule that decides the case wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from sysone.benchmark.records import Condition, Outcome, Record


@dataclass(frozen=True)
class PolicyParams:
    policy_id: str = "refund_v0"
    window_days: int = 30
    defect_window_days: int = 90


@dataclass(frozen=True)
class PolicyResult:
    outcome: Outcome
    # ID of the rule that decided the case.
    rule: str


def evaluate(record: Record, params: PolicyParams = PolicyParams()) -> PolicyResult:
    if record.manager_preapproval:
        return PolicyResult(Outcome.APPROVE, "R1_manager_preapproval")
    if record.final_sale:
        return PolicyResult(Outcome.DENY, "R2_final_sale")
    if record.has_receipt is None:
        return PolicyResult(Outcome.UNDETERMINED, "R3_receipt_unknown")
    if not record.has_receipt:
        return PolicyResult(Outcome.DENY, "R3_no_receipt")
    if record.condition is None:
        return PolicyResult(Outcome.UNDETERMINED, "R4_condition_unknown")
    if record.condition == Condition.OPENED:
        return PolicyResult(Outcome.DENY, "R4_opened")
    if record.purchase_date is None:
        return PolicyResult(Outcome.UNDETERMINED, "R5_purchase_date_unknown")
    window = params.defect_window_days if record.condition == Condition.DEFECTIVE else params.window_days
    if counted_days(record) > window:
        return PolicyResult(Outcome.DENY, "R5_outside_window")
    return PolicyResult(Outcome.APPROVE, "R5_within_window")


def counted_days(record: Record) -> int:
    """Days since purchase that count against the return window."""
    assert record.purchase_date is not None
    return (record.request_date - record.purchase_date).days - record.shipping_delay_days


def policy_text(params: PolicyParams = PolicyParams()) -> str:
    return f"""Larkspur Home Goods — Refund Policy ({params.policy_id})

1. A refund that a store manager has pre-approved in writing is approved, regardless of the rules below.
2. Items marked final sale are not refundable.
3. A receipt or other proof of purchase is required.
4. Opened items are refundable only if they are defective. Unopened items are refundable.
5. Refund requests must be made within {params.window_days} days of purchase, or within {params.defect_window_days} days for defective items. Days of documented carrier delay in delivering the item do not count toward this limit.
"""
