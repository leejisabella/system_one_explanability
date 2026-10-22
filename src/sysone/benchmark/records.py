"""Refund request records and decision outcomes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    # The available facts do not determine the outcome under the policy.
    UNDETERMINED = "undetermined"


class Condition(StrEnum):
    UNOPENED = "unopened"
    OPENED = "opened"
    DEFECTIVE = "defective"


@dataclass(frozen=True)
class Record:
    """One refund request. None means the fact is missing from the record."""

    customer_name: str
    # Formal or informal phrasing of the customer's message. Not policy-relevant.
    message_register: str
    order_id: str
    item: str
    category: str
    price_usd: float
    purchase_date: date | None
    request_date: date
    condition: Condition | None
    has_receipt: bool | None
    final_sale: bool
    # Days of carrier delay documented for the delivery; they don't count against the window.
    shipping_delay_days: int
    # Written pre-approval from a store manager, which overrides the other rules.
    manager_preapproval: bool

    def replace(self, **changes: Any) -> Record:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["purchase_date"] = self.purchase_date.isoformat() if self.purchase_date else None
        d["request_date"] = self.request_date.isoformat()
        d["condition"] = str(self.condition) if self.condition else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        return cls(
            **{
                **d,
                "purchase_date": date.fromisoformat(d["purchase_date"]) if d["purchase_date"] else None,
                "request_date": date.fromisoformat(d["request_date"]),
                "condition": Condition(d["condition"]) if d["condition"] else None,
            }
        )
