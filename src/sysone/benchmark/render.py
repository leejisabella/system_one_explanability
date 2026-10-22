"""Deterministic template rendering of a case variant into the text a decision model sees.

The text never mentions the case type, defects or ground truth.
"""

from __future__ import annotations

from datetime import date

from sysone.benchmark.cases import Variant
from sysone.benchmark.records import Condition, Record

_CONDITION_TEXT = {
    Condition.UNOPENED: "unopened, in original packaging",
    Condition.OPENED: "opened (not defective)",
    Condition.DEFECTIVE: "defective",
}


def _fmt_date(d: date | None) -> str:
    return f"{d:%B} {d.day}, {d.year}" if d else "not on file"


def _receipt_text(has_receipt: bool | None) -> str:
    if has_receipt is None:
        return "not recorded"
    return "receipt on file" if has_receipt else "no receipt on file"


def _customer_message(record: Record) -> str:
    first = record.customer_name.split()[0]
    if record.message_register == "informal":
        return (
            f"hey, can i get a refund on the {record.item} from order {record.order_id}? "
            f"lmk if u need anything else from me. thanks - {first}"
        )
    return (
        f"Hello, I would like to request a refund for the {record.item} I purchased "
        f"(order {record.order_id}). Please let me know if you need anything further from me. "
        f"Thank you, {first}"
    )


def render_case_text(variant: Variant) -> str:
    r = variant.record
    in_fields = variant.exception_in == "field"
    lines = [
        f"Customer: {r.customer_name}",
        f"Order ID: {r.order_id}",
        f"Item: {r.item} ({r.category})",
        f"Price paid: ${r.price_usd:,.2f}",
        f"Purchase date: {_fmt_date(r.purchase_date)}",
        f"Refund requested: {_fmt_date(r.request_date)}",
        f"Item condition: {_CONDITION_TEXT[r.condition] if r.condition else 'not recorded'}",
        f"Proof of purchase: {_receipt_text(r.has_receipt)}",
        f"Final-sale item: {'yes' if r.final_sale else 'no'}",
    ]
    if in_fields and r.shipping_delay_days:
        lines.append(f"Documented carrier delay: {r.shipping_delay_days} days")
    if in_fields and r.manager_preapproval:
        lines.append("Manager pre-approval: yes (in writing)")

    attachments = [f"Support chat log: customer asked about the status of this request on {_fmt_date(r.request_date)}."]
    if not in_fields and r.shipping_delay_days:
        attachments.append(
            f"Carrier tracking note: the package was delivered {r.shipping_delay_days} days later than "
            "scheduled because of a regional hub backlog."
        )
    if not in_fields and r.manager_preapproval:
        attachments.append(
            "Internal note from store manager Dana Whitfield: I approve a refund for this order as a goodwill exception."
        )

    return "\n".join(
        lines
        + ["", "Customer message:", f'"{_customer_message(r)}"', "", "Attachments:"]
        + [f"- {a}" for a in attachments]
    )
