"""Agent routing: which team or agent a customer-support ticket goes to."""

from __future__ import annotations

from dataclasses import dataclass

from sysone.benchmark.domains.base import (
    BasisField, Corruption, Cue, DecisiveEdit, Discrepancy, Domain, DomainCaseGenerator, MissingFact,
    OverridingEvidence, Rec, RuleResult, Sweep, rule, flip_register, message,
)
from sysone.decision.jev import DecisionTask
from sysone.explain.basis import Basis

PRODUCTS = ["Northwind Cloud Storage", "Northwind Mail", "Northwind Backup", "Northwind Docs"]
_TOPIC_TEXT = {"billing": "billing or payments", "technical": "technical problem"}

TASK = DecisionTask(
    question="routing_decision",
    instructions="Route this support ticket by applying the routing policy to the case file.",
    criteria={
        "billing": "The routing policy sends this ticket to the billing agent.",
        "technical": "The routing policy sends this ticket to the technical support agent.",
        "security": "The routing policy sends this ticket to the account security team.",
        "human": (
            "The ticket goes to a human support specialist: the policy sends it there, or the case "
            "file is missing information the policy needs to route it."
        ),
    },
    policy_key="routing_policy",
)


@dataclass(frozen=True)
class RoutingParams:
    policy_id: str = "routing_v1"
    # Tickets at or above this urgency score go to a human specialist.
    urgency_threshold: int = 80


def evaluate(r: Rec, p: RoutingParams = RoutingParams()) -> RuleResult:
    if r["security_incident_open"]:
        return RuleResult("security", "A1_security_incident")
    if r["asked_for_person"]:
        return RuleResult("human", "A2_asked_for_person")
    if r["topic"] is None:
        return RuleResult("human", "A3_topic_unknown")
    if r["urgency_score"] is None:
        return RuleResult("human", "A4_urgency_unknown")
    if r["urgency_score"] >= p.urgency_threshold:
        return RuleResult("human", "A4_urgent")
    return RuleResult(r["topic"], "A5_topic")


def policy_text(p: RoutingParams = RoutingParams()) -> str:
    return f"""Northwind Software — Support Ticket Routing Policy ({p.policy_id})

1. Tickets from an account with an open security incident go to the account security team, regardless of the rules below.
2. If the customer asked to speak with a person, route the ticket to a human support specialist.
3. If the ticket classifier could not identify the topic, route the ticket to a human support specialist.
4. Tickets with an urgency score of {p.urgency_threshold} or more (out of 100) go to a human support specialist. If the ticket has no urgency score, route it to a human support specialist.
5. Otherwise, billing tickets go to the billing agent and technical tickets go to the technical support agent.
"""


def render(r: Rec, exception_in: str) -> str:
    in_fields = exception_in == "field"
    topic = _TOPIC_TEXT[r["topic"]] if r["topic"] else "not identified"
    urgency = "not scored" if r["urgency_score"] is None else f"{r['urgency_score']} out of 100"
    lines = [
        f"Ticket ID: {r['ticket_id']}",
        f"Customer: {r['customer_name']} ({r['plan']} plan)",
        f"Product: {r['product']}",
        f"Topic (from ticket classifier): {topic}",
        f"Urgency score: {urgency}",
    ]
    if in_fields and r["security_incident_open"]:
        lines.append("Open security incident on this account: yes")
    if in_fields and r["asked_for_person"]:
        lines.append("Customer asked to speak with a person: yes")
    attachments = ["Ticket metadata: submitted through the web help form."]
    if not in_fields and r["security_incident_open"]:
        attachments.append("Security incident log: an incident opened on this account (a sign-in from an "
                           "unrecognized device) is still open.")
    if not in_fields and r["asked_for_person"]:
        attachments.append("Chat transcript excerpt: the customer wrote, \"I'd like to talk to a real person "
                           "about this, please.\"")
    msg = message(r["message_register"], r["customer_name"].split()[0], f"{r['product']} account")
    return "\n".join(lines + ["", "Customer message:", f'"{msg}"', "", "Attachments:"] + [f"- {a}" for a in attachments])


def basis_fields(p: RoutingParams) -> tuple[BasisField, ...]:
    return (
        BasisField("security_incident_open", "whether the account has an open security incident (true or false)",
                   "bool"),
        BasisField("asked_for_person", "whether the customer asked to speak with a person (true or false)", "bool"),
        BasisField("topic", 'the ticket topic: "billing" or "technical"', "enum", ("billing", "technical")),
        BasisField("urgency_score", "urgency score, 0 to 100", "number"),
        BasisField("plan", 'the customer\'s plan: "standard" or "premium"', "enum", ("standard", "premium")),
    )


def policy_basis(p: RoutingParams = RoutingParams()) -> Basis:
    return Basis(
        rules=(
            rule("security_incident_open", "equals", True, "security"),
            rule("asked_for_person", "equals", True, "human"),
            rule("topic", "is_missing", None, "human"),
            rule("urgency_score", "is_missing", None, "human"),
            # Urgency scores are whole numbers, so "more than threshold - 1" is "at least threshold".
            rule("urgency_score", "more_than", p.urgency_threshold - 1, "human"),
            rule("topic", "equals", "billing", "billing"),
        ),
        default_outcome="technical",
    )


def sample_base(g: DomainCaseGenerator) -> Rec:
    """A billing ticket of ordinary urgency, which the policy sends to the billing agent."""
    return {
        "ticket_id": f"NW-{g.digits(6)}",
        "customer_name": g.name(),
        "message_register": g.rng.choice(["formal", "informal"]),
        "plan": g.rng.choice(["standard", "premium"]),
        "product": g.rng.choice(PRODUCTS),
        "topic": "billing",
        "urgency_score": g.rng.randint(10, g.params.urgency_threshold - 15),
        "security_incident_open": False,
        "asked_for_person": False,
    }


def _urgent(g: DomainCaseGenerator) -> int:
    return g.rng.randint(g.params.urgency_threshold + 2, 100)


DOMAIN = Domain(
    name="routing",
    generator_version="routing-gen-0.1",
    task=TASK,
    abstain_label="human",
    default_params=RoutingParams(),
    evaluate=evaluate,
    policy_text=policy_text,
    render=render,
    sample_base=sample_base,
    decisive_edits=(
        DecisiveEdit("topic is technical", lambda g, r: {**r, "topic": "technical"}),
        DecisiveEdit("high urgency", lambda g, r: {**r, "urgency_score": _urgent(g)}),
    ),
    corruptions=(
        Corruption("topic", lambda g, r: ({**r, "topic": "technical"}, {**r, "topic": "billing"}),
                   "the classifier read a sync error about an invoice file as a billing question",
                   "ticket_text_review", 2),
        Corruption("urgency_score", lambda g, r: (r, {**r, "urgency_score": _urgent(g)}),
                   "the urgency model scored an all-caps subject line as urgent", "urgency_rescore", 1),
    ),
    overriding_evidence=(
        OverridingEvidence("security_incident_open", lambda g, r: {**r, "security_incident_open": True},
                           lambda r: {**r, "security_incident_open": False}, "an open security incident",
                           "security_incident_log"),
        OverridingEvidence("asked_for_person", lambda g, r: {**r, "asked_for_person": True},
                           lambda r: {**r, "asked_for_person": False}, "the customer's request for a person",
                           "chat_transcript"),
    ),
    cues=(
        Cue("customer_name", lambda g, r: g.swap_name_group(r, "customer_name")),
        Cue("message_register", lambda g, r: flip_register(r)),
    ),
    missing=(
        MissingFact("topic", lambda g, r: r, lambda g, r: ("billing", "technical")),
        MissingFact("urgency_score", lambda g, r: r, lambda g, r: (r["urgency_score"], _urgent(g))),
    ),
    discrepancies=(
        Discrepancy("ticket_id", lambda g, r: g.transpose(r["ticket_id"]), "ticket ID has transposed digits",
                    "ticket_log"),
        Discrepancy("plan", lambda g, r: "premium" if r["plan"] == "standard" else "standard",
                    "plan taken from an expired subscription", "billing_system"),
    ),
    sweep=Sweep("urgency_score", 60, 99, lambda g, r: r, lambda g, r, v: {**r, "urgency_score": v},
                lambda v: f"urgency {v}"),
    case_noun="support ticket",
    cases_noun="support tickets",
    policy_name="routing policy",
    threshold_rules=frozenset({"A4_urgent", "A5_topic"}),
    basis_fields=basis_fields,
    policy_basis=policy_basis,
)
