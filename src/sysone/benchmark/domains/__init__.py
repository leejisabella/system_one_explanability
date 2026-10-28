"""Decision domains beyond refunds, built by one shared generator (domains/base.py).

Refunds keep their own generator and record types (sysone.benchmark.generator);
REFUND_TASK in sysone.decision.jev is their question.
"""

from __future__ import annotations

from sysone.benchmark.domains import eligibility, fraud, moderation, routing, tool_selection
from sysone.benchmark.domains.base import Domain, DomainCase, DomainCaseGenerator, render_case_text

DOMAINS: dict[str, Domain] = {
    d.name: d
    for d in (moderation.DOMAIN, fraud.DOMAIN, tool_selection.DOMAIN, routing.DOMAIN, eligibility.DOMAIN)
}

__all__ = ["DOMAINS", "Domain", "DomainCase", "DomainCaseGenerator", "render_case_text"]
