"""What the explainer and simulator prompts need to know about a decision domain.

REFUND_SETTING reproduces the refund prompts exactly, so refund requests (and their
cache keys) are the same as before other domains existed. Other domains get theirs from
sysone.benchmark.adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from sysone.decision.jev import REFUND_TASK, DecisionTask
from sysone.explain.basis import REFUND_BASIS, BasisLanguage


@dataclass(frozen=True)
class ExplainSetting:
    task: DecisionTask
    # "refund request" / "refund requests": what one case is, in prompts.
    case_noun: str
    cases_noun: str
    # "refund policy": what the policy is called, in prompts and as a section heading.
    policy_name: str
    basis: BasisLanguage

    @property
    def labels(self) -> tuple[str, ...]:
        return self.task.labels

    @property
    def policy_heading(self) -> str:
        return self.policy_name[0].upper() + self.policy_name[1:]


REFUND_SETTING = ExplainSetting(REFUND_TASK, "refund request", "refund requests", "refund policy", REFUND_BASIS)
