"""One interface to a frozen benchmark's domain, for the experiment runners and analyses.

Refund benchmarks keep their own records and generator, and the other domains share
sysone.benchmark.domains; both look the same through a BenchmarkAdapter: the policy
text, the case text of a variant, the rule that decides a record, Jev's question, the
explainer and simulator prompt setting, and the labels the analyses need.

A benchmark's meta.json names its domain; benchmarks frozen before domains existed
have none and are refund benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sysone.benchmark.domains import DOMAINS
from sysone.benchmark.domains.base import Domain, basis_language, missing_fact_rule
from sysone.benchmark.refund_policy import PolicyParams, evaluate, policy_text
from sysone.benchmark.render import render_case_text
from sysone.decision.jev import DecisionTask
from sysone.explain.setting import REFUND_SETTING, ExplainSetting

REFUND = "refund"


@dataclass(frozen=True)
class BenchmarkAdapter:
    domain: str
    policy: str
    setting: ExplainSetting
    # The option the policy gives when the case file lacks a fact it needs.
    abstain_label: str
    # The option whose probability the Pilot 0 analysis tracks (refund: approve).
    reference_label: str
    # Rules decided by comparing a quantity with the policy's cutoff, reported apart.
    threshold_rules: frozenset[str]
    threshold_group: str
    # Who a harmful decision harms, or None where no decision harms the person in the case.
    harmed_party: str | None
    _render: Callable[[Any], str]
    _evaluate: Callable[[Any], tuple[str, str]]

    @property
    def task(self) -> DecisionTask:
        return self.setting.task

    @property
    def labels(self) -> tuple[str, ...]:
        return self.setting.labels

    def render(self, variant: Any) -> str:
        return self._render(variant)

    def outcome(self, record: Any) -> str:
        return self._evaluate(record)[0]

    def rule(self, record: Any) -> str:
        return self._evaluate(record)[1]

    @staticmethod
    def undetermined(rule: str | None) -> bool:
        """Whether the rule decides because a fact is missing, so the case has no determined outcome."""
        return missing_fact_rule(rule)

    def rule_group(self, rule: str | None) -> str:
        return self.threshold_group if rule in self.threshold_rules else "other rules"

    def harm_category(self, decision: str, truth: str, harmful: str | None) -> str:
        """Who a wrong decision harms."""
        if self.domain == REFUND:
            # The labels refund analyses (and E0's figures) already use.
            if decision == "deny" and truth == "approve":
                return "customer: wrongly denied"
            return "store: wrongly approved" if decision == "approve" else "neither: answered undetermined"
        if decision == self.abstain_label:
            return f"neither: answered {decision}"
        if harmful is not None and decision == harmful:
            return f"{self.harmed_party}: harmed by {decision}"
        return f"other: {decision} instead of {truth}"


def refund_adapter(params: PolicyParams = PolicyParams()) -> BenchmarkAdapter:
    def ev(record: Any) -> tuple[str, str]:
        result = evaluate(record, params)
        return str(result.outcome), result.rule

    return BenchmarkAdapter(
        domain=REFUND,
        policy=policy_text(params),
        setting=REFUND_SETTING,
        abstain_label="undetermined",
        reference_label="approve",
        # Days counted against the 30- or 90-day window, net of carrier delay. Jev's own
        # docs list date arithmetic as unreliable.
        threshold_rules=frozenset({"R5_within_window", "R5_outside_window"}),
        threshold_group="date window",
        harmed_party="customer",
        _render=render_case_text,
        _evaluate=ev,
    )


def domain_adapter(domain: Domain, params: Any = None) -> BenchmarkAdapter:
    params = params if params is not None else domain.default_params

    def ev(record: Any) -> tuple[str, str]:
        result = domain.evaluate(record, params)
        return result.label, result.rule

    setting = ExplainSetting(domain.task, domain.case_noun, domain.cases_noun, domain.policy_name,
                             basis_language(domain, params))
    return BenchmarkAdapter(
        domain=domain.name,
        policy=domain.policy_text(params),
        setting=setting,
        abstain_label=domain.abstain_label,
        reference_label=domain.task.labels[0],
        threshold_rules=domain.threshold_rules,
        threshold_group=f"{domain.sweep.field} threshold",
        harmed_party=domain.harmed_party,
        _render=lambda variant: domain.render(variant.record, variant.exception_in),
        _evaluate=ev,
    )


def adapter_for(meta: dict[str, Any]) -> BenchmarkAdapter:
    """The adapter for a frozen benchmark, from its meta.json."""
    name = meta.get("domain", REFUND)
    if name == REFUND:
        return refund_adapter(PolicyParams(**meta["policy"]))
    if name not in DOMAINS:
        raise ValueError(f"benchmark {meta.get('name')} names unknown domain {name!r}")
    domain = DOMAINS[name]
    return domain_adapter(domain, type(domain.default_params)(**meta["policy"]))
