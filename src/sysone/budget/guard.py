"""Per-call, per-run and per-provider project spending limits for outside API requests.

Every request is checked before it is sent: its worst-case cost must be within
the per-call cap, the run's spend plus all in-flight worst cases must stay
within the run cap, and, when project caps are set, the provider's spend across
all runs plus in-flight worst cases must stay within that provider's project cap. After the request, the actual cost is recorded. If any
single request turns out to have cost more than the per-call cap, the estimate
was wrong, so the guard trips and refuses every later request in the run.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from sysone.budget.ledger import Ledger


class BudgetError(RuntimeError):
    pass


class PerCallCapExceeded(BudgetError):
    pass


class RunBudgetExceeded(BudgetError):
    pass


class CircuitBreakerTripped(BudgetError):
    pass


class ProjectBudgetExceeded(BudgetError):
    pass


# Offline providers used by tests and dry runs. They cost nothing and need no project cap.
UNCAPPED_PROVIDERS = frozenset({"fake"})


@dataclass(frozen=True)
class Reservation:
    worst_case_usd: float
    provider: str
    model: str
    tags: dict[str, str]


class CostGuard:
    def __init__(
        self,
        ledger: Ledger,
        *,
        per_call_cap_usd: float,
        run_cap_usd: float,
        project_caps_usd: dict[str, float] | None = None,
        other_runs_spent_usd: dict[str, float] | None = None,
    ):
        """project_caps_usd: total allowed per provider across all runs, or None for no
        project caps. other_runs_spent_usd: what each provider has already cost in other
        runs; this run's own spend is read from its ledger."""
        if per_call_cap_usd <= 0 or run_cap_usd <= 0:
            raise ValueError("cost caps must be positive")
        self.ledger = ledger
        self.per_call_cap_usd = per_call_cap_usd
        self.run_cap_usd = run_cap_usd
        self.project_caps_usd = project_caps_usd
        self._spent = ledger.total_spent()
        self._reserved = 0.0
        self._provider_spent: dict[str, float] = dict(other_runs_spent_usd or {})
        for e in ledger.entries():
            provider = e.get("provider", "")
            self._provider_spent[provider] = self._provider_spent.get(provider, 0.0) + e.get("cost_usd", 0.0)
        self._provider_reserved: dict[str, float] = {}
        self._tripped: str | None = None
        self._lock = threading.Lock()
        # A trip is permanent for the run: resuming the same run_id stays halted.
        over = [e for e in ledger.entries() if e.get("cost_usd", 0.0) > per_call_cap_usd]
        if over:
            self._tripped = (
                f"run {ledger.run_id} already halted: a call cost ${over[0]['cost_usd']:.4f}, over the "
                f"per-call cap ${per_call_cap_usd:.2f}; fix the cost estimate and start a new run_id"
            )

    @property
    def spent_usd(self) -> float:
        return self._spent

    def project_spent_usd(self, provider: str) -> float:
        """What this provider has cost across all runs, including this one."""
        return self._provider_spent.get(provider, 0.0)

    def reserve(
        self, worst_case_usd: float, *, provider: str, model: str, tags: dict[str, str]
    ) -> Reservation:
        with self._lock:
            if self._tripped:
                raise CircuitBreakerTripped(self._tripped)
            if worst_case_usd > self.per_call_cap_usd:
                raise PerCallCapExceeded(
                    f"{provider}/{model}: worst-case ${worst_case_usd:.4f} exceeds "
                    f"per-call cap ${self.per_call_cap_usd:.2f}; lower max_output_tokens or input size"
                )
            committed = self._spent + self._reserved
            if committed + worst_case_usd > self.run_cap_usd:
                raise RunBudgetExceeded(
                    f"run {self.ledger.run_id}: ${committed:.4f} spent or in flight, worst-case "
                    f"${worst_case_usd:.4f} more would exceed run cap ${self.run_cap_usd:.2f}"
                )
            if self.project_caps_usd is not None and provider not in UNCAPPED_PROVIDERS:
                cap = self.project_caps_usd.get(provider)
                if cap is None:
                    raise ProjectBudgetExceeded(
                        f"no project cap is set for provider {provider} in configs/budget.yaml; add one to use it"
                    )
                committed = self._provider_spent.get(provider, 0.0) + self._provider_reserved.get(provider, 0.0)
                if committed + worst_case_usd > cap:
                    raise ProjectBudgetExceeded(
                        f"{provider}: ${committed:.4f} spent across all runs or in flight, worst-case "
                        f"${worst_case_usd:.4f} more would exceed the project cap ${cap:.2f}"
                    )
            self._reserved += worst_case_usd
            self._provider_reserved[provider] = self._provider_reserved.get(provider, 0.0) + worst_case_usd
        return Reservation(worst_case_usd, provider, model, dict(tags))

    def settle(self, reservation: Reservation, *, cost_usd: float, cost_source: str, **details: Any) -> None:
        """Record a request that reached the provider and may have been billed."""
        with self._lock:
            self._reserved -= reservation.worst_case_usd
            self._provider_reserved[reservation.provider] -= reservation.worst_case_usd
            self._spent += cost_usd
            self._provider_spent[reservation.provider] = self._provider_spent.get(reservation.provider, 0.0) + cost_usd
            over_cap = cost_usd > self.per_call_cap_usd
            if over_cap:
                self._tripped = (
                    f"{reservation.provider}/{reservation.model} cost ${cost_usd:.4f}, over the "
                    f"per-call cap ${self.per_call_cap_usd:.2f}; the cost estimate is wrong, run halted"
                )
        self._log(reservation, cost_usd=cost_usd, cost_source=cost_source, **details)
        if over_cap:
            raise CircuitBreakerTripped(self._tripped)

    def release(self, reservation: Reservation, **details: Any) -> None:
        """Record a request that the provider rejected without billing."""
        with self._lock:
            self._reserved -= reservation.worst_case_usd
            self._provider_reserved[reservation.provider] -= reservation.worst_case_usd
        self._log(reservation, cost_usd=0.0, cost_source="not_billed", **details)

    def _log(self, reservation: Reservation, **fields: Any) -> None:
        self.ledger.append(
            provider=reservation.provider,
            model_requested=reservation.model,
            worst_case_usd=reservation.worst_case_usd,
            tags=reservation.tags,
            **fields,
        )
