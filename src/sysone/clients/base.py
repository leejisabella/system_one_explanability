"""Provider interface and the guarded client that every outside call goes through."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal

from sysone.budget.guard import CostGuard
from sysone.budget.pricing import Price, worst_case_cost
from sysone.clients.cache import ResponseCache
from sysone.types import ChatRequest, ChatResponse, ProviderResult


class ModelNotConfigured(RuntimeError):
    """Raised for placeholder model IDs (those starting with "TODO")."""


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        billed: Literal["no", "unknown"],
    ):
        super().__init__(message)
        self.retryable = retryable
        # "no": the provider answered with an error, so nothing was charged.
        # "unknown": the request may have run (timeout, dropped connection), so
        # the guard charges it at its worst case.
        self.billed = billed


class Provider(ABC):
    name: str
    # Reasoning tokens assumed in the worst case when a request sets no reasoning budget.
    # Nonzero for providers whose models may reason without being asked.
    reasoning_allowance_tokens: int = 0

    @abstractmethod
    def send(self, model: str, request: ChatRequest, price: Price) -> ProviderResult:
        """Make one request. Raise ProviderError on failure."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model: str
    price: Price | None

    @property
    def is_placeholder(self) -> bool:
        return self.model.startswith("TODO")


class GuardedClient:
    """Wraps one provider/model with the cache, cost guard, retries and spending log."""

    def __init__(
        self,
        spec: ModelSpec,
        provider: Provider,
        guard: CostGuard,
        cache: ResponseCache | None = None,
        *,
        max_retries: int = 2,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.spec = spec
        self.provider = provider
        self.guard = guard
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep

    def complete(self, request: ChatRequest) -> ChatResponse:
        if self.spec.is_placeholder:
            raise ModelNotConfigured(
                f"model alias '{self.spec.alias}' is a placeholder ({self.spec.model}); "
                "set a real model ID in configs/models.yaml"
            )
        if self.spec.price is None:
            raise ModelNotConfigured(f"no price for {self.spec.provider}/{self.spec.model}")

        key = ResponseCache.key(self.spec.provider, self.spec.model, request)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit

        price = self.spec.price
        worst_case = worst_case_cost(request, price, self.provider.reasoning_allowance_tokens)
        attempt = 0
        while True:
            reservation = self.guard.reserve(
                worst_case, provider=self.spec.provider, model=self.spec.model, tags=request.tags
            )
            try:
                result = self.provider.send(self.spec.model, request, price)
            except ProviderError as e:
                details = {"status": "error", "error": str(e), "attempt": attempt}
                if e.billed == "unknown":
                    self.guard.settle(
                        reservation, cost_usd=worst_case, cost_source="worst_case_unknown_outcome", **details
                    )
                else:
                    self.guard.release(reservation, **details)
                if not e.retryable or attempt >= self.max_retries:
                    raise
                attempt += 1
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
                continue
            except BaseException as e:
                # Unexpected failure inside the adapter: charge the worst case to stay safe.
                self.guard.settle(
                    reservation,
                    cost_usd=worst_case,
                    cost_source="worst_case_unknown_outcome",
                    status="error",
                    error=repr(e),
                    attempt=attempt,
                )
                raise

            if result.reported_cost_usd is not None:
                cost, source = result.reported_cost_usd, "provider_reported"
            elif result.usage.input_tokens == 0 and result.usage.output_tokens == 0:
                # No usage data came back, so the real cost is unknown: charge the worst case.
                cost, source = worst_case, "worst_case_missing_usage"
            else:
                cost, source = price.cost(result.usage), "price_table"
            self.guard.settle(
                reservation,
                cost_usd=cost,
                cost_source=source,
                status="ok",
                attempt=attempt,
                model_used=result.model_used,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                reasoning_tokens=result.usage.reasoning_tokens,
                finish_reason=result.finish_reason,
            )
            response = ChatResponse(
                text=result.text,
                usage=result.usage,
                cost_usd=cost,
                provider=self.spec.provider,
                model_requested=self.spec.model,
                model_used=result.model_used,
                finish_reason=result.finish_reason,
                raw=result.raw,
            )
            if self.cache is not None:
                self.cache.put(key, response)
            return response
