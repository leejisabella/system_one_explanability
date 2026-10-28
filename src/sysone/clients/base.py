"""Provider interface and the guarded client that every outside call goes through."""

from __future__ import annotations

import dataclasses
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal

from sysone.budget.guard import CostGuard
from sysone.budget.pricing import Price, worst_case_cost
from sysone.clients.cache import ResponseCache
from sysone.types import ChatRequest, ChatResponse, ProviderResult, Request


class ModelNotConfigured(RuntimeError):
    """Raised for placeholder model IDs (those starting with "TODO")."""


class ModelVersionChanged(RuntimeError):
    """Raised when a response comes from a different model version than the run is locked to."""

    def __init__(self, expected: str, got: str):
        super().__init__(f"model version changed from {expected} to {got}; the run is locked to {expected}")
        self.expected = expected
        self.got = got


class VersionLock:
    """Keeps every response in a run on one model version.

    An alias such as jev-latest moves when a new release ships. The lock records the
    first version that answers (unless one is given up front) and raises
    ModelVersionChanged for any later response from a different version.
    """

    def __init__(self, expected: str | None = None, *, on_lock: Callable[[str], None] | None = None):
        self.expected = expected
        self._on_lock = on_lock
        self._lock = threading.Lock()

    def check(self, model_used: str) -> None:
        with self._lock:
            if self.expected is None:
                self.expected = model_used
                newly_locked = True
            elif model_used != self.expected:
                raise ModelVersionChanged(self.expected, model_used)
            else:
                newly_locked = False
        if newly_locked and self._on_lock is not None:
            self._on_lock(model_used)


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
    def send(self, model: str, request: Request, price: Price) -> ProviderResult:
        """Make one request. Raise ProviderError on failure."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model: str
    # Standard price. Worst-case estimates always use it, even when a cheaper tier is requested.
    price: Price | None
    # Per-model request settings from configs/models.yaml, applied to chat requests that
    # don't set them: the service tier (e.g. "flex") and the reasoning level.
    service_tier: str | None = None
    reasoning_effort: str | None = None
    # Price of service_tier, charged only when the provider reports serving that tier.
    tier_price: Price | None = None

    @property
    def is_placeholder(self) -> bool:
        return self.model.startswith("TODO")


class GuardedClient:
    """Wraps one provider/model with the cache, cost guard, retries and spending log.

    Set `version_lock` to stop on any response, fresh or cached, from a different model
    version than the run is locked to. The call is still recorded in the spending log.
    """

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
        version_lock: VersionLock | None = None,
    ):
        self.version_lock = version_lock
        self.spec = spec
        self.provider = provider
        self.guard = guard
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep

    def prepare(self, request: Request) -> Request:
        """The request as sent: chat requests get this model's settings from models.yaml.
        Applied before the cache key, so answers from different settings never mix."""
        if not isinstance(request, ChatRequest):
            return request
        return dataclasses.replace(
            request,
            service_tier=request.service_tier or self.spec.service_tier,
            reasoning_effort=request.reasoning_effort or self.spec.reasoning_effort,
        )

    def request_key(self, request: Request) -> str:
        """The cache key of the request as sent, which result rows record to join to the cache."""
        return ResponseCache.key(self.spec.provider, self.spec.model, self.prepare(request))

    def complete(self, request: Request) -> ChatResponse:
        if self.spec.is_placeholder:
            raise ModelNotConfigured(
                f"model alias '{self.spec.alias}' is a placeholder ({self.spec.model}); "
                "set a real model ID in configs/models.yaml"
            )
        if self.spec.price is None:
            raise ModelNotConfigured(f"no price for {self.spec.provider}/{self.spec.model}")
        if self.spec.service_tier is not None and self.spec.tier_price is None:
            raise ModelNotConfigured(
                f"no {self.spec.service_tier} price for {self.spec.provider}/{self.spec.model}"
            )
        request = self.prepare(request)
        key = ResponseCache.key(self.spec.provider, self.spec.model, request)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            if self.version_lock is not None:
                self.version_lock.check(hit.model_used)
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

            requested_tier = request.service_tier if isinstance(request, ChatRequest) else None
            if result.reported_cost_usd is not None:
                cost, source = result.reported_cost_usd, "provider_reported"
            elif result.usage.input_tokens == 0 and result.usage.output_tokens == 0:
                # No usage data came back, so the real cost is unknown: charge the worst case.
                cost, source = worst_case, "worst_case_missing_usage"
            elif requested_tier is not None and result.service_tier == requested_tier and self.spec.tier_price:
                cost, source = self.spec.tier_price.cost(result.usage), f"price_table_{requested_tier}"
            else:
                # Includes a requested tier the provider didn't confirm serving: charged at standard.
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
                service_tier_requested=requested_tier,
                service_tier_served=result.service_tier,
            )
            # Checked after the spending log is written, so a mismatched call is still
            # accounted for, and before caching, so its answer is never reused.
            if self.version_lock is not None:
                self.version_lock.check(result.model_used)
            response = ChatResponse(
                text=result.text,
                usage=result.usage,
                cost_usd=cost,
                provider=self.spec.provider,
                model_requested=self.spec.model,
                model_used=result.model_used,
                finish_reason=result.finish_reason,
                raw=result.raw,
                service_tier=result.service_tier,
            )
            if self.cache is not None:
                self.cache.put(key, response)
            return response
