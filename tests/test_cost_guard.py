import pytest

from sysone.budget import (
    CircuitBreakerTripped,
    CostGuard,
    Ledger,
    PerCallCapExceeded,
    Price,
    RunBudgetExceeded,
)
from sysone.budget.pricing import estimate_input_tokens, worst_case_cost
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.cache import ResponseCache
from sysone.clients.fake import FakeProvider, ok_result
from sysone.types import ChatRequest

# $10 / $50 per 1M tokens: 40k output tokens has a worst case of $2.00 on output alone.
PRICE = Price(input_per_m=10.0, output_per_m=50.0)


def make_guard(tmp_path, *, per_call=2.0, run_cap=25.0, run_id="run"):
    return CostGuard(Ledger.for_run(tmp_path, run_id), per_call_cap_usd=per_call, run_cap_usd=run_cap)


def make_client(tmp_path, provider=None, *, guard=None, cache=False, model="m", price=PRICE):
    guard = guard or make_guard(tmp_path)
    return GuardedClient(
        ModelSpec("alias", "fake", model, price),
        provider or FakeProvider(),
        guard,
        ResponseCache(tmp_path / "cache.sqlite") if cache else None,
        sleep=lambda s: None,
    )


def req(prompt="hi", max_output_tokens=100, **kw):
    return ChatRequest.from_prompt(prompt, max_output_tokens=max_output_tokens, **kw)


def test_worst_case_includes_reasoning_budget():
    base = worst_case_cost(req(max_output_tokens=1000), PRICE)
    with_reasoning = worst_case_cost(req(max_output_tokens=1000, reasoning_max_tokens=1000), PRICE)
    assert with_reasoning == pytest.approx(base + 1000 * 50.0 / 1e6)


def test_input_estimate_overestimates_typical_english():
    text = "The quick brown fox jumps over the lazy dog. " * 100  # ~4.5 chars/token in practice
    assert estimate_input_tokens(req(text)) > len(text) / 4


def test_over_cap_call_is_refused_before_sending(tmp_path):
    provider = FakeProvider()
    client = make_client(tmp_path, provider)
    with pytest.raises(PerCallCapExceeded):
        client.complete(req(max_output_tokens=50_000))  # $2.50 worst case
    assert provider.calls == []


def test_run_cap_refuses_once_budget_is_used(tmp_path):
    guard = make_guard(tmp_path, run_cap=0.05)
    provider = FakeProvider([ok_result(reported_cost_usd=0.04)])
    client = make_client(tmp_path, provider, guard=guard)
    client.complete(req("first", max_output_tokens=100))
    with pytest.raises(RunBudgetExceeded):
        client.complete(req("second", max_output_tokens=500))  # $0.04 + ~$0.025 worst case > $0.05
    assert len(provider.calls) == 1


def test_actual_cost_over_cap_trips_breaker_and_halts_run(tmp_path):
    provider = FakeProvider([ok_result(reported_cost_usd=3.00)])
    client = make_client(tmp_path, provider)
    with pytest.raises(CircuitBreakerTripped):
        client.complete(req("a"))
    with pytest.raises(CircuitBreakerTripped):
        client.complete(req("b"))
    assert len(provider.calls) == 1


def test_placeholder_model_is_never_called(tmp_path):
    provider = FakeProvider()
    client = make_client(tmp_path, provider, model="TODO_JEV_MODEL_ID")
    with pytest.raises(ModelNotConfigured):
        client.complete(req())
    assert provider.calls == []


def test_unpriced_model_is_never_called(tmp_path):
    provider = FakeProvider()
    client = make_client(tmp_path, provider, price=None)
    with pytest.raises(ModelNotConfigured):
        client.complete(req())
    assert provider.calls == []


def test_cost_from_price_table_when_provider_does_not_report(tmp_path):
    provider = FakeProvider([ok_result(input_tokens=1000, output_tokens=100)])
    response = make_client(tmp_path, provider).complete(req())
    assert response.cost_usd == pytest.approx((1000 * 10 + 100 * 50) / 1e6)


def test_cache_hit_is_free_and_skips_provider(tmp_path):
    provider = FakeProvider([ok_result(reported_cost_usd=0.01)])
    client = make_client(tmp_path, provider, cache=True)
    client.complete(req())
    second = client.complete(req())
    assert second.cached and second.cost_usd == 0.0
    assert len(provider.calls) == 1


def test_repeat_samples_bypass_cache(tmp_path):
    provider = FakeProvider()
    client = make_client(tmp_path, provider, cache=True)
    client.complete(req(sample_index=0))
    client.complete(req(sample_index=1))
    assert len(provider.calls) == 2


def test_retries_are_separate_calls_and_capped(tmp_path):
    err = ProviderError("429", retryable=True, billed="no")
    provider = FakeProvider([err, err, err, ok_result()])
    client = make_client(tmp_path, provider)
    with pytest.raises(ProviderError):
        client.complete(req())
    assert len(provider.calls) == 3  # first attempt + max_retries=2
    entries = client.guard.ledger.entries()
    assert len(entries) == 3 and all(e["cost_source"] == "not_billed" for e in entries)


def test_unknown_outcome_is_charged_worst_case(tmp_path):
    provider = FakeProvider([ProviderError("timeout", retryable=False, billed="unknown")])
    client = make_client(tmp_path, provider)
    request = req(max_output_tokens=1000)
    with pytest.raises(ProviderError):
        client.complete(request)
    assert client.guard.spent_usd == pytest.approx(worst_case_cost(request, PRICE))


def test_resumed_run_keeps_its_spend(tmp_path):
    client = make_client(tmp_path, FakeProvider([ok_result(reported_cost_usd=1.5)]))
    client.complete(req())
    resumed = make_guard(tmp_path)
    assert resumed.spent_usd == pytest.approx(1.5)


def test_ledger_records_routed_model(tmp_path):
    provider = FakeProvider([ok_result(model="routed/underlying-model", reported_cost_usd=0.001)])
    client = make_client(tmp_path, provider, model="typesafe/router")
    client.complete(req(tags={"case_id": "c1"}))
    [entry] = client.guard.ledger.entries()
    assert entry["model_used"] == "routed/underlying-model"
    assert entry["tags"] == {"case_id": "c1"}
