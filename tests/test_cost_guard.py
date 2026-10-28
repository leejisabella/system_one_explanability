import dataclasses

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
from sysone.clients.base import (
    GuardedClient,
    ModelNotConfigured,
    ModelSpec,
    ModelVersionChanged,
    ProviderError,
    VersionLock,
)
from sysone.clients.cache import ResponseCache
from sysone.clients.fake import FakeProvider, ok_result, typesafe_result
from sysone.types import ChatRequest, SystemOneRequest

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


# --- System One requests and the version lock ---


def s1_req(state="the case", sample_index=0):
    return SystemOneRequest(state=state, questions={"q": {"type": "noul"}}, sample_index=sample_index)


def test_system_one_estimate_counts_state_and_questions():
    small = estimate_input_tokens(s1_req("x"))
    big = estimate_input_tokens(s1_req("x" * 3000))
    assert big - small >= 1000  # at least one token per 3 characters of state
    long_question = SystemOneRequest(state="x", questions={"q": {"type": "noul", "instructions": "y" * 3000}})
    assert estimate_input_tokens(long_question) - small >= 1000  # questions are billed too


def test_system_one_worst_case_covers_output_if_it_is_ever_billed():
    request = s1_req()
    assert worst_case_cost(request, Price(10.0, 0.0)) == pytest.approx(estimate_input_tokens(request) * 10 / 1e6)
    assert worst_case_cost(request, Price(10.0, 50.0)) > worst_case_cost(request, Price(10.0, 0.0))


def test_system_one_request_needs_typed_questions():
    with pytest.raises(ValueError):
        SystemOneRequest(state="s", questions={})
    with pytest.raises(ValueError):
        SystemOneRequest(state="s", questions={"q": {"instructions": "no type"}})


def test_version_lock_records_first_version_then_holds():
    seen = []
    lock = VersionLock(on_lock=seen.append)
    lock.check("jev-1.14.0")
    lock.check("jev-1.14.0")
    assert seen == ["jev-1.14.0"]
    with pytest.raises(ModelVersionChanged) as exc:
        lock.check("jev-1.15.0")
    assert exc.value.expected == "jev-1.14.0" and exc.value.got == "jev-1.15.0"


def test_version_change_is_logged_but_never_cached(tmp_path):
    provider = FakeProvider([
        typesafe_result({}, model="jev-1.14.0", input_tokens=100),
        typesafe_result({}, model="jev-1.15.0", input_tokens=100),
    ])
    client = make_client(tmp_path, provider, cache=True, price=Price(1.0, 0.0))
    client.version_lock = VersionLock()
    client.complete(s1_req(sample_index=0))
    with pytest.raises(ModelVersionChanged):
        client.complete(s1_req(sample_index=1))
    entries = client.guard.ledger.entries()
    assert [e["model_used"] for e in entries] == ["jev-1.14.0", "jev-1.15.0"]
    assert entries[-1]["cost_usd"] > 0
    # The mismatched answer was not cached, so the same request calls the provider again.
    with pytest.raises(ModelVersionChanged):
        client.complete(s1_req(sample_index=1))
    assert len(provider.calls) == 3


def test_cached_answer_from_another_version_is_refused(tmp_path):
    client = make_client(tmp_path, FakeProvider([typesafe_result({}, model="jev-1.13.0")]), cache=True, price=Price(1.0, 0.0))
    client.complete(s1_req())  # cached under the alias, answered by 1.13.0
    client.version_lock = VersionLock("jev-1.14.0")
    with pytest.raises(ModelVersionChanged):
        client.complete(s1_req())


# --- Project caps per provider ---


def test_project_cap_counts_other_runs_and_refuses_before_sending(tmp_path):
    from sysone.budget import ProjectBudgetExceeded

    guard = CostGuard(Ledger.for_run(tmp_path, "now"), per_call_cap_usd=2.0, run_cap_usd=25.0,
                      project_caps_usd={"bedrock": 5.0}, other_runs_spent_usd={"bedrock": 4.5})
    guard.reserve(0.4, provider="bedrock", model="m", tags={})
    with pytest.raises(ProjectBudgetExceeded):
        guard.reserve(0.2, provider="bedrock", model="m", tags={})
    # Providers are capped separately, and unlisted ones are refused.
    with pytest.raises(ProjectBudgetExceeded, match="no project cap"):
        guard.reserve(0.01, provider="gemini", model="m", tags={})
    # Offline fakes are never capped.
    guard.reserve(0.5, provider="fake", model="m", tags={})


def test_project_cap_includes_this_runs_ledger_on_resume(tmp_path):
    from sysone.budget import ProjectBudgetExceeded

    for _ in range(5):
        Ledger.for_run(tmp_path, "now").append(provider="typesafe", cost_usd=0.98)
    guard = CostGuard(Ledger.for_run(tmp_path, "now"), per_call_cap_usd=2.0, run_cap_usd=25.0,
                      project_caps_usd={"typesafe": 5.0})
    assert guard.project_spent_usd("typesafe") == pytest.approx(4.9)
    with pytest.raises(ProjectBudgetExceeded):
        guard.reserve(0.2, provider="typesafe", model="jev", tags={})


def test_released_and_settled_calls_free_their_project_reservation(tmp_path):
    guard = CostGuard(Ledger.for_run(tmp_path, "now"), per_call_cap_usd=2.0, run_cap_usd=25.0,
                      project_caps_usd={"bedrock": 1.0})
    r = guard.reserve(0.9, provider="bedrock", model="m", tags={})
    guard.release(r, status="error")
    r = guard.reserve(0.9, provider="bedrock", model="m", tags={})
    guard.settle(r, cost_usd=0.1, cost_source="price_table")
    assert guard.project_spent_usd("bedrock") == pytest.approx(0.1)
    guard.reserve(0.9, provider="bedrock", model="m", tags={})


# --- Service tiers ---


def tier_client(tmp_path, served):
    flex = Price(input_per_m=5.0, output_per_m=25.0)
    spec = ModelSpec("sim", "fake", "m", PRICE, service_tier="flex", reasoning_effort="low", tier_price=flex)
    result = ok_result(input_tokens=1000, output_tokens=1000)
    provider = FakeProvider([dataclasses.replace(result, service_tier=served)])
    return GuardedClient(spec, provider, make_guard(tmp_path), sleep=lambda s: None), provider


def test_flex_price_only_when_the_provider_confirms_flex(tmp_path):
    client, provider = tier_client(tmp_path, "flex")
    response = client.complete(ChatRequest.from_prompt("x", max_output_tokens=10))
    sent = provider.calls[0][1]
    assert sent.service_tier == "flex" and sent.reasoning_effort == "low"
    assert response.cost_usd == pytest.approx((1000 * 5.0 + 1000 * 25.0) / 1e6)
    entry = client.guard.ledger.entries()[-1]
    assert entry["cost_source"] == "price_table_flex" and entry["service_tier_served"] == "flex"


@pytest.mark.parametrize("served", ["default", None])
def test_unconfirmed_flex_is_charged_at_the_standard_price(tmp_path, served):
    client, _ = tier_client(tmp_path / str(served), served)
    response = client.complete(ChatRequest.from_prompt("x", max_output_tokens=10))
    assert response.cost_usd == pytest.approx((1000 * 10.0 + 1000 * 50.0) / 1e6)


def test_worst_case_uses_the_standard_price_even_for_flex(tmp_path):
    client, _ = tier_client(tmp_path, "flex")
    client.complete(ChatRequest.from_prompt("x", max_output_tokens=10))
    entry = client.guard.ledger.entries()[-1]
    request = ChatRequest.from_prompt("x", max_output_tokens=10, service_tier="flex", reasoning_effort="low")
    assert entry["worst_case_usd"] == pytest.approx(worst_case_cost(request, PRICE))


def test_a_tier_without_a_price_is_refused(tmp_path):
    spec = ModelSpec("sim", "fake", "m", PRICE, service_tier="flex", tier_price=None)
    provider = FakeProvider()
    with pytest.raises(ModelNotConfigured):
        GuardedClient(spec, provider, make_guard(tmp_path)).complete(ChatRequest.from_prompt("x", max_output_tokens=5))
    assert provider.calls == []


def test_recorded_request_key_matches_the_cache_after_model_settings(tmp_path):
    from sysone.explain.simulator import Simulator

    spec = ModelSpec("sim", "fake", "m", PRICE, service_tier="flex", reasoning_effort="low",
                     tier_price=Price(input_per_m=5.0, output_per_m=25.0))
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = GuardedClient(spec, FakeProvider(), make_guard(tmp_path), cache, sleep=lambda s: None)
    sim = Simulator(client)
    result = sim.simulate("P", "O", "deny", None, "V")
    assert cache.get(result.request_key) is not None
