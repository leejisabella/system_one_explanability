"""Regression tests for issues found in the 2026-10-21 bug review."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from sysone.benchmark.cases import CaseType
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.store import freeze_benchmark
from sysone.budget import CircuitBreakerTripped, CostGuard, Ledger, PerCallCapExceeded, Price
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.clients.bedrock import BedrockProvider
from sysone.clients.cache import ResponseCache
from sysone.clients.fake import FakeDecisionProvider, FakeProvider, ok_result
from sysone.clients.gemini import GeminiProvider
from sysone.decision.jev import parse_decision
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.registry import REPO_ROOT
from sysone.runs import Run
from sysone.types import ChatRequest

PRICE = Price(10.0, 50.0)


def guard_for(tmp_path, run_id="r"):
    return CostGuard(Ledger.for_run(tmp_path, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)


def client_for(tmp_path, provider, *, run_id="r", cache=None, price=PRICE):
    return GuardedClient(ModelSpec("a", "fake", "m", price), provider, guard_for(tmp_path, run_id), cache, sleep=lambda s: None)


def req(**kw):
    return ChatRequest.from_prompt("hi", max_output_tokens=kw.pop("max_output_tokens", 100), **kw)


def test_breaker_trip_survives_resume(tmp_path):
    with pytest.raises(CircuitBreakerTripped):
        client_for(tmp_path, FakeProvider([ok_result(reported_cost_usd=3.0)])).complete(req())
    provider = FakeProvider()
    with pytest.raises(CircuitBreakerTripped):
        client_for(tmp_path, provider).complete(req())
    assert provider.calls == []


def test_missing_usage_is_charged_worst_case(tmp_path):
    client = client_for(tmp_path, FakeProvider([ok_result(input_tokens=0, output_tokens=0)]))
    response = client.complete(req())
    assert response.cost_usd > 0
    assert client.guard.ledger.entries()[-1]["cost_source"] == "worst_case_missing_usage"


class ReasoningProvider(FakeProvider):
    reasoning_allowance_tokens = 32_768


def test_reasoning_allowance_counts_toward_worst_case(tmp_path):
    # 32,768 reasoning tokens at $50/M is $1.64; with 8,000 output tokens the worst case is over $2.
    provider = ReasoningProvider()
    with pytest.raises(PerCallCapExceeded):
        client_for(tmp_path, provider).complete(req(max_output_tokens=8_000))
    # An explicit reasoning budget replaces the allowance.
    client_for(tmp_path, provider, run_id="r2").complete(req(max_output_tokens=8_000, reasoning_max_tokens=0))
    assert len(provider.calls) == 1


def test_bool_confidence_is_rejected():
    assert not parse_decision('{"decision": "approve", "confidence": true}').parse_ok


def test_bedrock_rejects_temperature_with_thinking():
    with pytest.raises(ProviderError):
        BedrockProvider(client=object()).build_body(
            "us.anthropic.claude-opus-4-5-20251101-v1:0", req(temperature=0.5, reasoning_max_tokens=1024)
        )


def test_gemini_timeout_is_unknown_billing():
    def boom(**kwargs):
        raise httpx.ReadTimeout("slow")

    provider = GeminiProvider(client=SimpleNamespace(models=SimpleNamespace(generate_content=boom)))
    with pytest.raises(ProviderError) as exc:
        provider.send("gemini-2.5-flash", req(), PRICE)
    assert exc.value.billed == "unknown" and exc.value.retryable


def test_hard_negative_prices_stay_positive():
    for seed in range(30):
        for c in RefundCaseGenerator(seed).generate(per_type=10):
            if c.case_type == CaseType.HARD_NEGATIVE:
                assert c.presented_record.price_usd > 0


def test_ledger_creates_no_folder_until_written(tmp_path):
    ledger = Ledger.for_run(tmp_path, "unused")
    assert ledger.total_spent() == 0 and not ledger.path.parent.exists()


def test_frozen_benchmark_matches_generator():
    meta = json.loads((REPO_ROOT / "benchmarks" / "refund_v0" / "meta.json").read_text())
    cases = RefundCaseGenerator(meta["seed"]).generate(meta["per_type"])
    body = "".join(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in sorted(cases, key=lambda c: c.case_id))
    assert hashlib.sha256(body.encode()).hexdigest() == meta["sha256"]


@pytest.fixture
def bench(tmp_path):
    return freeze_benchmark(RefundCaseGenerator(seed=2).generate(per_type=2), tmp_path / "bench", {"policy": {}})


def test_consecutive_errors_halt_the_run(tmp_path, bench):
    err = ProviderError("unsupported parameter", retryable=False, billed="no")
    provider = FakeProvider([err] * 100)
    config = Pilot0Config(benchmark=str(bench), samples_per_variant=1, max_consecutive_errors=3)
    run = Run("errs", tmp_path / "results")
    status = run_pilot0(config, run, client_for(tmp_path / "results", provider, run_id="errs"), benchmark_dir=bench)
    assert status.startswith("halted") and len(provider.calls) == 3


class Explodes(FakeProvider):
    def send(self, model, request, price):
        raise ValueError("unexpected response shape")


def test_unexpected_crash_is_recorded_as_failed(tmp_path, bench):
    config = Pilot0Config(benchmark=str(bench), samples_per_variant=1)
    run = Run("crash", tmp_path / "results")
    with pytest.raises(ValueError):
        run_pilot0(config, run, client_for(tmp_path / "results", Explodes(), run_id="crash"), benchmark_dir=bench)
    assert run.manifest()["status"].startswith("failed")


def test_pilot0_does_not_reuse_cache_by_default(tmp_path, bench):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    config = Pilot0Config(benchmark=str(bench), samples_per_variant=1)
    first = FakeDecisionProvider()
    run_pilot0(config, Run("one", tmp_path / "results"), client_for(tmp_path / "results", first, run_id="one", cache=cache), benchmark_dir=bench)
    second = FakeDecisionProvider()
    run_pilot0(config, Run("two", tmp_path / "results"), client_for(tmp_path / "results", second, run_id="two", cache=cache), benchmark_dir=bench)
    assert second.calls == first.calls > 0
