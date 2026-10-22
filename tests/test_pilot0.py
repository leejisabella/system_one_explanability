import json

import pytest

from sysone.analysis.pilot0 import analyze_run, majority, summarize
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.store import freeze_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.clients.fake import FakeDecisionProvider, FakeProvider, ok_result
from sysone.decision.jev import JevDecisionModel, QueryBudget, QueryBudgetExhausted, parse_decision
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.runs import Run


@pytest.mark.parametrize("text,label", [
    ('{"decision": "approve", "confidence": 0.9}', "approve"),
    ('```json\n{"decision": "Deny", "confidence": 1}\n```', "deny"),
    ('Sure: {"decision": "deny", "confidence": 0.2}', "deny"),
])
def test_parse_valid(text, label):
    d = parse_decision(text)
    assert d.parse_ok and d.label == label


@pytest.mark.parametrize("text", [
    "approve", '{"decision": "maybe", "confidence": 0.5}', '{"decision": "approve", "confidence": 1.5}',
    '{"decision": "approve"}', "{not json}",
])
def test_parse_invalid(text):
    d = parse_decision(text)
    assert not d.parse_ok and d.label is None and d.parse_error


def guarded(tmp_path, provider, run_id="r"):
    guard = CostGuard(Ledger.for_run(tmp_path, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec("jev", "fake", "fake/jev", Price(1.0, 1.0)), provider, guard, sleep=lambda s: None)


def test_query_budget_limits_decisions(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeDecisionProvider(malformed_every=0)))
    budget = QueryBudget(2)
    jev.decide("policy", "case", budget=budget)
    jev.decide("policy", "case", sample_index=1, budget=budget)
    with pytest.raises(QueryBudgetExhausted):
        jev.decide("policy", "case", sample_index=2, budget=budget)


def test_decision_request_is_bounded_and_structured(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeProvider()), max_output_tokens=64)
    req = jev.build_request("POLICY TEXT", "CASE TEXT", sample_index=2)
    assert req.max_output_tokens == 64 and req.sample_index == 2
    assert req.json_schema["properties"]["decision"]["enum"] == ["approve", "deny"]
    assert "POLICY TEXT" in req.messages[0].content and "CASE TEXT" in req.messages[0].content


@pytest.fixture
def small_benchmark(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=2)
    return freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})


def test_dry_run_writes_results_and_resumes(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=2)
    provider = FakeDecisionProvider()
    run = Run("dry", tmp_path / "results")
    status = run_pilot0(config, run, guarded(tmp_path / "results", provider, "dry"), benchmark_dir=small_benchmark)
    rows = run.writer("decisions").rows()
    n_variants = json.loads((small_benchmark / "meta.json").read_text())["n_variants"]
    assert status == "completed" and len(rows) == n_variants * 2
    manifest = run.manifest()
    assert manifest["benchmark"]["sha256"] and manifest["config"]["samples_per_variant"] == 2
    assert "budget.yaml" in manifest["config_files"]

    # A second run with the same ID resumes and makes no new calls.
    calls = provider.calls
    run_pilot0(config, run, guarded(tmp_path / "results", provider, "dry"), benchmark_dir=small_benchmark)
    assert provider.calls == calls and len(run.writer("decisions").rows()) == len(rows)

    summary = analyze_run(run.dir)
    assert summary["n_rows"] == len(rows)
    assert (run.dir / "report.md").exists() and (run.dir / "summary.json").exists()


def test_failed_requests_are_logged_and_retried_on_resume(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1)
    err = ProviderError("bad request", retryable=False, billed="no")
    run = Run("err", tmp_path / "results")
    run_pilot0(config, run, guarded(tmp_path / "results", FakeProvider([err]), "err"), benchmark_dir=small_benchmark)
    rows = run.writer("decisions").rows()
    assert sum(r["status"] == "error" for r in rows) == 1

    provider = FakeDecisionProvider()
    run_pilot0(config, run, guarded(tmp_path / "results", provider, "err"), benchmark_dir=small_benchmark)
    assert provider.calls == 1


def test_budget_halt_is_recorded(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1)
    provider = FakeProvider([ok_result('{"decision": "deny", "confidence": 0.5}', reported_cost_usd=3.0)])
    run = Run("halt", tmp_path / "results")
    status = run_pilot0(config, run, guarded(tmp_path / "results", provider, "halt"), benchmark_dir=small_benchmark)
    assert status.startswith("halted") and run.manifest()["status"] == status
    assert len(provider.calls) == 1


def row(case_id, kind, sample, decision, policy, truth, case_type="correct", model="m", vid=None):
    return {
        "case_id": case_id, "variant_id": vid or f"{case_id}-{kind}", "variant_kind": kind,
        "sample_index": sample, "decision": decision, "parse_ok": decision is not None,
        "policy_outcome": policy, "true_outcome": truth, "case_type": case_type,
        "model_used": model, "cost_usd": 0.01, "cached": False, "status": "ok",
    }


def test_summary_metrics_on_known_rows():
    rows = [
        # Case a: incorrect input. Jev follows the presented record (deny) but truth is approve.
        *[row("a", "original", s, "deny", "deny", "approve", "incorrect_input") for s in range(3)],
        *[row("a", "correction", s, "approve", "approve", "approve", "incorrect_input") for s in range(3)],
        # Case b: correct approve; a cue edit wrongly flips the decision; one sample unparsed.
        *[row("b", "original", s, "approve", "approve", "approve") for s in range(3)],
        row("b", "cue_edit", 0, "deny", "approve", "approve"),
        row("b", "cue_edit", 1, "deny", "approve", "approve"),
        row("b", "cue_edit", 2, None, "approve", "approve"),
    ]
    s = summarize(rows)
    assert s["P0.1_parse"]["k"] == 11 and s["P0.1_parse"]["n"] == 12
    assert s["P0.3_contestable_cases"] == ["a"]
    flips = {(f["variant_kind"], f["expectation"]): f["rate"] for f in s["P0.5_6_flips"]}
    assert flips[("correction", "policy flips")] == 1.0
    assert flips[("cue_edit", "policy unchanged")] == 1.0
    assert s["P0.4_stability"]["unanimous_share"] == 1.0
    assert s["P0.8_cost"]["total_usd"] == pytest.approx(0.12)


def test_majority_ties():
    assert majority(["approve", "deny"]) is None
    assert majority(["approve", "deny", "deny"]) == "deny"
    assert majority([]) is None
