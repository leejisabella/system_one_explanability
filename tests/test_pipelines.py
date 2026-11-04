"""Latency logging, the explainer's own-decision explanations, and the pipeline cost/latency/quality comparison."""

import json

import pytest

from sysone.analysis.pipelines import pipelines, render_report
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.store import freeze_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.cache import ResponseCache
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider, FakeProvider
from sysone.experiments.judge import JudgeConfig, run_judge
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.experiments.pilot1 import Pilot1Config, run_pilot1
from sysone.explain.explainer import OWN_DECISION, ExplainerModel
from sysone.runs import Run
from sysone.types import ChatRequest

ALIASES = ("explainer_open", "explainer_gemini", "simulator")
CONDITIONS = ("free_form", "structured", "decision_probabilities", OWN_DECISION)


def guarded(results, provider, alias="x", run_id="r", cache=None):
    guard = CostGuard(Ledger.for_run(results, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(1.0, 1.0)), provider, guard,
                         cache=cache, sleep=lambda s: None)


def test_every_call_records_latency_in_the_response_and_ledger(tmp_path):
    client = guarded(tmp_path, FakeProvider(), cache=ResponseCache(tmp_path / "cache.sqlite"))
    request = ChatRequest.from_prompt("hello", max_output_tokens=10)
    first = client.complete(request)
    assert first.latency_s is not None and first.wall_s is not None and first.wall_s >= first.latency_s >= 0
    entry = [json.loads(line) for line in (tmp_path / "runs" / "r" / "ledger.jsonl").read_text().splitlines()][-1]
    assert entry["latency_s"] == first.latency_s and entry["wall_s"] == first.wall_s
    # A cache hit is not a provider call: no latency.
    again = client.complete(request)
    assert again.cached and again.latency_s is None and again.wall_s is None


def test_own_decision_explanations_explain_the_explainers_own_decision(tmp_path):
    explainer = ExplainerModel(guarded(tmp_path, FakeProvider()))
    prompt = explainer.build_explanation_request(OWN_DECISION, "POLICY", "CASE", "approve").messages[-1].content
    assert "You decided the refund request below" in prompt and "## Your decision\napprove" in prompt
    assert "automated decision system" not in prompt
    with pytest.raises(ValueError):
        explainer.build_explanation_request("self", "POLICY", "CASE", "approve")


@pytest.fixture
def runs(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=1)
    bench = freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})
    results = tmp_path / "results"
    p0 = Run("p0", results)
    run_pilot0(Pilot0Config(benchmark=str(bench)), p0,
               guarded(results, FakeDecisionProvider(malformed_every=0), alias="jev", run_id="p0"), benchmark_dir=bench)
    p1 = Run("p1", results)
    clients = {a: guarded(results, FakeChatProvider(malformed_every=0), alias=a, run_id="p1") for a in ALIASES}
    assert run_pilot1(Pilot1Config(conditions=CONDITIONS), p1, clients, pilot0_dir=p0.dir) == "completed"
    j = Run("j", results)
    run_judge(JudgeConfig(), j, guarded(results, FakeChatProvider(malformed_every=0), alias="judge", run_id="j"),
              pilot1_dir=p1.dir)
    return p0, p1, j


def test_own_decision_rows_are_explained_but_never_simulated(runs):
    _, p1, _ = runs
    own = [r for r in p1.writer("explanations").rows() if r["condition"] == OWN_DECISION]
    assert own and all(r["explained_decision"] for r in own)
    assert all(r["latency_s"] is not None for r in p1.writer("explanations").rows() if r.get("raw_text"))
    assert OWN_DECISION not in {r["condition"] for r in p1.writer("simulations").rows()}


def test_pipeline_costs_add_their_calls(runs):
    p0, p1, j = runs
    s = pipelines(p1.dir, j.dir)
    ps = s["pipelines"]
    for name in ("Jev decides", "explainer_open decides", "explainer_open decides and explains",
                 "Jev + explainer_open explains (free_form)", "Jev + explainer_open explains adverse decisions (free_form)"):
        assert name in ps, name
    # Per case: Jev's decision cost (mean of its samples) plus the explanation's cost.
    jev = [json.loads(line) for line in (p0.dir / "decisions.jsonl").read_text().splitlines()]
    expl = {r["explanation_id"]: r for r in p1.writer("explanations").rows()}
    case_id = sorted({r["case_id"] for r in expl.values() if r["condition"] == "free_form"})[0]
    original = next(r["variant_id"] for r in jev if r["case_id"] == case_id and r["variant_kind"] == "original")
    jev_cost = sum(r["cost_usd"] for r in jev if r["variant_id"] == original) / 3
    both = jev_cost + expl[f"explainer_open/free_form/{case_id}"]["cost_usd"]
    assert ps["Jev decides"]["cost_per_case_usd"] > 0
    assert ps["Jev + explainer_open explains (free_form)"]["cost_per_case_usd"] > ps["Jev decides"]["cost_per_case_usd"]
    # Explaining only adverse decisions costs between deciding alone and explaining everything.
    adverse = ps["Jev + explainer_open explains adverse decisions (free_form)"]
    assert (ps["Jev decides"]["cost_per_case_usd"] <= adverse["cost_per_case_usd"]
            <= ps["Jev + explainer_open explains (free_form)"]["cost_per_case_usd"])
    assert 0 < adverse["explained_share"] < 1
    assert both > jev_cost
    assert all(p["latency_cases"] == p["n_cases"] for p in ps.values())
    assert ps["Jev decides"]["wrong_decisions_with_grounds"] == 0
    report = render_report(s)
    assert "none (no explanation)" in report and "Cost per 1,000 cases" in report


def test_a_judge_run_over_another_pilot1_run_is_refused(runs, tmp_path):
    _, p1, j = runs
    manifest = json.loads((j.dir / "manifest.json").read_text())
    manifest["pilot1"]["run_id"] = "someone-else"
    (j.dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pipelines(p1.dir, j.dir)
