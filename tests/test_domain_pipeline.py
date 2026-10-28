"""The runners, analyses and explainer/simulator prompts on every decision domain."""

import dataclasses
import hashlib
import json

import pytest

from sysone.analysis.pilot0 import summarize as summarize_pilot0
from sysone.analysis.pilot1 import analyze_run as analyze_pilot1
from sysone.benchmark.adapter import REFUND, adapter_for, refund_adapter
from sysone.benchmark.domains import DOMAINS, DomainCaseGenerator
from sysone.benchmark.domains.base import basis_language
from sysone.benchmark.refund_policy import policy_text
from sysone.benchmark.store import freeze_benchmark, load_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider, FakeProvider
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.experiments.pilot1 import Pilot1Config, run_pilot1
from sysone.explain.basis import BasisError, execute, parse_basis
from sysone.explain.explainer import ExplainerModel
from sysone.explain.simulator import Simulator, parse_simulation
from sysone.registry import REPO_ROOT
from sysone.runs import Run
from sysone.types import ChatResponse, Usage

NAMES = sorted(DOMAINS)
ALIASES = ("explainer_open", "explainer_gemini", "simulator")


def guarded(results, provider, alias="x", provider_name="fake", run_id="r"):
    guard = CostGuard(Ledger.for_run(results, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec(alias, provider_name, f"fake-{alias}", Price(0.0, 0.0)), provider, guard,
                         cache=None, sleep=lambda s: None)


# ---- the policy as a basis -----------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_policy_basis_agrees_with_the_rules_on_every_variant(name):
    domain = DOMAINS[name]
    lang = basis_language(domain)
    basis = domain.policy_basis(domain.default_params)
    assert parse_basis(basis.to_dict(), lang) == basis
    for seed in range(3):
        for case in DomainCaseGenerator(domain, seed).generate(10):
            for v in case.variants:
                assert execute(basis, v.record, lang).decision == v.policy_outcome, v.variant_id


@pytest.mark.parametrize("name", NAMES)
def test_basis_language_rejects_other_domains_fields_and_outcomes(name):
    lang = basis_language(DOMAINS[name])
    with pytest.raises(BasisError):
        parse_basis({"rules": [{"conditions": [{"field": "has_receipt", "operator": "equals", "value": True}],
                                "outcome": DOMAINS[name].task.labels[0]}], "default_outcome": None}, lang)
    with pytest.raises(BasisError):
        parse_basis({"rules": [], "default_outcome": "maybe"}, lang)


# ---- adapters ------------------------------------------------------------------------------


def test_refund_adapter_matches_the_refund_code():
    cases, meta = load_benchmark(REPO_ROOT / "benchmarks" / "refund_v1")
    adapter = adapter_for(meta)
    assert adapter.domain == REFUND and adapter.policy == policy_text()
    assert adapter.abstain_label == "undetermined" and adapter.reference_label == "approve"
    assert all(adapter.outcome(v.record) == str(v.policy_outcome) for c in cases for v in c.variants)


@pytest.mark.parametrize("name", NAMES)
def test_domain_adapters_load_from_frozen_meta(name):
    frozen = {"moderation": "moderation_v1", "fraud": "fraud_v1", "tool_selection": "tools_v1",
              "routing": "routing_v1", "eligibility": "eligibility_v1"}[name]
    cases, meta = load_benchmark(REPO_ROOT / "benchmarks" / frozen)
    adapter = adapter_for(meta)
    assert adapter.domain == name and adapter.labels == DOMAINS[name].task.labels
    for c in cases:
        assert adapter.undetermined(c.decisive_rule) == (c.case_type == "ambiguous")
        for v in c.variants:
            assert adapter.outcome(v.record) == v.policy_outcome
            assert adapter.render(v) == DOMAINS[name].render(v.record, v.exception_in)


def test_unknown_domain_is_refused():
    with pytest.raises(ValueError):
        adapter_for({"name": "x", "domain": "astrology", "policy": {}})


# ---- prompts ---------------------------------------------------------------------------------

# sha256 of the refund prompts and schemas, pinned before other domains existed: a change
# here changes the cache keys of every refund request and breaks comparisons with Pilot 1.
REFUND_PROMPTS_SHA256 = "9eb3d74ff5e048bbcc2e1e3a61a8f2591b45471059e8017d255af3e2a5ef9a65"


def _refund_prompts(tmp_path) -> str:
    cases, _ = load_benchmark(REPO_ROOT / "benchmarks" / "refund_v1")
    case = next(c for c in cases if len(c.variants) > 2)
    a = refund_adapter()
    o, v = a.render(case.original), a.render(case.variants[-1])
    client = guarded(tmp_path, FakeProvider(), provider_name="gemini")
    e, s = ExplainerModel(client), Simulator(client)
    requests = [e.build_decision_request(a.policy, o, sample_index=i) for i in range(3)]
    requests += [e.build_explanation_request(c, a.policy, o, "deny") for c in ("free_form", "structured")]
    requests += [s.build_request(a.policy, o, "deny", x, v) for x in (None, "Because.")]
    return json.dumps([dataclasses.asdict(r) for r in requests], sort_keys=True, default=str)


def test_refund_prompts_are_unchanged(tmp_path):
    digest = hashlib.sha256(_refund_prompts(tmp_path).encode()).hexdigest()
    assert digest == REFUND_PROMPTS_SHA256


@pytest.mark.parametrize("name", NAMES)
def test_domain_prompts_use_the_domains_policy_options_and_fields(tmp_path, name):
    domain = DOMAINS[name]
    a = adapter_for({"domain": name, "policy": dataclasses.asdict(domain.default_params)})
    case = DomainCaseGenerator(domain, 0).generate(1)[0]
    client = guarded(tmp_path, FakeProvider(), provider_name="gemini")
    o = a.render(case.original)
    decision = ExplainerModel(client, setting=a.setting).build_decision_request(a.policy, o)
    prompt = decision.messages[-1].content
    assert prompt.startswith(f"## {a.setting.policy_heading}\n") and "refund" not in prompt.lower()
    assert all(f'- "{label}":' in prompt for label in a.labels)
    assert decision.json_schema["properties"]["decision"]["enum"] == list(a.labels)
    structured = ExplainerModel(client, setting=a.setting).build_explanation_request("structured", a.policy, o,
                                                                                     a.labels[0])
    text = structured.messages[-1].content
    assert domain.case_noun in text and all(f'- "{f.name}":' in text for f in domain.basis_fields(domain.default_params))
    sim = Simulator(client, setting=a.setting).build_request(a.policy, o, a.labels[0], None, o)
    assert domain.cases_noun in sim.messages[-1].content and "refund" not in sim.messages[-1].content.lower()
    assert sim.json_schema["properties"]["probabilities"]["required"] == list(a.labels)


def test_simulations_parse_with_the_domains_options():
    a = adapter_for({"domain": "routing", "policy": {}})
    reply = {"predicted_decision": "human", "probabilities": {"billing": 0.1, "technical": 0.1, "security": 0.1,
                                                              "human": 0.7},
             "basis": {"rules": [{"conditions": [{"field": "urgency_score", "operator": "more_than", "value": 79}],
                                  "outcome": "human"}], "default_outcome": "billing"}, "covered": True}
    response = ChatResponse(text=json.dumps(reply), usage=Usage(0, 0), cost_usd=0.0, provider="", model_requested="",
                            model_used="", finish_reason=None, raw={})
    sim = parse_simulation(response, "k", a.setting)
    assert sim.parse_ok and sim.basis is not None and sim.predicted == "human"
    # Refund options don't parse as routing ones.
    refund_reply = {**reply, "predicted_decision": "approve"}
    refund_response = dataclasses.replace(response, text=json.dumps(refund_reply))
    assert not parse_simulation(refund_response, "k", a.setting).parse_ok


# ---- Pilot 0 analysis ----------------------------------------------------------------------


def _row(case_id, kind, decision, policy, truth, *, rule, true_rule=None, harmful=None, sample=0):
    return {"case_id": case_id, "variant_id": f"{case_id}-{kind}", "variant_kind": kind, "sample_index": sample,
            "decision": decision, "parse_ok": True, "policy_outcome": policy, "true_outcome": truth,
            "case_type": "correct", "model_used": "m", "cost_usd": 0.0, "cached": False, "status": "ok",
            "probabilities": None, "variant_rule": rule, "decisive_rule": true_rule or rule,
            "harmful_decision": harmful, "option_order": None}


def test_abstain_option_as_a_real_outcome_is_contestable():
    """An urgent ticket belongs with a person: routing it to billing is a wrong decision,
    while a ticket with no topic on file has no determined outcome."""
    a = adapter_for({"domain": "routing", "policy": {}})
    rows = [*[_row("urgent", "original", "billing", "human", "human", rule="A4_urgent", sample=s) for s in range(3)],
            *[_row("no_topic", "original", "billing", "human", "human", rule="A3_topic_unknown", sample=s)
              for s in range(3)]]
    s = summarize_pilot0(rows, adapter=a)
    assert s["P0.3_contestable_cases"] == ["urgent"]
    assert s["P0.3_contestable_by_harm"] == {"other: billing instead of human": ["urgent"]}
    # Routing has no harmful decision, so the go/no-go counts every contestable case.
    assert s["domain"] == "routing" and s["harmed_party"] is None


def test_harmful_decisions_are_counted_per_domain():
    a = adapter_for({"domain": "fraud", "policy": {}})
    rows = [_row("c", "original", "block", "approve", "approve", rule="F6_no_risk_signal", harmful="block",
                 sample=s) for s in range(3)]
    s = summarize_pilot0(rows, adapter=a)
    assert s["P0.3_user_harmful_cases"] == ["c"]
    assert s["P0.3_contestable_by_harm"] == {"cardholder: harmed by block": ["c"]}


# ---- runners, end to end with fake models --------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_pilot0_and_pilot1_run_on_every_domain(tmp_path, name):
    domain = DOMAINS[name]
    cases = DomainCaseGenerator(domain, seed=1).generate(1)
    bench = freeze_benchmark(cases, tmp_path / "bench",
                             {"domain": name, "policy": dataclasses.asdict(domain.default_params)})
    results = tmp_path / "results"
    p0 = Run("p0", results)
    jev = guarded(results, FakeDecisionProvider(malformed_every=0), alias="jev", run_id="p0")
    assert run_pilot0(Pilot0Config(benchmark=str(bench)), p0, jev, benchmark_dir=bench) == "completed"
    decisions = p0.writer("decisions").rows()
    assert len(decisions) == 3 * sum(len(c.variants) for c in cases)
    assert {r["decision"] for r in decisions} <= set(domain.task.labels) and all(r["parse_ok"] for r in decisions)
    assert p0.manifest()["domain"] == name

    p1 = Run("p1", results)
    clients = {a: guarded(results, FakeChatProvider(malformed_every=0), alias=a, run_id="p1") for a in ALIASES}
    assert run_pilot1(Pilot1Config(), p1, clients, pilot0_dir=p0.dir) == "completed"
    explanations = p1.writer("explanations").rows()
    simulations = p1.writer("simulations").rows()
    assert explanations and all(e["parse_ok"] for e in explanations)
    assert simulations and all(s["parse_ok"] for s in simulations)
    assert {s["predicted_decision"] for s in simulations} <= set(domain.task.labels)

    summary = analyze_pilot1(p1.dir)
    # Every structured explanation runs against every hidden variant's record.
    assert all(b["decidable"] == 1.0 for b in summary["option_b_structured_executed"].values())


def test_fake_chat_model_answers_with_the_requests_options(tmp_path):
    a = adapter_for({"domain": "tool_selection", "policy": {}})
    client = guarded(tmp_path, FakeChatProvider(malformed_every=0), alias="explainer_open")
    case = DomainCaseGenerator(DOMAINS["tool_selection"], 0).generate(1)[0]
    explainer = ExplainerModel(client, setting=a.setting)
    labels = {explainer.decide(a.policy, a.render(case.original), sample_index=i,
                               tags={"step": "explainer_decision"}).label for i in range(12)}
    assert labels <= set(a.labels) and None not in labels
