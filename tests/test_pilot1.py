import json

import pytest

from sysone.analysis.pilot1 import analyze_run, summarize
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.records import Condition
from sysone.benchmark.refund_policy import evaluate
from sysone.benchmark.render import render_case_text
from sysone.benchmark.store import freeze_benchmark, load_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider, ok_result
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.experiments.pilot1 import Pilot1Config, check_families, model_family, run_pilot1
from sysone.explain.basis import BasisError, execute, parse_basis, policy_basis
from sysone.explain.explainer import DECISION_SYSTEM, ExplainerModel
from sysone.explain.jsonout import parse_json_object
from sysone.explain.simulator import Simulator, parse_simulation
from sysone.registry import REPO_ROOT
from sysone.runs import Run
from sysone.types import ChatResponse, Usage

ALIASES = ("explainer_open", "explainer_gemini", "simulator")


# --- Basis ---------------------------------------------------------------------------


def test_policy_basis_agrees_with_the_policy_on_every_benchmark_record():
    cases, _ = load_benchmark(REPO_ROOT / "benchmarks" / "refund_v0")
    records = [r for c in cases for r in (c.true_record, c.presented_record, *(v.record for v in c.variants))]
    basis = policy_basis()
    assert all(execute(basis, r).decision == str(evaluate(r).outcome) for r in records)


def cond(field, op, value=None):
    return {"field": field, "operator": op, "value": value}


@pytest.mark.parametrize("basis", [
    None,
    {"rules": "x", "default_outcome": None},
    {"rules": [{"conditions": [cond("customer_name", "equals", "x")], "outcome": "deny"}], "default_outcome": None},
    {"rules": [{"conditions": [cond("has_receipt", "more_than", 1)], "outcome": "deny"}], "default_outcome": None},
    {"rules": [{"conditions": [cond("has_receipt", "equals", "yes")], "outcome": "deny"}], "default_outcome": None},
    {"rules": [{"conditions": [cond("price_usd", "at_most", True)], "outcome": "deny"}], "default_outcome": None},
    {"rules": [{"conditions": [cond("condition", "equals", "broken")], "outcome": "deny"}], "default_outcome": None},
    {"rules": [{"conditions": [], "outcome": "refer"}], "default_outcome": None},
    {"rules": [], "default_outcome": "maybe"},
])
def test_invalid_bases_are_rejected(basis):
    with pytest.raises(BasisError):
        parse_basis(basis)


def test_basis_execution_order_missing_facts_and_defaults():
    record = RefundCaseGenerator(seed=0).generate(per_type=1)[0].true_record.replace(
        has_receipt=None, condition=Condition.UNOPENED, final_sale=False, manager_preapproval=False)
    first_wins = parse_basis({"rules": [
        {"conditions": [cond("condition", "equals", "unopened")], "outcome": "approve"},
        {"conditions": [], "outcome": "deny"},
    ], "default_outcome": None})
    assert execute(first_wins, record).decision == "approve" and execute(first_wins, record).rule_index == 0

    needs_receipt = parse_basis({"rules": [{"conditions": [cond("has_receipt", "equals", False)], "outcome": "deny"}],
                                 "default_outcome": "approve"})
    result = execute(needs_receipt, record)
    assert result.decision is None and "has_receipt" in result.note

    checks_missing = parse_basis({"rules": [{"conditions": [cond("has_receipt", "is_missing")],
                                             "outcome": "undetermined"}], "default_outcome": None})
    assert execute(checks_missing, record).decision == "undetermined"
    assert execute(checks_missing, record.replace(has_receipt=True)).decision is None  # no rule, no default


def test_json_is_found_in_fenced_or_chatty_replies():
    assert parse_json_object('```json\n{"decision": "deny"}\n```') == {"decision": "deny"}
    assert parse_json_object('Sure. {"a": 1} Hope that helps.') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None


# --- Model wrappers ------------------------------------------------------------------


def guarded(tmp_path, provider, alias="m", provider_name="fake", model=None, run_id="r"):
    guard = CostGuard(Ledger.for_run(tmp_path, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    spec = ModelSpec(alias, provider_name, model or f"fake-{alias}", Price(0.0, 0.0))
    return GuardedClient(spec, provider, guard, sleep=lambda s: None)


def response(text):
    return ChatResponse(text, Usage(1, 1), 0.0, "fake", "m", "m", "stop", {})


def sim_reply(**overrides):
    reply = {"predicted_decision": "deny", "probabilities": {"approve": 0.1, "deny": 0.85, "undetermined": 0.05},
             "basis": policy_basis().to_dict(), "covered": True}
    return json.dumps({**reply, **overrides})


def test_simulation_parsing():
    sim = parse_simulation(response(sim_reply()), "k")
    assert sim.parse_ok and sim.predicted == "deny" and sim.covered and sim.basis is not None

    # A basis that can't be run keeps the prediction but is marked.
    sim = parse_simulation(response(sim_reply(basis={"rules": "no"})), "k")
    assert sim.parse_ok and sim.basis is None and sim.basis_error and sim.basis_raw == {"rules": "no"}

    for bad in (sim_reply(predicted_decision="maybe"), sim_reply(covered="yes"),
                sim_reply(probabilities={"approve": 0.9, "deny": 0.9, "undetermined": 0.0}),
                sim_reply(probabilities={"approve": 1.0}), "not json"):
        sim = parse_simulation(response(bad), "k")
        assert not sim.parse_ok and sim.predicted is None and sim.parse_error


def test_control_differs_from_the_explained_prompt_only_by_the_explanation(tmp_path):
    sim = Simulator(guarded(tmp_path, FakeChatProvider()))
    with_expl = sim.build_request("POLICY", "ORIGINAL", "deny", "THE EXPLANATION", "VARIANT").messages[0].content
    control = sim.build_request("POLICY", "ORIGINAL", "deny", None, "VARIANT").messages[0].content
    assert "THE EXPLANATION" in with_expl and "Explanation of the system's decision" not in control
    for part in ("POLICY", "ORIGINAL", "deny", "VARIANT"):
        assert part in control
    stripped = with_expl.replace("\n\n## Explanation of the system's decision\nTHE EXPLANATION", "")
    assert stripped.replace(", an explanation of that decision", "") == control


def test_providers_without_structured_output_get_the_format_in_the_prompt(tmp_path):
    reply = ok_result('Here you go:\n```json\n{"decision": "approve"}\n```')
    provider = type("Scripted", (FakeChatProvider,), {"send": lambda self, m, r, p: reply})()
    bedrock_like = ExplainerModel(guarded(tmp_path, provider, provider_name="bedrock", model="us.anthropic.x"))
    request = bedrock_like.build_decision_request("POLICY", "CASE")
    assert request.json_schema is None and '{"decision"' in request.messages[0].content
    assert bedrock_like.decide("POLICY", "CASE").label == "approve"
    assert ExplainerModel(guarded(tmp_path, provider)).build_decision_request("P", "C").json_schema is not None


def test_explainer_decisions_carry_the_answer_only_system_message(tmp_path):
    # Without it Llama 4 Maverick reasons past the 64-token cap (pilot1-v1-2026-10-28).
    for name, model in (("bedrock", "us.meta.llama-x"), ("fake", "fake-model")):
        explainer = ExplainerModel(guarded(tmp_path, FakeChatProvider(), provider_name=name, model=model))
        assert explainer.build_decision_request("P", "C").system == DECISION_SYSTEM


def test_family_check():
    def spec(alias, provider, model):
        return ModelSpec(alias, provider, model, Price(1, 1))

    assert model_family(spec("e", "bedrock", "us.anthropic.claude-opus-4-5-20251101-v1:0")) == "anthropic"
    assert model_family(spec("e", "bedrock", "anthropic.claude-x")) == "anthropic"
    assert model_family(spec("e", "gemini", "gemini-3.8-flash")) == "google"
    assert model_family(spec("s", "openrouter", "openai/gpt-6.1-terra")) == "openai"

    class C:
        def __init__(self, s):
            self.spec = s

    config = Pilot1Config(explainers=("explainer_open", "explainer_gemini"))
    explainers = {"explainer_open": C(spec("explainer_open", "bedrock", "us.meta.llama-x")),
                  "explainer_gemini": C(spec("explainer_gemini", "gemini", "gemini-3.8-flash"))}
    check_families(config, {**explainers, "simulator": C(spec("simulator", "openrouter", "openai/gpt-6.1-terra"))})
    for bad in (spec("simulator", "openrouter", "google/gemini-3.8-flash"),
                spec("simulator", "bedrock", "meta.llama3-x"),
                spec("simulator", "typesafe", "jev-latest")):
        with pytest.raises(ValueError):
            check_families(config, {**explainers, "simulator": C(bad)})


# --- Runner --------------------------------------------------------------------------


class Recording(FakeChatProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def send(self, model, request, price):
        self.requests.append(request)
        return super().send(model, request, price)


@pytest.fixture
def pilot0_run(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=1)
    bench = freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})
    run = Run("p0", tmp_path / "results")
    guard = CostGuard(Ledger.for_run(tmp_path / "results", "p0"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    client = GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(0.0, 0.0)),
                           FakeDecisionProvider(malformed_every=0), guard)
    run_pilot0(Pilot0Config(benchmark=str(bench), samples_per_variant=3), run, client, benchmark_dir=bench)
    return run.dir


def fake_clients(tmp_path, providers, run_id="p1"):
    return {a: guarded(tmp_path / "results", providers[a], alias=a, run_id=run_id) for a in ALIASES}


def test_dry_run_writes_all_steps_and_resumes_without_new_calls(tmp_path, pilot0_run):
    providers = {a: Recording(malformed_every=0) for a in ALIASES}
    run = Run("p1", tmp_path / "results")
    status = run_pilot1(Pilot1Config(), run, fake_clients(tmp_path, providers), pilot0_dir=pilot0_run)
    assert status == "completed"

    cases, _ = load_benchmark(json.loads((pilot0_run / "manifest.json").read_text())["benchmark"]["path"])
    n_variants = sum(len(c.variants) for c in cases)
    skipped = run.manifest()["skipped_cases_jev_tie"]
    explained = [c for c in cases if c.case_id not in skipped]
    n_hidden = sum(len(c.variants) - 1 for c in explained)

    decisions = run.writer("explainer_decisions").rows()
    explanations = run.writer("explanations").rows()
    simulations = run.writer("simulations").rows()
    assert len(decisions) == 2 * n_variants * 3
    assert len(explanations) == len(explained) * (2 * 2 + 1)
    # The control, plus one simulation per explanation, for every hidden variant.
    assert len(simulations) == n_hidden * (1 + 2 * 2 + 1)
    assert {r["condition"] for r in simulations} == {"control", "free_form", "structured", "decision_probabilities"}
    assert run.manifest()["model_versions"] == {a: "fake-chat-1.0" for a in ALIASES}

    # Explanations see the original case only, never a hidden variant.
    texts = {v.variant_id: render_case_text(v) for c in cases for v in c.variants}
    by_case = {c.case_id: c for c in cases}
    for alias in ("explainer_open", "explainer_gemini"):
        for req in providers[alias].requests:
            if req.tags["step"] != "explanation":
                continue
            case = by_case[req.tags["case_id"]]
            prompt = req.messages[0].content
            # Compared exactly: an exception-removed variant's text is a prefix of the original's.
            case_file = prompt.split("## Case file\n")[1].split("\n\n## System decision")[0]
            assert case_file == texts[case.original.variant_id]
            assert all(texts[v.variant_id] != case_file for v in case.variants[1:])
    # The simulator never gets Jev's decision on the variant: only the original decision appears.
    for req in providers["simulator"].requests:
        assert "System decision on the original case file" in req.messages[0].content

    calls = {a: p.calls for a, p in providers.items()}
    run_pilot1(Pilot1Config(), run, fake_clients(tmp_path, providers), pilot0_dir=pilot0_run)
    assert {a: p.calls for a, p in providers.items()} == calls
    assert len(run.writer("explanations").rows()) == len(explanations)

    summary = analyze_run(run.dir)
    assert set(summary["faithfulness"]) == {f"{a}/{c}" for a in ("explainer_open", "explainer_gemini")
                                            for c in ("free_form", "structured")} | {"none/decision_probabilities"}
    assert (run.dir / "report.md").exists()


def test_a_frozen_explanation_is_never_rewritten(tmp_path, pilot0_run):
    run = Run("p1", tmp_path / "results")
    config = Pilot1Config(conditions=("free_form",), explainer_decision_samples=0)
    run_pilot1(config, run, fake_clients(tmp_path, {a: FakeChatProvider() for a in ALIASES}), pilot0_dir=pilot0_run)
    first = {r["explanation_id"]: r["explanation_text"] for r in run.writer("explanations").rows()}

    # A different explainer on resume must not replace what was saved.
    other = {a: FakeChatProvider(version="fake-chat-1.0") for a in ALIASES}
    other["explainer_open"] = Recording()
    run_pilot1(config, run, fake_clients(tmp_path, other), pilot0_dir=pilot0_run)
    assert not [r for r in other["explainer_open"].requests if r.tags["step"] == "explanation"]
    assert {r["explanation_id"]: r["explanation_text"] for r in run.writer("explanations").rows()} == first


def test_version_change_halts(tmp_path, pilot0_run):
    run = Run("p1", tmp_path / "results")
    config = Pilot1Config(explainer_decision_samples=1)
    run_pilot1(config, run, fake_clients(tmp_path, {a: FakeChatProvider() for a in ALIASES}), pilot0_dir=pilot0_run)
    # Delete the simulations so the resume has work, then resume with a moved simulator.
    (run.dir / "simulations.jsonl").unlink()
    providers = {a: FakeChatProvider() for a in ALIASES}
    providers["simulator"] = FakeChatProvider(version="fake-chat-2.0")
    status = run_pilot1(config, run, fake_clients(tmp_path, providers), pilot0_dir=pilot0_run)
    assert status.startswith("halted") and "model version changed" in status
    assert providers["simulator"].calls == 1


def test_incomplete_pilot0_is_refused(tmp_path, pilot0_run):
    manifest = json.loads((pilot0_run / "manifest.json").read_text())
    (pilot0_run / "manifest.json").write_text(json.dumps({**manifest, "status": "interrupted"}))
    providers = {a: FakeChatProvider() for a in ALIASES}
    with pytest.raises(ValueError, match="not a completed Pilot 0 run"):
        run_pilot1(Pilot1Config(), Run("p1", tmp_path / "results"), fake_clients(tmp_path, providers),
                   pilot0_dir=pilot0_run)
    assert all(p.calls == 0 for p in providers.values())


# --- Analysis on known rows ----------------------------------------------------------


def test_scores_on_known_rows():
    base = RefundCaseGenerator(seed=0).generate(per_type=1)[0].true_record.replace(
        manager_preapproval=False, final_sale=False, has_receipt=True, condition=Condition.UNOPENED)
    records = {"v1": base.replace(has_receipt=False), "v2": base, "v3": base.replace(has_receipt=None)}
    receipt_basis = {"rules": [{"conditions": [cond("has_receipt", "equals", False)], "outcome": "deny"}],
                     "default_outcome": "approve"}
    jev = {"v1": "deny", "v2": "approve", "v3": "deny"}

    def sim(eid, condition, vid, predicted, covered=True, basis=receipt_basis, explainer="e"):
        return {"status": "ok", "explanation_id": eid, "explainer": explainer, "condition": condition,
                "case_id": "c", "variant_id": vid, "variant_kind": "decisive_edit", "jev_decision": jev[vid],
                "parse_ok": True, "predicted_decision": predicted, "covered": covered, "basis": basis,
                "basis_ok": True, "probabilities": {"approve": 0.5, "deny": 0.5, "undetermined": 0.0},
                "model_used": "m"}

    simulations = [
        # Control: right on v2 only.
        sim("control/c", "control", "v1", "approve", explainer="control"),
        sim("control/c", "control", "v2", "approve", explainer="control"),
        sim("control/c", "control", "v3", "approve", explainer="control"),
        # With the explanation: right on v1 and v2.
        sim("e/free_form/c", "free_form", "v1", "deny"),
        sim("e/free_form/c", "free_form", "v2", "approve"),
        sim("e/free_form/c", "free_form", "v3", "approve", covered=False),
    ]
    explanations = [{"status": "ok", "explanation_id": "e/structured/c", "explainer": "e", "condition": "structured",
                     "case_id": "c", "parse_ok": True, "basis": receipt_basis, "model_used": "m"}]
    decisions = [
        {"explainer": "e", "variant_id": vid, "parse_ok": True, "decision": d, "model_used": "m"}
        for vid, ds in {"v0": ["deny"] * 3, "v1": ["deny", "deny", "approve"], "v2": ["deny"] * 3}.items()
        for d in ds
    ]
    jev["v0"] = "approve"
    s = summarize(decisions, explanations, simulations, records={**records, "v0": base},
                  hidden={"c": ["v1", "v2", "v3"]},
                  kinds={"v0": "original", "v1": "decisive_edit", "v2": "decisive_edit", "v3": "decisive_edit"},
                  policy={"v0": "approve", "v1": "deny", "v2": "approve", "v3": "undetermined"}, jev=jev)

    f = s["faithfulness"]["e/free_form"]
    assert f["a_direct_agreement_with_jev"] == pytest.approx(2 / 3, abs=1e-4)
    assert f["control_agreement_with_jev"] == pytest.approx(1 / 3, abs=1e-4)
    assert f["normalized_simulatability_gain"] == pytest.approx(0.5)
    assert f["covered"]["k"] == 2 and f["a_direct_agreement_when_covered"] == 1.0
    # The basis can't decide v3 (receipt missing); on v1 and v2 it matches the prediction.
    assert f["basis_decidable"]["k"] == 2 and f["a_direct_vs_a_executed_agreement"] == 1.0

    b = s["option_b_structured_executed"]["e"]
    assert b["n_variants"] == 3 and b["decidable"] == pytest.approx(2 / 3, abs=1e-4)
    assert b["executed_agreement_with_jev"] == 1.0

    d = s["jev_explainer_disagreement"]["e"]["disagreement_with_jev"]
    assert d["originals"] == {"n": 1, "k": 1, "rate": 1.0}
    assert d["variants"] == {"n": 2, "k": 1, "rate": 0.5}
