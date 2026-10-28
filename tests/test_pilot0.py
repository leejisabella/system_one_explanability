import json

import pytest

from sysone.analysis.pilot0 import analyze_run, majority, summarize
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.store import freeze_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.clients.fake import FAKE_JEV_VERSION, FakeDecisionProvider, FakeProvider, choice_answer, typesafe_result
from sysone.decision.jev import (
    LABELS,
    QUESTION,
    JevDecisionModel,
    QueryBudget,
    QueryBudgetExhausted,
    option_order,
    parse_answer,
)
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.runs import Run
from sysone.types import SystemOneRequest


@pytest.mark.parametrize("probs,label", [
    ({"approve": 0.9, "deny": 0.1, "undetermined": 0.0}, "approve"),
    ({"approve": 0.33, "deny": 0.33, "undetermined": 0.33}, "approve"),  # two-decimal rounding: sum 0.99
    ({"approve": 0.0, "deny": 1.0, "undetermined": 0.0}, "deny"),
    ({"approve": 0.1, "deny": 0.2, "undetermined": 0.7}, "undetermined"),
])
def test_parse_valid(probs, label):
    d = parse_answer(choice_answer(probs))
    assert d.parse_ok and d.label == label
    assert d.probabilities == probs and d.p_approve == probs["approve"]


@pytest.mark.parametrize("answer", [
    "approve",
    {"type": "noul", "noul": 0.7},
    {"type": "choice", "choice": "approve", "confidence": 0.8},  # no probabilities
    {**choice_answer({"approve": 0.9, "deny": 0.1, "undetermined": 0.0}), "choice": "maybe"},
    choice_answer({"approve": 0.9, "deny": 0.1, "undetermined": 0.0, "refer": 0.0}),  # unexpected option
    choice_answer({"approve": 0.9, "deny": 0.1}),  # missing option
    choice_answer({"approve": 1.5, "deny": -0.5, "undetermined": 0.0}),
    choice_answer({"approve": 0.7, "deny": 0.7, "undetermined": 0.0}),  # does not sum to 1
    {**choice_answer({"approve": 0.9, "deny": 0.1, "undetermined": 0.0}), "confidence": None},
])
def test_parse_invalid(answer):
    d = parse_answer(answer)
    assert not d.parse_ok and d.label is None and d.probabilities is None and d.parse_error


def guarded(tmp_path, provider, run_id="r"):
    guard = CostGuard(Ledger.for_run(tmp_path, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(1.0, 0.0)), provider, guard, sleep=lambda s: None)


def test_query_budget_limits_decisions(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeDecisionProvider(malformed_every=0)))
    budget = QueryBudget(2)
    jev.decide("policy", "case", budget=budget)
    jev.decide("policy", "case", sample_index=1, budget=budget)
    with pytest.raises(QueryBudgetExhausted):
        jev.decide("policy", "case", sample_index=2, budget=budget)


def test_decision_request_is_one_typed_choice_question(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeProvider()), rotate_options=False)
    req = jev.build_request("POLICY TEXT", "CASE TEXT", sample_index=2)
    assert isinstance(req, SystemOneRequest) and req.sample_index == 2
    assert req.state == {"refund_policy": "POLICY TEXT", "case_file": "CASE TEXT"}
    question = req.questions[QUESTION]
    assert question["type"] == "choice" and list(question["criteria"]) == ["approve", "deny", "undetermined"]
    assert "cannot be determined" in question["criteria"]["undetermined"]


def test_option_order_rotates_so_each_option_is_listed_first_once():
    orders = [option_order(i) for i in range(3)]
    assert [o[0] for o in orders] == ["approve", "deny", "undetermined"]
    assert all(sorted(o) == sorted(LABELS) for o in orders)
    assert option_order(3) == orders[0]
    assert option_order(1, rotate=False) == LABELS


def test_request_lists_options_in_the_rotated_order(tmp_path):
    jev = JevDecisionModel(guarded(tmp_path, FakeDecisionProvider(malformed_every=0)))
    for i in range(3):
        result = jev.decide("policy", "case", sample_index=i)
        assert result.option_order == option_order(i)


@pytest.fixture
def small_benchmark(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=2)
    return freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})


def n_variants(bench):
    return json.loads((bench / "meta.json").read_text())["n_variants"]


def test_dry_run_writes_results_and_resumes(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=2)
    provider = FakeDecisionProvider()
    run = Run("dry", tmp_path / "results")
    status = run_pilot0(config, run, guarded(tmp_path / "results", provider, "dry"), benchmark_dir=small_benchmark)
    rows = run.writer("decisions").rows()
    assert status == "completed" and len(rows) == n_variants(small_benchmark) * 2
    manifest = run.manifest()
    assert manifest["benchmark"]["sha256"] and manifest["config"]["samples_per_variant"] == 2
    assert "budget.yaml" in manifest["config_files"]
    assert manifest["model_version"] == FAKE_JEV_VERSION

    # Every row records the version and, when parsed, the full probabilities.
    assert {r["model_used"] for r in rows} == {FAKE_JEV_VERSION}
    for r in rows:
        if r["parse_ok"]:
            assert set(r["probabilities"]) == set(LABELS) and r["p_approve"] == r["probabilities"]["approve"]
        assert r["option_order"] == list(option_order(r["sample_index"]))
        assert r["decisive_rule"] and r["variant_rule"]

    # A second run with the same ID resumes and makes no new calls.
    calls = provider.calls
    run_pilot0(config, run, guarded(tmp_path / "results", provider, "dry"), benchmark_dir=small_benchmark)
    assert provider.calls == calls and len(run.writer("decisions").rows()) == len(rows)

    summary = analyze_run(run.dir)
    assert summary["n_rows"] == len(rows)
    assert summary["P0.7_model_versions"]["n_versions"] == 1
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
    answer = {QUESTION: choice_answer({"approve": 0.5, "deny": 0.5, "undetermined": 0.0})}
    provider = FakeProvider([typesafe_result(answer, reported_cost_usd=3.0)])
    run = Run("halt", tmp_path / "results")
    status = run_pilot0(config, run, guarded(tmp_path / "results", provider, "halt"), benchmark_dir=small_benchmark)
    assert status.startswith("halted") and run.manifest()["status"] == status
    assert len(provider.calls) == 1


def test_version_change_halts_the_run(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1)
    provider = FakeDecisionProvider(version_change_after=2)
    run = Run("moved", tmp_path / "results")
    client = guarded(tmp_path / "results", provider, "moved")
    status = run_pilot0(config, run, client, benchmark_dir=small_benchmark)

    assert status.startswith("halted") and "model version changed" in status
    assert provider.calls == 3
    assert run.manifest()["model_version"] == FAKE_JEV_VERSION
    rows = run.writer("decisions").rows()
    assert [r["status"] for r in rows] == ["ok", "ok", "error"]
    assert rows[-1]["model_used"] == f"{FAKE_JEV_VERSION}-next"
    # The mismatched call was still billed and logged.
    assert len(client.guard.ledger.entries()) == 3


def test_resume_stays_on_the_locked_version(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1)
    run = Run("resume", tmp_path / "results")
    run_pilot0(config, run, guarded(tmp_path / "results", FakeDecisionProvider(version_change_after=2), "resume"),
               benchmark_dir=small_benchmark)

    # The alias has moved: resuming halts on the first answer from the new version.
    moved = FakeDecisionProvider(version=f"{FAKE_JEV_VERSION}-next")
    status = run_pilot0(config, run, guarded(tmp_path / "results", moved, "resume"), benchmark_dir=small_benchmark)
    assert status.startswith("halted") and moved.calls == 1

    # Back on the locked version, the run finishes.
    status = run_pilot0(config, run, guarded(tmp_path / "results", FakeDecisionProvider(), "resume"),
                        benchmark_dir=small_benchmark)
    assert status == "completed"
    ok = [r for r in run.writer("decisions").rows() if r["status"] == "ok"]
    assert len(ok) == n_variants(small_benchmark) and {r["model_used"] for r in ok} == {FAKE_JEV_VERSION}


def test_expected_version_is_enforced_from_the_first_call(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1, expected_model_version="jev-1.14.0")
    provider = FakeDecisionProvider()
    run = Run("pinned", tmp_path / "results")
    status = run_pilot0(config, run, guarded(tmp_path / "results", provider, "pinned"), benchmark_dir=small_benchmark)
    assert status.startswith("halted") and provider.calls == 1
    assert run.manifest()["model_version"] == "jev-1.14.0"


def test_expected_version_cannot_differ_from_a_locked_run(tmp_path, small_benchmark):
    run = Run("conflict", tmp_path / "results")
    run.update_manifest(model_version="jev-1.13.0")
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=1, expected_model_version="jev-1.14.0")
    provider = FakeDecisionProvider()
    with pytest.raises(ValueError, match="locked to jev-1.13.0"):
        run_pilot0(config, run, guarded(tmp_path / "results", provider, "conflict"), benchmark_dir=small_benchmark)
    assert provider.calls == 0


def row(case_id, kind, sample, decision, policy, truth, case_type="correct", model="m", vid=None, p=None,
        rule="R4_opened", true_rule=None, order=None):
    return {
        "case_id": case_id, "variant_id": vid or f"{case_id}-{kind}", "variant_kind": kind,
        "sample_index": sample, "decision": decision, "parse_ok": decision is not None,
        "policy_outcome": policy, "true_outcome": truth, "case_type": case_type,
        "model_used": model, "cost_usd": 0.01, "cached": False, "status": "ok", "p_approve": p,
        "variant_rule": rule, "decisive_rule": true_rule or rule, "option_order": order,
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
    assert s["go_no_go"]["stability_ok"]
    assert s["P0.8_cost"]["total_usd"] == pytest.approx(0.12)
    # Rows without probabilities (as from older runs) leave the probability metrics empty.
    assert s["P0.4_probability_noise"]["n_variants"] == 0


def test_probability_noise_and_shifts():
    rows = [
        *[row("a", "original", s, "approve", "approve", "approve", p=p) for s, p in enumerate([0.80, 0.82, 0.80])],
        *[row("a", "cue_edit", s, "approve", "approve", "approve", p=p) for s, p in enumerate([0.70, 0.70, 0.70])],
        *[row("a", "decisive_edit", s, "deny", "deny", "approve", p=p) for s, p in enumerate([0.10, 0.20, 0.15])],
    ]
    s = summarize(rows)
    noise = s["P0.4_probability_noise"]
    assert noise["n_variants"] == 3
    assert noise["identical_across_repeats_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert noise["max_range_p_approve"] == pytest.approx(0.10)
    shift = {f["variant_kind"]: f["mean_abs_shift_p_approve"] for f in s["P0.5_6_flips"]}
    assert shift["cue_edit"] == pytest.approx(0.1067, abs=1e-4)
    assert shift["decisive_edit"] == pytest.approx(0.6567, abs=1e-4)


def test_errors_are_broken_down_by_rule_and_cause():
    W, O = "R5_within_window", "R4_opened"
    rows = [
        # a: Jev follows a corrupted record (presented deny, truth approve): a planted defect.
        *[row("a", "original", s, "deny", "deny", "approve", "incorrect_input", rule=O, true_rule=W) for s in range(3)],
        # b: the window decides, and Jev gets it wrong: its own date-window error.
        *[row("b", "original", s, "deny", "approve", "approve", rule=W) for s in range(3)],
        # c: a non-date rule decides, and Jev gets it wrong.
        *[row("c", "original", s, "approve", "deny", "deny", rule=O) for s in range(3)],
        # d: ambiguous case, and Jev correctly says it can't be determined.
        *[row("d", "original", s, "undetermined", "undetermined", "undetermined", "ambiguous",
              rule="R3_receipt_unknown") for s in range(3)],
    ]
    s = summarize(rows)
    assert sorted(s["P0.3_contestable_cases"]) == ["a", "b", "c"]
    assert s["P0.3_contestable_by_cause"] == {
        "follows presented record": 1,
        "misapplies policy: date window": 1,
        "misapplies policy: other rules": 1,
    }
    assert s["P0.3_truth_agreement_by_type"]["ambiguous"]["rate"] == 1.0
    groups = s["P0.2_policy_agreement_by_rule_group"]
    assert groups["date window"] == {"n": 3, "k": 0, "rate": 0.0}
    assert groups["other rules"]["k"] == 6 and groups["other rules"]["n"] == 9
    assert s["P0.3_truth_agreement_by_rule"][W] == {"n": 2, "k": 0, "rate": 0.0}


def test_option_order_effect():
    orders = [list(option_order(i)) for i in range(3)]
    # Jev picks whatever is listed first, and P(approve) is 0.2 higher when approve is first.
    rows = [
        row("a", "original", i, orders[i][0], "approve", "approve", p=0.6 if i == 0 else 0.4, order=orders[i])
        for i in range(3)
    ]
    s = summarize(rows)
    oo = s["P0.4_option_order"]
    assert oo["orders_vary_across_repeats"] and oo["first_listed_chosen_share"] == 1.0
    assert oo["no_effect_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert oo["mean_p_approve_shift_when_approve_first"] == pytest.approx(0.2)
    assert s["P0.4_probability_noise"]["includes_order_effects"]


def test_noise_run_without_rotation_keeps_repeats_identical(tmp_path, small_benchmark):
    config = Pilot0Config(benchmark=str(small_benchmark), samples_per_variant=2, rotate_option_order=False)
    run = Run("noise", tmp_path / "results")
    run_pilot0(config, run, guarded(tmp_path / "results", FakeDecisionProvider(), "noise"), benchmark_dir=small_benchmark)
    assert {tuple(r["option_order"]) for r in run.writer("decisions").rows()} == {LABELS}
    assert not analyze_run(run.dir)["P0.4_probability_noise"]["includes_order_effects"]


def test_majority_ties():
    assert majority(["approve", "deny"]) is None
    assert majority(["approve", "deny", "deny"]) == "deny"
    assert majority([]) is None


def test_wrong_decisions_are_split_by_who_they_harm():
    rows = [
        # a: owed a refund, Jev denies (harms the customer).
        *[row("a", "original", s, "deny", "deny", "approve", "incorrect_input") for s in range(3)],
        # b: not owed one, Jev approves (harms the store).
        *[row("b", "original", s, "approve", "deny", "deny", rule="R5_outside_window") for s in range(3)],
        # c: owed a refund, Jev says it can't be determined.
        *[row("c", "original", s, "undetermined", "approve", "approve") for s in range(3)],
    ]
    s = summarize(rows)
    assert s["P0.3_contestable_by_harm"] == {
        "customer: wrongly denied": ["a"],
        "neither: answered undetermined": ["c"],
        "store: wrongly approved": ["b"],
    }
    assert s["P0.3_user_harmful_cases"] == ["a"]
    assert not s["go_no_go"]["enough_contestable_cases"]
