"""The LLM judge, its analysis and human agreement; and the "whose answer" (explains itself) measures."""

import csv
import json

import pytest

from sysone.analysis.judge import cohens_kappa, contest_hit, export_labeling, load_labels, summarize
from sysone.analysis.pilot1 import identification, standing_apart, whose_answer
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.store import freeze_benchmark, load_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider, FakeProvider
from sysone.experiments.judge import JudgeConfig, error_kind, run_judge
from sysone.experiments.pilot0 import Pilot0Config, run_pilot0
from sysone.experiments.pilot1 import Pilot1Config, run_pilot1
from sysone.explain.judge import JUDGMENT_SCHEMA, Judge, evaluator_facts, parse_judgment, quote_found
from sysone.registry import REPO_ROOT
from sysone.runs import Run
from sysone.types import ChatResponse, Usage

ALIASES = ("explainer_open", "explainer_gemini", "simulator")


def guarded(results, provider, alias="x", provider_name="fake", run_id="r"):
    guard = CostGuard(Ledger.for_run(results, run_id), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec(alias, provider_name, f"fake-{alias}", Price(0.0, 0.0)), provider, guard,
                         cache=None, sleep=lambda s: None)


def response(text):
    return ChatResponse(text=text, usage=Usage(0, 0), cost_usd=0.0, provider="", model_requested="",
                        model_used="", finish_reason=None, raw={})


GOOD = {"justification": "false_premise", "names_defect_field": False, "questions_defect_value": False,
        "cites_correcting_evidence": None, "recommends_review": False, "quote": "within 30 days"}


# ---- the judge ----------------------------------------------------------------------------


def test_judgments_parse_and_bad_ones_are_rejected():
    j = parse_judgment(response(json.dumps(GOOD)), "k")
    assert j.parse_ok and j.justification == "false_premise" and j.cites_correcting_evidence is None
    for bad in ({**GOOD, "justification": "rationalizes"}, {**GOOD, "names_defect_field": "yes"},
                {**GOOD, "recommends_review": None}, {**GOOD, "quote": 3}):
        assert not parse_judgment(response(json.dumps(bad)), "k").parse_ok
    assert not parse_judgment(response("I think it is a false premise."), "k").parse_ok


def test_evaluator_facts_state_the_truth_and_the_defect():
    cases, _ = load_benchmark(REPO_ROOT / "benchmarks" / "refund_v1")
    corrupted = next(c for c in cases if c.case_type == "incorrect_input")
    facts = evaluator_facts(corrupted, "deny", str(corrupted.original.policy_outcome))
    d = corrupted.defects[0]
    assert "wrong under ground truth" in facts and f'"{d.field}" is wrong' in facts
    assert all(e.source in facts for e in corrupted.evidence if e.field == d.field)
    clean = next(c for c in cases if c.case_type == "correct")
    facts = evaluator_facts(clean, str(clean.true_outcome), str(clean.original.policy_outcome))
    assert "correct under ground truth" in facts and "defects in the case file: none" in facts


def test_judge_request_shows_the_facts_and_asks_for_the_schema(tmp_path):
    judge = Judge(guarded(tmp_path, FakeProvider(), provider_name="gemini"))
    req = judge.build_request("POLICY", "CASE", "deny", "EXPLANATION", "- FACTS")
    prompt = req.messages[-1].content
    order = [prompt.index(h) for h in ("## Refund policy", "## Case file", "## Decision",
                                        "## Explanation to grade", "## Evaluator facts", "## Task")]
    assert order == sorted(order) and req.json_schema == JUDGMENT_SCHEMA
    bedrock = Judge(guarded(tmp_path, FakeProvider(), provider_name="bedrock"))
    assert bedrock.build_request("P", "C", "deny", "E", "F").json_schema is None


def test_quotes_are_checked_against_the_explanation():
    text = "The request was made\nwithin 30 days of purchase."
    assert quote_found("within 30 days", text) and quote_found('"The request was made within 30 days."', text)
    assert not quote_found("within 45 days", text) and not quote_found("", text)


def test_error_kinds():
    assert error_kind("deny", "deny", "approve") == "follows presented record"
    assert error_kind("approve", "deny", "deny") == "misapplies policy"
    assert error_kind("deny", "deny", "deny") == "jev correct" and error_kind(None, "deny", "deny") == "jev tied"


# ---- analysis and human agreement ------------------------------------------------------


def jrow(case_id, kind, justification, *, defect=True, named=False, questioned=False, cited=False, review=False,
         explainer="explainer_open", condition="free_form"):
    return {"explanation_id": f"{explainer}/{condition}/{case_id}", "explainer": explainer, "condition": condition,
            "case_id": case_id, "error_kind": kind, "defect_fields": ["condition"] if defect else [],
            "justification": justification, "names_defect_field": named if defect else None,
            "questions_defect_value": questioned if defect else None,
            "cites_correcting_evidence": cited if defect else None, "recommends_review": review,
            "quote": "q", "quote_found": True, "parse_ok": True, "status": "ok", "model_used": "m"}


def test_judge_summary_on_known_rows():
    rows = [
        jrow("a", "follows presented record", "flags_conflict", named=True, cited=True),
        jrow("b", "follows presented record", "false_premise", named=True),
        jrow("c", "misapplies policy", "invented_fact", defect=False),
        jrow("d", "misapplies policy", "flags_conflict", defect=False),
        jrow("e", "jev correct", "accurate", defect=False),
        jrow("f", "jev correct", "flags_conflict", defect=False),
    ]
    g = summarize(rows)["judge"]["explainer_open/free_form"]
    assert g["wrong_decisions_defended"]["k"] == 2 and g["wrong_decisions_defended"]["n"] == 4
    assert g["wrong_decisions_flagged"]["k"] == 2
    # Named alone isn't contestable: it has to question the value or point to the evidence.
    assert g["contestability_on_defects"] == {"n": 2, "k": 1, "rate": 0.5}
    assert g["flags_own_policy_errors"] == {"n": 2, "k": 1, "rate": 0.5}
    assert g["false_alarms_on_correct_decisions"] == {"n": 2, "k": 1, "rate": 0.5}
    assert contest_hit(rows[0]) and not contest_hit(rows[1])


def test_kappa():
    assert cohens_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohens_kappa([("a", "b"), ("b", "a")]) == -1.0
    assert cohens_kappa([]) is None


def test_human_labels_are_scored_including_pilot1_hand_coding(tmp_path):
    hand = REPO_ROOT / "research_analyses" / "E1_pilot1" / "data" / "justification_coding.csv"
    labels = load_labels([hand])
    assert len(labels) == 12 and all("justification" in v for v in labels.values())
    rows = [jrow(c, "misapplies policy", lab["justification"], defect=False, explainer=e, condition=cond)
            for (c, e, cond), lab in labels.items()]
    rows[0] = {**rows[0], "justification": "accurate"}
    ha = summarize(rows, labels)["human_agreement"]
    assert ha["n_matched"] == 12 and ha["justification"]["n"] == 12
    assert ha["justification"]["agreement"] == pytest.approx(11 / 12, abs=1e-4)

    sheet = tmp_path / "extra.csv"
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "explainer", "condition", "justification", "names_defect_field"])
        w.writerow([*next(iter(labels)), "", "no"])
    merged = load_labels([hand, sheet])
    key = next(iter(labels))
    assert merged[key]["names_defect_field"] is False and merged[key]["justification"] == labels[key]["justification"]


# ---- runners ------------------------------------------------------------------------------


@pytest.fixture
def pilot1_run(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=1)
    bench = freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})
    results = tmp_path / "results"
    p0 = Run("p0", results)
    run_pilot0(Pilot0Config(benchmark=str(bench)), p0,
               guarded(results, FakeDecisionProvider(malformed_every=0), alias="jev", run_id="p0"), benchmark_dir=bench)
    p1 = Run("p1", results)
    clients = {a: guarded(results, FakeChatProvider(malformed_every=0), alias=a, run_id="p1") for a in ALIASES}
    run_pilot1(Pilot1Config(), p1, clients, pilot0_dir=p0.dir)
    return p1.dir


def test_judge_run_writes_one_judgment_per_explanation_and_resumes(tmp_path, pilot1_run):
    results = tmp_path / "results"
    run = Run("j", results)
    client = guarded(results, FakeChatProvider(malformed_every=0), alias="judge", run_id="j")
    assert run_judge(JudgeConfig(), run, client, pilot1_dir=pilot1_run) == "completed"
    expls = [r for r in run.writer("judgments").rows()]
    written = [json.loads(line) for line in (pilot1_run / "explanations.jsonl").read_text().splitlines()]
    judged = [e for e in written if e["parse_ok"] and e["explainer"] != "none"]
    assert len(expls) == len(judged) and all(r["parse_ok"] for r in expls)
    calls = client.provider.calls
    assert run_judge(JudgeConfig(), run, client, pilot1_dir=pilot1_run) == "completed"
    assert client.provider.calls == calls
    out = tmp_path / "sheet.csv"
    assert export_labeling(run.dir, out, n=6) == 6
    header = next(csv.reader(out.open()))
    assert "justification" in header and "quote" not in header


def test_judge_from_an_explainers_family_is_refused(tmp_path, pilot1_run):
    results = tmp_path / "results"
    client = guarded(results, FakeChatProvider(), alias="explainer_open", run_id="j")
    with pytest.raises(ValueError):
        run_judge(JudgeConfig(), Run("j", results), client, pilot1_dir=pilot1_run)


# ---- whose answer ---------------------------------------------------------------------------


def test_standing_apart():
    assert standing_apart("approve", "deny", "deny") == "jev"
    assert standing_apart("deny", "approve", "deny") == "own"
    assert standing_apart("approve", "approve", "deny") == "policy"
    assert standing_apart("approve", "deny", "undetermined") == "all"
    assert standing_apart("deny", "deny", "deny") is None


def test_pull_is_measured_against_the_control():
    # Ten variants where the explainer's own answer (approve) stands apart from Jev and the policy (deny).
    sims, control, own, policy = [], {}, {}, {}
    for i in range(10):
        vid = f"v{i}"
        policy[vid] = "deny"
        own[("e", vid)] = "approve"
        control[vid] = {"variant_id": vid, "parse_ok": True, "predicted_decision": "approve" if i < 2 else "deny"}
        sims.append({"explainer": "e", "condition": "free_form", "variant_id": vid, "parse_ok": True,
                     "jev_decision": "deny", "predicted_decision": "approve" if i < 7 else "deny"})
    w = whose_answer(sims, control, own, policy)["e/free_form"]
    assert w["toward_own"] == {"n": 10, "with_explanation": 0.7, "control": 0.2, "pull": 0.5, "identified": True}
    assert w["toward_jev"]["n"] == 0
    ident = identification(own, {vid: "deny" for vid in policy}, {vid: "sweep_point" for vid in policy}, policy)
    assert ident["e"]["apart:own"] == 10 and ident["e"]["own_identifiable"]


def test_screening_run_then_resume_with_separating_cases(tmp_path):
    cases = RefundCaseGenerator(seed=1).generate(per_type=1)
    bench = freeze_benchmark(cases, tmp_path / "bench", {"policy": {}})
    results = tmp_path / "results"
    p0 = Run("p0", results)
    run_pilot0(Pilot0Config(benchmark=str(bench)), p0,
               guarded(results, FakeDecisionProvider(malformed_every=0), alias="jev", run_id="p0"), benchmark_dir=bench)
    p1 = Run("p1", results)
    clients = {a: guarded(results, FakeChatProvider(malformed_every=0), alias=a, run_id="p1") for a in ALIASES}
    assert run_pilot1(Pilot1Config(steps=("decisions",)), p1, clients, pilot0_dir=p0.dir) == "completed"
    assert p1.writer("explainer_decisions").rows() and not p1.writer("explanations").rows()
    assert not p1.writer("simulations").rows()

    calls = {a: c.provider.calls for a, c in clients.items()}
    config = Pilot1Config(explain_cases="separating")
    assert run_pilot1(config, p1, clients, pilot0_dir=p0.dir) == "completed"
    explained = set(p1.manifest()["explained_cases"])
    assert explained and explained <= {c.case_id for c in cases}
    assert {r["case_id"] for r in p1.writer("explanations").rows()} <= explained
    # The decisions were not asked again.
    decisions = p1.writer("explainer_decisions").rows()
    assert len(decisions) == 2 * 3 * sum(len(c.variants) for c in cases)
    assert clients["explainer_open"].provider.calls > calls["explainer_open"]
    with pytest.raises(ValueError):
        Pilot1Config(steps=("simulate",))
