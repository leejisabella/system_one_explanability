import json

import pytest

from sysone.benchmark.cases import CaseType, VariantKind
from sysone.benchmark.domains import DOMAINS, DomainCase, DomainCaseGenerator, render_case_text
from sysone.benchmark.store import freeze_benchmark, load_benchmark
from sysone.budget import CostGuard, Ledger, Price
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.clients.fake import FakeProvider, choice_answer
from sysone.decision.jev import REFUND_TASK, JevDecisionModel, parse_answer
from sysone.registry import REPO_ROOT

NAMES = sorted(DOMAINS)
# The frozen benchmark of each domain.
FROZEN = {"moderation": "moderation_v1", "fraud": "fraud_v1", "tool_selection": "tools_v1",
          "routing": "routing_v1", "eligibility": "eligibility_v1"}


@pytest.fixture(scope="module", params=NAMES)
def domain_cases(request):
    domain = DOMAINS[request.param]
    return domain, DomainCaseGenerator(domain, seed=0).generate(per_type=10)


def outcome(domain, record):
    return domain.evaluate(record, domain.default_params).label


def test_ten_cases_of_every_type(domain_cases):
    _, cases = domain_cases
    assert len(cases) == 70
    for t in CaseType:
        assert sum(c.case_type == t for c in cases) == 10


def test_generation_is_deterministic(domain_cases):
    domain, cases = domain_cases
    again = DomainCaseGenerator(domain, seed=0).generate(per_type=10)
    assert [c.to_dict() for c in again] == [c.to_dict() for c in cases]


@pytest.mark.parametrize("name", NAMES)
def test_other_seeds_generate(name):
    for seed in range(1, 6):
        DomainCaseGenerator(DOMAINS[name], seed=seed).generate(per_type=10)


def test_case_ids_do_not_follow_case_type(domain_cases):
    _, cases = domain_cases
    assert len({c.case_type for c in cases[:10]}) > 1


def test_outcomes_match_policy_and_task_options(domain_cases):
    domain, cases = domain_cases
    for c in cases:
        assert outcome(domain, c.true_record) == c.true_outcome
        assert outcome(domain, c.presented_record) == c.presented_outcome
        for v in c.variants:
            assert v.policy_outcome == outcome(domain, v.record) and v.policy_outcome in domain.task.labels
    assert domain.abstain_label in domain.task.labels


def test_case_types_have_their_defining_shape(domain_cases):
    domain, cases = domain_cases
    for c in cases:
        kinds = [v.kind for v in c.variants]
        outs = [v.policy_outcome for v in c.variants]
        assert kinds[0] == VariantKind.ORIGINAL
        if c.case_type == CaseType.CORRECT:
            assert c.true_record == c.presented_record and outs[0] != outs[1]
        elif c.case_type == CaseType.INCORRECT_INPUT:
            assert c.true_outcome != c.presented_outcome and [d.type for d in c.defects] == ["I"]
            assert outs[1] == c.true_outcome
        elif c.case_type == CaseType.IGNORED_EVIDENCE:
            assert [v.exception_in for v in c.variants] == ["attachment", "field", "attachment"]
            assert outs[0] == outs[1] == c.true_outcome != outs[2]
        elif c.case_type == CaseType.IRRELEVANT_CUE:
            assert len(set(outs)) == 1 and len(c.variants) == 1 + len(domain.cues)
        elif c.case_type == CaseType.AMBIGUOUS:
            assert c.true_outcome == domain.abstain_label and outs[1] != outs[2]
        elif c.case_type == CaseType.HARD_NEGATIVE:
            assert c.true_outcome == c.presented_outcome == outs[1] != outs[2]
            assert c.true_record != c.presented_record
        elif c.case_type == CaseType.THRESHOLD_SWEEP:
            assert kinds[1:] == [VariantKind.SWEEP_POINT] * 4 and len(set(outs)) > 1
            assert len({json.dumps(v.record, sort_keys=True) for v in c.variants}) == 5


def test_sweeps_cover_the_threshold_in_every_domain(domain_cases):
    domain, cases = domain_cases
    sweep_outcomes = {v.policy_outcome for c in cases if c.case_type == CaseType.THRESHOLD_SWEEP for v in c.variants}
    assert len(sweep_outcomes) >= 2


def test_render_hides_labels_and_places_exceptions(domain_cases):
    domain, cases = domain_cases
    for c in cases:
        for v in c.variants:
            text = render_case_text(domain, v)
            assert c.case_type not in text and "ground truth" not in text.lower()
            assert "Attachments:" in text
        if c.case_type == CaseType.IGNORED_EVIDENCE:
            original, salient = (render_case_text(domain, v) for v in c.variants[:2])
            assert original != salient


def test_case_round_trip(domain_cases):
    _, cases = domain_cases
    for c in cases:
        assert DomainCase.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_freeze_and_load_round_trip(tmp_path, domain_cases):
    domain, cases = domain_cases
    path = freeze_benchmark(cases, tmp_path / domain.name, {"domain": domain.name})
    loaded, meta = load_benchmark(path)
    assert meta["n_cases"] == 70 and all(isinstance(c, DomainCase) for c in loaded)
    assert [c.to_dict() for c in loaded] == [c.to_dict() for c in sorted(cases, key=lambda c: c.case_id)]


@pytest.mark.parametrize("name", NAMES)
def test_frozen_benchmark_regenerates(name):
    cases, meta = load_benchmark(REPO_ROOT / "benchmarks" / FROZEN[name])
    assert meta["domain"] == name and meta["generator_version"] == DOMAINS[name].generator_version
    again = DomainCaseGenerator(DOMAINS[name], seed=meta["seed"]).generate(meta["per_type"])
    assert [c.to_dict() for c in sorted(again, key=lambda c: c.case_id)] == [c.to_dict() for c in cases]


# ---- Jev on other domains ------------------------------------------------------------------


def guarded(tmp_path):
    guard = CostGuard(Ledger.for_run(tmp_path, "r"), per_call_cap_usd=2.0, run_cap_usd=25.0)
    return GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(1.0, 0.0)), FakeProvider(), guard,
                         sleep=lambda s: None)


def test_refund_request_is_unchanged_by_the_task_parameter(tmp_path):
    client = guarded(tmp_path)
    default = JevDecisionModel(client).build_request("P", "C", sample_index=1)
    explicit = JevDecisionModel(client, task=REFUND_TASK).build_request("P", "C", sample_index=1)
    assert default == explicit and client.request_key(default) == client.request_key(explicit)


@pytest.mark.parametrize("name", NAMES)
def test_domain_task_builds_its_own_question(tmp_path, name):
    task = DOMAINS[name].task
    jev = JevDecisionModel(guarded(tmp_path), task=task)
    orders = [list(jev.build_request("P", "C", sample_index=i).questions[task.question]["criteria"])
              for i in range(len(task.labels))]
    assert [o[0] for o in orders] == list(task.labels)
    req = jev.build_request("P", "C")
    assert req.state == {task.policy_key: "P", "case_file": "C"}
    probs = {label: 1 / len(task.labels) for label in task.labels}
    assert parse_answer(choice_answer(probs), task.labels).parse_ok
    # A refund answer doesn't parse as an answer to another domain's question.
    refund = choice_answer({"approve": 0.5, "deny": 0.5, "undetermined": 0.0})
    assert not parse_answer(refund, task.labels).parse_ok
