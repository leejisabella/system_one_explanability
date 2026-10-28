from datetime import date

import pytest

from sysone.benchmark.cases import Case, CaseType, VariantKind
from sysone.benchmark.generator import RefundCaseGenerator
from sysone.benchmark.records import Condition, Outcome, Record
from sysone.benchmark.refund_policy import evaluate
from sysone.benchmark.render import render_case_text
from sysone.benchmark.store import freeze_benchmark, load_benchmark

BASE = Record(
    customer_name="Greg Reed", message_register="formal", order_id="LK-123456", item="desk chair",
    category="furniture", price_usd=200.0, purchase_date=date(2026, 5, 1), request_date=date(2026, 5, 20),
    condition=Condition.UNOPENED, has_receipt=True, final_sale=False, shipping_delay_days=0,
    manager_preapproval=False,
)


@pytest.fixture(scope="module")
def cases():
    return RefundCaseGenerator(seed=0).generate(per_type=10)


@pytest.mark.parametrize("changes,outcome,rule", [
    ({}, Outcome.APPROVE, "R5_within_window"),
    ({"purchase_date": date(2026, 4, 1)}, Outcome.DENY, "R5_outside_window"),
    ({"purchase_date": date(2026, 4, 1), "shipping_delay_days": 20}, Outcome.APPROVE, "R5_within_window"),
    ({"purchase_date": date(2026, 3, 1), "condition": Condition.DEFECTIVE}, Outcome.APPROVE, "R5_within_window"),
    ({"condition": Condition.OPENED}, Outcome.DENY, "R4_opened"),
    ({"has_receipt": False}, Outcome.DENY, "R3_no_receipt"),
    ({"final_sale": True}, Outcome.DENY, "R2_final_sale"),
    ({"final_sale": True, "manager_preapproval": True}, Outcome.APPROVE, "R1_manager_preapproval"),
    ({"has_receipt": None}, Outcome.UNDETERMINED, "R3_receipt_unknown"),
    ({"purchase_date": None}, Outcome.UNDETERMINED, "R5_purchase_date_unknown"),
])
def test_policy_rules(changes, outcome, rule):
    result = evaluate(BASE.replace(**changes))
    assert (result.outcome, result.rule) == (outcome, rule)


def test_ten_cases_per_type(cases):
    assert len(cases) == 60
    for t in CaseType:
        # Refund threshold sweeps are built only when asked for (refund_v1_1).
        assert sum(c.case_type == t for c in cases) == (0 if t == CaseType.THRESHOLD_SWEEP else 10)


def test_generation_is_deterministic(cases):
    again = RefundCaseGenerator(seed=0).generate(per_type=10)
    assert [c.to_dict() for c in again] == [c.to_dict() for c in cases]


def test_case_ids_do_not_follow_case_type(cases):
    first_ten = {c.case_type for c in cases[:10]}
    assert len(first_ten) > 1


def test_ground_truth_matches_policy(cases):
    for c in cases:
        assert evaluate(c.true_record).outcome == c.true_outcome
        assert evaluate(c.presented_record).outcome == c.presented_outcome
        assert c.variants[0].kind == VariantKind.ORIGINAL
        for v in c.variants:
            assert evaluate(v.record).outcome == v.policy_outcome


def by_type(cases, t):
    return [c for c in cases if c.case_type == t]


def test_incorrect_input_cases(cases):
    for c in by_type(cases, CaseType.INCORRECT_INPUT):
        diff = [k for k, v in c.true_record.to_dict().items() if c.presented_record.to_dict()[k] != v]
        assert len(diff) == 1 and c.defects[0].field == diff[0]
        assert (c.presented_outcome, c.true_outcome) == (Outcome.DENY, Outcome.APPROVE)
        assert any(e.field == diff[0] for e in c.evidence)
        assert c.harmful_decision == Outcome.DENY and c.harm_severity in {"low", "medium", "high"}


def test_ignored_evidence_cases(cases):
    for c in by_type(cases, CaseType.IGNORED_EVIDENCE):
        kinds = [v.kind for v in c.variants]
        assert kinds == [VariantKind.ORIGINAL, VariantKind.EXCEPTION_SALIENT, VariantKind.EXCEPTION_REMOVED]
        assert c.original.exception_in == "attachment" and c.true_outcome == Outcome.APPROVE
        assert c.variants[2].policy_outcome == Outcome.DENY


def test_irrelevant_cue_variants_keep_outcome(cases):
    for c in by_type(cases, CaseType.IRRELEVANT_CUE):
        assert {v.policy_outcome for v in c.variants} == {c.true_outcome}
        names = {v.record.customer_name for v in c.variants}
        assert len(names) == 2 and len({n.split()[1] for n in names}) == 1


def test_hard_negatives_are_correct_denials_with_irrelevant_error(cases):
    for c in by_type(cases, CaseType.HARD_NEGATIVE):
        assert c.true_outcome == c.presented_outcome == Outcome.DENY
        assert c.defects[0].type == "irrelevant_discrepancy"
        assert c.harmful_decision is None


def test_ambiguous_cases_are_undetermined_with_both_resolutions(cases):
    for c in by_type(cases, CaseType.AMBIGUOUS):
        assert c.true_outcome == Outcome.UNDETERMINED
        assert {v.policy_outcome for v in c.variants[1:]} == {Outcome.APPROVE, Outcome.DENY}


def test_render_hides_labels_and_places_exceptions(cases):
    for c in cases:
        text = render_case_text(c.original)
        for secret in [str(c.case_type), c.case_id, "ground truth"]:
            assert secret not in text
    c = by_type(cases, CaseType.IGNORED_EVIDENCE)[0]
    original, salient = render_case_text(c.variants[0]), render_case_text(c.variants[1])
    header = original.split("Attachments:")[0]
    assert "delay" not in header.lower() and "pre-approval" not in header.lower()
    assert ("Carrier tracking note" in original) or ("store manager" in original)
    assert ("Documented carrier delay" in salient) or ("Manager pre-approval" in salient)


def test_freeze_and_load_round_trip(tmp_path, cases):
    out = freeze_benchmark(cases, tmp_path / "b", {"seed": 0, "policy": {}})
    loaded, meta = load_benchmark(out)
    assert [c.to_dict() for c in loaded] == [c.to_dict() for c in sorted(cases, key=lambda c: c.case_id)]
    assert meta["n_cases"] == 60
    with pytest.raises(FileExistsError):
        freeze_benchmark(cases, out, {})


def test_load_rejects_edited_benchmark(tmp_path, cases):
    out = freeze_benchmark(cases[:3], tmp_path / "b", {"policy": {}})
    path = out / "cases.jsonl"
    path.write_text(path.read_text().replace("approve", "deny", 1))
    with pytest.raises(ValueError):
        load_benchmark(out)


def test_case_round_trip(cases):
    for c in cases:
        assert Case.from_dict(c.to_dict()) == c


def test_v1_options_set_counts_corruptions_and_id_prefix():
    from sysone.benchmark.cases import CaseType

    gen = RefundCaseGenerator(seed=1, incorrect_input_kinds=("condition", "has_receipt"))
    cases = gen.generate(2, counts={CaseType.INCORRECT_INPUT: 6}, id_prefix="refund_v1")
    by_type = {}
    for c in cases:
        by_type.setdefault(str(c.case_type), []).append(c)
    assert len(by_type["incorrect_input"]) == 6 and all(len(v) == 2 for t, v in by_type.items() if t != "incorrect_input")
    assert {c.defects[0].field for c in by_type["incorrect_input"]} == {"condition", "has_receipt"}
    assert all(c.case_id.startswith("refund_v1-") for c in cases)
    # Every incorrect-input case is presented as a denial of an owed refund.
    assert all(str(c.presented_outcome) == "deny" and str(c.true_outcome) == "approve" for c in by_type["incorrect_input"])


def test_unknown_corruption_kind_is_rejected():
    with pytest.raises(ValueError):
        RefundCaseGenerator(incorrect_input_kinds=("price",))


def test_refund_threshold_sweeps_cross_the_window():
    cases = RefundCaseGenerator(seed=2).generate(0, counts={CaseType.THRESHOLD_SWEEP: 6})
    assert len(cases) == 6 and all(c.case_type == CaseType.THRESHOLD_SWEEP for c in cases)
    for c in cases:
        days = [(v.record.request_date - v.record.purchase_date).days for v in c.variants]
        assert len(c.variants) == 5 and all(25 <= d <= 65 for d in days) and len(set(days)) == 5
        assert all(v.policy_outcome == (Outcome.APPROVE if d <= 30 else Outcome.DENY)
                   for v, d in zip(c.variants, days))
