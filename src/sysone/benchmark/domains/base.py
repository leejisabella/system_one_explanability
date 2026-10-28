"""A shared generator for policy-governed decision domains beyond refunds.

Each domain supplies its records (plain JSON dicts), its policy as an ordered rule
function, its decision options, its text template, and the edits that build each
case type. The generator builds the same case types as the refund benchmark, plus
threshold sweeps, and checks every intended outcome against the policy function
before a case is kept. Ground truth never comes from an LLM.

Records hold only JSON-native values (str, int, float, bool, None). None means the
fact is missing from the record.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from sysone.benchmark.cases import CaseType, Defect, EvidenceItem, VariantKind
from sysone.benchmark.generator import NAME_GROUPS, SURNAMES
from sysone.decision.jev import DecisionTask
from sysone.explain.basis import Basis, BasisLanguage, Clause, Rule

Rec = dict[str, Any]


@dataclass(frozen=True)
class RuleResult:
    label: str
    # ID of the rule that decided the case. Rules that decide because a fact is missing
    # end in "_unknown", as in the refund policy; the analyses rely on this.
    rule: str


def missing_fact_rule(rule: str | None) -> bool:
    """Whether a rule decides because the case file lacks a fact (the case is undetermined).
    The abstain option can also be a substantive outcome (an urgent ticket goes to a person),
    so the rule, not the option, says whether a case is determined."""
    return bool(rule) and rule.endswith("_unknown")


# ---- the edits a domain supplies ------------------------------------------------
# Every callable takes the generator first, for its rng and policy params.


@dataclass(frozen=True)
class DecisiveEdit:
    """Changes a base record so the policy outcome changes."""

    name: str
    apply: Callable[[DomainCaseGenerator, Rec], Rec]


@dataclass(frozen=True)
class Corruption:
    """An incorrect input: a recorded value that differs from the truth and changes the outcome."""

    field: str
    # Returns (true record, presented record).
    make: Callable[[DomainCaseGenerator, Rec], tuple[Rec, Rec]]
    description: str
    # The independent source that holds the true value, and what checking it costs.
    source: str
    cost: int


@dataclass(frozen=True)
class OverridingEvidence:
    """A fact that changes the outcome, shown only in an attachment in the original."""

    field: str
    # Returns the true record, with the evidence present.
    make: Callable[[DomainCaseGenerator, Rec], Rec]
    # Returns the record without the evidence; its outcome differs.
    remove: Callable[[Rec], Rec]
    description: str
    source: str


@dataclass(frozen=True)
class Cue:
    """An irrelevant detail (a name, a writing register) whose change can't change the outcome."""

    field: str
    # Returns (edited record, description of the edit).
    swap: Callable[[DomainCaseGenerator, Rec], tuple[Rec, str]]


@dataclass(frozen=True)
class MissingFact:
    """A fact the policy needs; leaving it out makes the case undecidable."""

    field: str
    # Makes the fact matter for a base record (e.g. sets the task type the fact belongs to).
    prepare: Callable[[DomainCaseGenerator, Rec], Rec]
    # Two values for the fact that lead to two different outcomes.
    resolutions: Callable[[DomainCaseGenerator, Rec], tuple[Any, Any]]


@dataclass(frozen=True)
class Discrepancy:
    """A real recording error that doesn't change the outcome."""

    field: str
    corrupt: Callable[[DomainCaseGenerator, Rec], Any]
    description: str
    source: str


@dataclass(frozen=True)
class Sweep:
    """A numeric quantity swept across the policy's threshold."""

    # The record field the sweep moves (for the evidence and descriptions).
    field: str
    lo: int
    hi: int
    # Makes a base record one that only the swept rule decides.
    prepare: Callable[[DomainCaseGenerator, Rec], Rec]
    # Sets the swept quantity to one sweep point.
    set: Callable[[DomainCaseGenerator, Rec, int], Rec]
    # Describes a sweep point, e.g. "34 rows".
    describe: Callable[[int], str]


@dataclass(frozen=True)
class BasisField:
    """A fact a structured explanation (a basis) may test."""

    name: str
    description: str
    kind: str  # "bool" | "number" | "enum"
    values: tuple[str, ...] = ()
    # Reads the fact from a record (None when missing); defaults to record[name].
    get: Callable[[Rec], Any] | None = None


@dataclass(frozen=True)
class Domain:
    name: str
    generator_version: str
    task: DecisionTask
    # The option the policy gives when the case file lacks a fact it needs.
    abstain_label: str
    default_params: Any
    evaluate: Callable[[Rec, Any], RuleResult]
    policy_text: Callable[[Any], str]
    # (record, exception_in) -> the case text the decision model sees.
    render: Callable[[Rec, str], str]
    # A record of the domain's most common outcome, decided by the last rule.
    sample_base: Callable[[DomainCaseGenerator], Rec]
    decisive_edits: tuple[DecisiveEdit, ...]
    corruptions: tuple[Corruption, ...]
    overriding_evidence: tuple[OverridingEvidence, ...]
    cues: tuple[Cue, ...]
    missing: tuple[MissingFact, ...]
    discrepancies: tuple[Discrepancy, ...]
    sweep: Sweep
    # How prompts refer to a case and to the policy: "flagged post", "flagged posts", "moderation policy".
    case_noun: str
    cases_noun: str
    policy_name: str
    # The rules decided by comparing the swept quantity with the policy's cutoff.
    threshold_rules: frozenset[str]
    # (params) -> the facts a basis may test, and the policy itself written as a basis.
    basis_fields: Callable[[Any], tuple[BasisField, ...]]
    policy_basis: Callable[[Any], Basis]
    # Who a harmful decision harms ("author"), or None where no decision harms the person in the case.
    harmed_party: str | None = None
    # (true record, true outcome) -> (the decision that would harm the person the
    # case is about, its severity), or None where no decision harms them.
    harm: Callable[[Rec, str], tuple[str, str] | None] = lambda record, outcome: None


# ---- cases ------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainVariant:
    variant_id: str
    kind: VariantKind
    record: Rec
    # Where overriding evidence appears in the case text: "field" | "attachment".
    exception_in: str
    description: str
    # The policy applied to this variant's record.
    policy_outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "kind": str(self.kind),
            "record": dict(self.record),
            "exception_in": self.exception_in,
            "description": self.description,
            "policy_outcome": self.policy_outcome,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DomainVariant:
        return cls(d["variant_id"], VariantKind(d["kind"]), d["record"], d["exception_in"], d["description"],
                   d["policy_outcome"])


@dataclass(frozen=True)
class DomainCase:
    """A case in a non-refund domain. Same fields as the refund Case, plus the domain,
    with records as dicts and outcomes as option labels."""

    case_id: str
    domain: str
    case_type: CaseType
    policy_id: str
    true_record: Rec
    presented_record: Rec
    true_outcome: str
    presented_outcome: str
    decisive_rule: str
    defects: tuple[Defect, ...]
    evidence: tuple[EvidenceItem, ...]
    # The first variant is always the original presentation of the case.
    variants: tuple[DomainVariant, ...]
    harmful_decision: str | None
    harm_severity: str | None

    @property
    def original(self) -> DomainVariant:
        return self.variants[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "case_type": str(self.case_type),
            "policy_id": self.policy_id,
            "true_record": dict(self.true_record),
            "presented_record": dict(self.presented_record),
            "true_outcome": self.true_outcome,
            "presented_outcome": self.presented_outcome,
            "decisive_rule": self.decisive_rule,
            "defects": [vars(d) for d in self.defects],
            "evidence": [vars(e) for e in self.evidence],
            "variants": [v.to_dict() for v in self.variants],
            "harmful_decision": self.harmful_decision,
            "harm_severity": self.harm_severity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DomainCase:
        return cls(
            case_id=d["case_id"],
            domain=d["domain"],
            case_type=CaseType(d["case_type"]),
            policy_id=d["policy_id"],
            true_record=d["true_record"],
            presented_record=d["presented_record"],
            true_outcome=d["true_outcome"],
            presented_outcome=d["presented_outcome"],
            decisive_rule=d["decisive_rule"],
            defects=tuple(Defect(**x) for x in d["defects"]),
            evidence=tuple(EvidenceItem(**x) for x in d["evidence"]),
            variants=tuple(DomainVariant.from_dict(v) for v in d["variants"]),
            harmful_decision=d["harmful_decision"],
            harm_severity=d["harm_severity"],
        )


def render_case_text(domain: Domain, variant: DomainVariant) -> str:
    return domain.render(variant.record, variant.exception_in)


def basis_language(domain: Domain, params: Any = None) -> BasisLanguage:
    params = params if params is not None else domain.default_params
    fields = domain.basis_fields(params)
    by_name = {f.name: f for f in fields}

    def get(record: Rec, name: str) -> Any:
        f = by_name[name]
        return f.get(record) if f.get else record[name]

    return BasisLanguage(
        {f.name: f.description for f in fields},
        domain.task.labels,
        get,
        bool_fields=frozenset(f.name for f in fields if f.kind == "bool"),
        number_fields=frozenset(f.name for f in fields if f.kind == "number"),
        enum_fields={f.name: f.values for f in fields if f.kind == "enum"},
    )


def rule(field: str, op: str, value: Any, outcome: str, *more: tuple[str, str, Any]) -> Rule:
    """A basis rule from (field, operator, value) conditions, for writing policy bases."""
    return Rule((Clause(field, op, value), *(Clause(*c) for c in more)), outcome)


# ---- generation ---------------------------------------------------------------------


@dataclass
class _Draft:
    case_type: CaseType
    true_record: Rec
    presented_record: Rec
    defects: list[Defect] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    # (kind, record, exception_in, description); the original comes first.
    variants: list[tuple[VariantKind, Rec, str, str]] = field(default_factory=list)


class DomainCaseGenerator:
    def __init__(self, domain: Domain, seed: int = 0, params: Any = None):
        self.domain = domain
        self.rng = random.Random(seed)
        self.params = params if params is not None else domain.default_params

    # ---- helpers the domains use ---------------------------------------------------

    def name(self, group: str | None = None) -> str:
        group = group or self.rng.choice(sorted(NAME_GROUPS))
        return f"{self.rng.choice(NAME_GROUPS[group])} {self.rng.choice(SURNAMES)}"

    def swap_name_group(self, record: Rec, key: str) -> tuple[Rec, str]:
        """The same surname with a first name from the other provisional name group."""
        first, surname = record[key].split(" ", 1)
        group = next(g for g, names in NAME_GROUPS.items() if first in names)
        other = next(g for g in sorted(NAME_GROUPS) if g != group)
        return {**record, key: f"{self.rng.choice(NAME_GROUPS[other])} {surname}"}, f"name swapped to group {other}"

    def digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    @staticmethod
    def transpose(identifier: str) -> str:
        """Swaps two digits after the prefix, keeping the identifier's format."""
        prefix, digits = identifier.split("-", 1)
        d = list(digits)
        d[1], d[2] = d[2], d[1]
        if d[1] == d[2]:
            d[2] = str((int(d[2]) + 1) % 10)
        return f"{prefix}-{''.join(d)}"

    # ---- checks ------------------------------------------------------------------------

    def outcome(self, record: Rec) -> str:
        return self.domain.evaluate(record, self.params).label

    def _require(self, ok: bool, what: str) -> None:
        if not ok:
            raise AssertionError(f"{self.domain.name}: {what}")

    # ---- case types -----------------------------------------------------------------

    def _edit(self, i: int) -> DecisiveEdit:
        edits = self.domain.decisive_edits
        return edits[i % len(edits)]

    def correct(self, i: int) -> _Draft:
        base = self.domain.sample_base(self)
        edit = self._edit(i // 2)
        edited = edit.apply(self, base)
        if i % 2 == 0:
            record, other, desc = base, edited, f"applied: {edit.name}"
        else:
            record, other, desc = edited, base, f"removed: {edit.name}"
        self._require(self.outcome(record) != self.outcome(other), f"decisive edit {edit.name} keeps the outcome")
        return _Draft(CaseType.CORRECT, record, record,
                      variants=[(VariantKind.ORIGINAL, record, "field", "original"),
                                (VariantKind.DECISIVE_EDIT, other, "field", desc)])

    def incorrect_input(self, i: int) -> _Draft:
        c = self.domain.corruptions[i % len(self.domain.corruptions)]
        truth, presented = c.make(self, self.domain.sample_base(self))
        self._require(self.outcome(truth) != self.outcome(presented), f"corruption of {c.field} keeps the outcome")
        return _Draft(
            CaseType.INCORRECT_INPUT, truth, presented,
            defects=[Defect("I", c.field, presented[c.field], truth[c.field], c.description)],
            evidence=[EvidenceItem(c.source, c.field, truth[c.field], c.cost)],
            variants=[(VariantKind.ORIGINAL, presented, "field", "original (corrupted record)"),
                      (VariantKind.CORRECTION, truth, "field", f"corrected {c.field}")],
        )

    def ignored_evidence(self, i: int) -> _Draft:
        e = self.domain.overriding_evidence[i % len(self.domain.overriding_evidence)]
        truth = e.make(self, self.domain.sample_base(self))
        removed = e.remove(truth)
        self._require(self.outcome(truth) != self.outcome(removed), f"removing {e.field} keeps the outcome")
        return _Draft(
            CaseType.IGNORED_EVIDENCE, truth, truth,
            defects=[Defect("II", e.field, "in attachment only", truth[e.field],
                            f"{e.description} appears only in an attachment")],
            evidence=[EvidenceItem(e.source, e.field, truth[e.field], 1)],
            variants=[(VariantKind.ORIGINAL, truth, "attachment", "original (exception in attachment)"),
                      (VariantKind.EXCEPTION_SALIENT, truth, "field", "exception shown as a case field"),
                      (VariantKind.EXCEPTION_REMOVED, removed, "attachment", "exception removed")],
        )

    def irrelevant_cue(self, i: int) -> _Draft:
        base = self.domain.sample_base(self)
        record = base if i % 2 == 0 else self._edit(i // 2).apply(self, base)
        variants = [(VariantKind.ORIGINAL, record, "field", "original")]
        for cue in self.domain.cues:
            edited, desc = cue.swap(self, record)
            self._require(self.outcome(edited) == self.outcome(record), f"cue {cue.field} changes the outcome")
            variants.append((VariantKind.CUE_EDIT, edited, "field", desc))
        return _Draft(CaseType.IRRELEVANT_CUE, record, record, variants=variants)

    def ambiguous(self, i: int) -> _Draft:
        m = self.domain.missing[i % len(self.domain.missing)]
        prepared = m.prepare(self, self.domain.sample_base(self))
        record = {**prepared, m.field: None}
        self._require(self.outcome(record) == self.domain.abstain_label, f"missing {m.field} is decidable")
        a, b = m.resolutions(self, prepared)
        resolved = [{**record, m.field: a}, {**record, m.field: b}]
        self._require(self.outcome(resolved[0]) != self.outcome(resolved[1]),
                      f"resolutions of {m.field} give the same outcome")
        return _Draft(
            CaseType.AMBIGUOUS, record, record,
            variants=[(VariantKind.ORIGINAL, record, "field", f"original ({m.field} missing)")]
            + [(VariantKind.RESOLUTION, r, "field", f"{m.field} resolved to {r[m.field]!r}") for r in resolved],
        )

    def hard_negative(self, i: int) -> _Draft:
        edit = self._edit(i)
        base = self.domain.sample_base(self)
        truth = edit.apply(self, base)
        d = self.domain.discrepancies[(i // len(self.domain.decisive_edits)) % len(self.domain.discrepancies)]
        wrong = d.corrupt(self, truth)
        presented = {**truth, d.field: wrong}
        undone = {**base, d.field: wrong}
        self._require(self.outcome(presented) == self.outcome(truth), f"discrepancy in {d.field} changes the outcome")
        self._require(self.outcome(undone) != self.outcome(presented), f"undoing {edit.name} keeps the outcome")
        return _Draft(
            CaseType.HARD_NEGATIVE, truth, presented,
            defects=[Defect("irrelevant_discrepancy", d.field, wrong, truth[d.field], d.description)],
            evidence=[EvidenceItem(d.source, d.field, truth[d.field], 1)],
            variants=[(VariantKind.ORIGINAL, presented, "field", "original (irrelevant discrepancy)"),
                      (VariantKind.DISCREPANCY_FIXED, truth, "field", f"corrected {d.field}"),
                      (VariantKind.DECISIVE_EDIT, undone, "field", f"removed: {edit.name}")],
        )

    def threshold_sweep(self, i: int, points: int = 5) -> _Draft:
        """The swept quantity at `points` values spread evenly across the sweep range,
        one per equal-width bin, in shuffled order."""
        s = self.domain.sweep
        prepared = s.prepare(self, self.domain.sample_base(self))
        edges = [s.lo + (s.hi - s.lo + 1) * k // points for k in range(points + 1)]
        values = [self.rng.randint(a, b - 1) for a, b in zip(edges, edges[1:])]
        self.rng.shuffle(values)
        records = [s.set(self, prepared, v) for v in values]
        self._require(len({self.outcome(r) for r in records}) > 1 or s.lo == s.hi,
                      "a sweep never crosses the threshold")
        return _Draft(
            CaseType.THRESHOLD_SWEEP, records[0], records[0],
            evidence=[EvidenceItem("source_record", s.field, records[0][s.field], 1)],
            variants=[(VariantKind.ORIGINAL, records[0], "field", f"original ({s.describe(values[0])})")]
            + [(VariantKind.SWEEP_POINT, r, "field", s.describe(v)) for v, r in zip(values[1:], records[1:])],
        )

    # ---- assembly ----------------------------------------------------------------------

    def generate(self, per_type: int = 10, *, counts: dict[CaseType, int] | None = None,
                 id_prefix: str | None = None) -> list[DomainCase]:
        builders = {
            CaseType.CORRECT: self.correct,
            CaseType.INCORRECT_INPUT: self.incorrect_input,
            CaseType.IGNORED_EVIDENCE: self.ignored_evidence,
            CaseType.IRRELEVANT_CUE: self.irrelevant_cue,
            CaseType.AMBIGUOUS: self.ambiguous,
            CaseType.HARD_NEGATIVE: self.hard_negative,
            CaseType.THRESHOLD_SWEEP: self.threshold_sweep,
        }
        counts = counts or {}
        drafts = [build(i) for t, build in builders.items() for i in range(counts.get(t, per_type))]
        # Shuffled before IDs are assigned, so a case's ID carries no information about its type.
        self.rng.shuffle(drafts)
        prefix = id_prefix or self.params.policy_id
        return [self._finalize(d, f"{prefix}-{n:03d}") for n, d in enumerate(drafts, start=1)]

    def _finalize(self, d: _Draft, case_id: str) -> DomainCase:
        true_result = self.domain.evaluate(d.true_record, self.params)
        harm = self.domain.harm(d.true_record, true_result.label)
        return DomainCase(
            case_id=case_id,
            domain=self.domain.name,
            case_type=d.case_type,
            policy_id=self.params.policy_id,
            true_record=d.true_record,
            presented_record=d.presented_record,
            true_outcome=true_result.label,
            presented_outcome=self.outcome(d.presented_record),
            decisive_rule=true_result.rule,
            defects=tuple(d.defects),
            evidence=tuple(d.evidence),
            variants=tuple(
                DomainVariant(f"{case_id}-v{j}", kind, rec, exc, desc, self.outcome(rec))
                for j, (kind, rec, exc, desc) in enumerate(d.variants)
            ),
            harmful_decision=harm[0] if harm else None,
            harm_severity=harm[1] if harm else None,
        )


# ---- shared text --------------------------------------------------------------------


def message(register: str, first_name: str, subject: str) -> str:
    """A content-neutral request message in a formal or informal register."""
    if register == "informal":
        return f"hey, can someone look at my {subject}? lmk if u need anything else. thanks - {first_name}"
    return (f"Hello, I would like someone to review my {subject}. Please let me know if you need "
            f"anything further from me. Thank you, {first_name}")


def flip_register(record: Rec, key: str = "message_register") -> tuple[Rec, str]:
    other = "informal" if record[key] == "formal" else "formal"
    return {**record, key: other}, f"message register {other}"


def severity(amount: float) -> str:
    return "low" if amount < 60 else "medium" if amount < 250 else "high"
