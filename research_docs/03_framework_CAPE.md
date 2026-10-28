# 03 — Proposed Framework: CAPE (Contestability-Aware Probing and Explanation)

## 4.1 Problem formulation

Let:

- $x$ — an input record.
- $f$ — an opaque AI decision model (primarily Jev).
- $y = f(x)$ — its decision.
- $P$ — the relevant decision policy.
- $e$ — independent factual evidence.
- $B$ — a budget of allowed model queries.

The task is to produce an **evidence package**:

$$
z = \mathcal{A}\left(x,\ y,\ P_{\mathrm{known}},\ e_{\mathrm{available}},\ B\right)
$$

where the information available about the policy ($P_{\mathrm{known}} \subseteq P$) and the record ($e_{\mathrm{available}} \subseteq e$) may be incomplete.

The evidence package $z$ must contain:

1. An explanation of the model's observed behavior.
2. Candidate grounds for review.
3. Evidence supporting or contradicting those grounds.
4. Uncertainty about each claim.
5. An indication of whether the available evidence is sufficient to recommend independent review.

**We distinguish recommending review from establishing that a decision must be overturned.** Three levels of conclusion are kept separate: *suspicion*, *review eligibility*, and *demonstrated decision error*.

### Suggested evidence-package schema (implementation sketch)

```json
{
  "case_id": "...",
  "decision": {"label": "...", "score": 0.0},
  "observed_behavior": [
    {"intervention": "...", "original_output": "...", "counterfactual_output": "...", "n_repeats": 0, "stable": true}
  ],
  "candidate_grounds": [
    {
      "type": "I | II | III | multiplicity",
      "claim_kind": "behavioral | factual | policy",
      "statement": "...",
      "supporting_evidence": ["..."],
      "contradicting_evidence": ["..."],
      "confidence": 0.0,
      "verified": true
    }
  ],
  "recommendation": "no_basis | insufficient_evidence | recommend_review",
  "queries_used": 0,
  "evidence_checks_used": 0
}
```

## 4.2 Candidate evidence classes

**Type I — Incorrect recorded information.** The decision depends on an input value that may not match the underlying record.
*Example:* A refund request is denied because the recorded purchase date is incorrect.

**Type II — Neglected overriding evidence or policy requirements.** The decision ignores information that should change the outcome under the applicable policy.
*Example:* An eligibility model denies a request despite a qualifying exception documented elsewhere in the case.

**Type III — Decision inconsistency or inappropriate sensitivity.** The model produces substantially different results for cases that should be treated equivalently under the policy.
*Example:* A content moderation decision changes when a dialect feature is altered without a meaningful change in the prohibited content.

**Predictive multiplicity.** We will additionally evaluate predictive multiplicity in settings where several plausible decision models are available, rather than treating a single model's local instability as equivalent to multiplicity.

(These classes follow the contestability evidence types identified by Freiesleben et al., 2026.)

## 4.3 Adaptive evidence-seeking

Given a limited query budget, CAPE selects counterfactual tests or evidence checks expected to provide the most useful information about possible decision defects. At each step it maintains a set of competing hypotheses about the decision.

Proposed acquisition objective:

$$
q_t^{*} = \arg\max_{q \in \mathcal{Q}} \frac{\mathbb{E}\left[\Delta V_{\mathrm{contest}}(q) \mid \mathcal{D}_t\right]}{C(q)}
$$

where:

- $\mathcal{Q}$ — the set of admissible queries (model probes and evidence checks).
- $\mathcal{D}_t$ — evidence collected so far.
- $C(q)$ — the cost of query $q$.
- $\Delta V_{\mathrm{contest}}(q)$ — expected improvement in identifying a valid, verifiable basis for review.

**This is an initial objective, not a claimed novel formula.** Open research questions:

- How to estimate $\mathbb{E}[\Delta V_{\mathrm{contest}}]$ efficiently (e.g., expected information gain over a hypothesis set of defect types; value-of-information relative to the review/no-review decision).
- How to select admissible interventions (semantic validity, single-variable changes).
- How it compares with simpler active-learning and black-box explanation strategies (Experiment 4; Experiment 3's probe rungs are a first, small version).

### Hypothesis loop (sketch)

```
initialize hypothesis set H = {no defect, Type I on field k, Type II exception j, Type III cue c, ...}
while budget remains:
    for each admissible q in Q: estimate E[ΔV_contest(q) | D_t] / C(q)
    execute best q; append result to D_t
    update posterior over H
    stop early if review recommendation is stable / confident
verify claims (§4.4) → generate explanation (§4.5)
```

## 4.4 Evidence verification

Rather than immediately converting probe results into explanations, the system verifies each proposed claim. Suppose changing a date changes the model's decision. The system must distinguish:

| Claim type | Example | How it can be established |
|------------|---------|---------------------------|
| **Observed behavior** | The model's decision changes when the date is modified. | Model queries (with repeats for stability). |
| **Factual claim** | The original date is incorrect. | Independent evidence (e.g., authoritative record/receipt). |
| **Policy claim** | The corrected date should change the legitimate outcome. | Independent policy evidence. |

The first can be established through model queries. The second and third require independent evidence. **If those sources are unavailable, the system should avoid confidently asserting that the original decision was erroneous.** This distinction is central to the approach and to RQ4/RQ5.

## 4.5 Explanation generation

Only after evidence collection and verification does an LLM generate a human-readable explanation. Explanations contain structured claims with supporting observations and uncertainty.

**CAPE-style (evidence-bearing):**

> "The recorded transaction date affected the automated decision. When the date was corrected from August 14 to August 4, the model changed its classification from ineligible to eligible. The original receipt supports August 4. Under the stated policy, this discrepancy provides grounds for reviewing the denial."

**Conventional (rationale only):**

> "The model denied the request because it was outside the eligibility window."

The first provides independently checkable evidence; the second only states a possible rationale.

## Relevance to the stretch goal

CAPE constrains the explainer LLM to verbalize *verified* claims from the evidence package rather than freely generating a rationale. This is a candidate mitigation for explainer misalignment (SQ5): an explainer that may only cite verified evidence has less room to fabricate policy or justify a harmful decision. We will test whether this constraint actually holds under adversarial or deployer-aligned pressure (see [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md)).
