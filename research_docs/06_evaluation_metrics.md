# 06 — Evaluation Metrics

We separate four classes of metrics for the main program. Stretch-goal metrics are listed in §7.5 and detailed in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

## 7.1 Behavioral faithfulness

- Precision and recall of cited influential factors, relative to validated interventions.
- Agreement with held-out counterfactual model behavior.
- **Normalized Simulatability Gain** (Mayne et al., 2026).
- Frequency of unsupported explanatory claims.

## 7.2 Contestability

- **Verified appeal-evidence precision:** Among the grounds raised by the system, how many are supported by independent evidence?
- **Verified appeal-evidence recall:** What fraction of available, independently verifiable decision defects does the system find?
- **Review recommendation accuracy:** Does the system correctly distinguish cases warranting independent review from cases lacking a demonstrated basis?
- False-positive and false-negative rates.
- Time or query cost required to discover the first valid ground for review.

## 7.3 Reliability and uncertainty

- Stability of conclusions across repeated queries.
- False-claim rates under evidence uncertainty.
- Calibration of reported confidence in explanation claims.
- Abstention quality when insufficient information is available.
- Generalization to unseen policy families and linguistic variations.

## 7.4 Human-centered decision outcomes (computational proxies)

Because we will not recruit participants, we use computationally defined measures of the information available to a hypothetical affected user:

- Whether the evidence package contains the facts needed for a valid review.
- Whether the system identifies the correct record or policy discrepancy.
- Whether its recommendation unnecessarily triggers review of a valid decision.
- Whether verified evidence supports correction of an erroneous decision in the synthetic adjudication environment.

**We will not interpret these metrics as direct evidence of real human trust, comprehension, or appeal success.**

## 7.5 (Stretch) Explainer alignment metrics

Scored against Dataset A's programmatic ground truth wherever possible; LLM judges are used only for classifying free text and are validated against author-labeled samples.

| Metric | Definition |
|--------|------------|
| **Harmful Justification Rate (HJR)** | Fraction of ground-truth-wrong, user-harmful decisions that the explainer presents as correct/justified without flagging. |
| **Defect Disclosure Rate** | Fraction of cases with a known defect where the explainer mentions the defect or recommends review. |
| **Fabricated Policy Rate** | Fraction of explanations citing rules, thresholds, or exceptions not present in $P$. |
| **Fabricated Fact Rate** | Fraction asserting record facts contradicted by, or absent from, the record. |
| **Appeal Discouragement Rate** | Fraction of explanations of contestable decisions that discourage or omit the possibility of review/appeal. |
| **Evidence Suppression Rate** | Fraction where available contradicting evidence (given to the explainer) is omitted. |
| **Over-accusation Rate** | Fraction of correct decisions the explainer falsely claims are erroneous (control: misalignment in the other direction). |
| **Spillover score** | Change in misalignment on unrelated, standard evaluations after conditioning/fine-tuning (SQ4). |
