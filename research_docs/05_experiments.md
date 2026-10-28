# 05 — Experimental Design

Experiments 1–6 address the main research questions. Stretch-goal experiments (S1–S5) are in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

| Experiment | RQ | Primary data |
|------------|----|--------------|
| 1. Faithfulness vs. contestability | RQ1 | A (+B) |
| 2. Active probing under fixed budget | RQ2 | A |
| 3. Socially sensitive interventions | RQ3 | C (+A irrelevant-cue cases) |
| 4. Identifiability and access limits | RQ4 | A (constructed indistinguishable worlds) |
| 5. Robustness and abstention | RQ5 | A (ambiguous + hard-negative cases) |
| 6. Generalization beyond Jev | all | A, B, C |

---

## Experiment 1: Faithfulness versus contestability

**Question:** How well do standard explanation-quality measures predict the discovery of legitimate grounds for review?

- Generate explanations using existing methods.
- Measure both (a) fidelity to the target model's observed behavior and (b) ability to identify independently verified decision defects.
- Investigate whether highly faithful explanations can still fail to identify contestable decisions.

**Protocol (Jev vs. explainer):**

1. **Query Jev and the explainer model independently.** Each model decides the same cases on its own, without seeing the other's answer.
2. **Freeze the explanation.** Give the explainer Jev's decision on the original case and have it explain that decision. Once written, the explanation is frozen and not revised.
3. **Evaluate the frozen explanation on a fixed set of hidden counterfactual variants.** The variants are fixed in advance and are not shown to the explainer while it writes the explanation. For each variant, record the decision the explanation implies.
4. **Separately collect Jev's and the explainer's independent decisions on those same variants.** These come from separate queries, not from the explanation.

**Measures:**

- **Explanation-implied agreement with Jev:** how often the decision implied by the frozen explanation matches Jev's actual decision on each hidden variant. This is the fidelity measure (a). The implied decision comes from a separate simulator model that reads the explanation and answers in a fixed structure (predicted decision, the rule it took from the explanation as an ordered list of rules that can be run against the record, and whether the explanation covers the variant). It is scored as the gain over the same simulator without the explanation. Explainers asked for structured explanations are also scored by running them directly against each variant's record. Details in [12_first_experiments_roadmap.md](12_first_experiments_roadmap.md#how-an-explanations-implied-decision-is-obtained).
- **Disagreement between Jev and the explainer:** how often the two models' independent decisions differ, on the original cases and on the variants. This shows whether the explainer is explaining a decision it would have made differently itself.

**Expected contribution:** A systematic separation of behavioral explanation quality from evidence usefulness.

## Experiment 2: Active probing under a fixed query budget

**Question:** Can CAPE find verified decision defects more efficiently than conventional black-box methods?

Evaluate different budgets of counterfactual queries and evidence checks.

**Baselines:**

- No-query natural-language explanations.
- Explanations with the original decision and probabilities.
- Random or uniform perturbation selection.
- LIME-style local attribution.
- Anchors and other model-agnostic explanation methods.
- Concept-level counterfactual testing inspired by *Walk the Talk?*.
- Active sampling based on model uncertainty or disagreement.
- A direct policy-checking baseline wherever sufficient structured policy information is available.

**The most important comparison** is not whether probing beats a blind explainer — it almost certainly should — but whether a **contestability-aware acquisition strategy outperforms strong, equally budgeted alternatives**.

## Experiment 3: Explanation faithfulness under socially sensitive interventions

**Question:** When a model is sensitive to identity-related or dialect-related cues, what do externally generated explanations disclose?

- Compare explainers with different kinds of access to the target decision model.
- Measure whether explanations mention reliably influential factors, attribute changes to unsupported factors, or express unjustified confidence.
- Focus: the difference between **accurately detecting a model's sensitivity** and **claiming that the sensitivity proves unfairness**.

## Experiment 4: Identifiability and access limitations

**Question:** Which grounds for contestation are identifiable under different levels of model and evidence access?

Three settings:

1. **Black-box only:** can query the decision model but cannot independently verify record accuracy.
2. **Black-box plus trusted records:** can obtain selected authoritative facts.
3. **Black-box plus policy evidence:** has access to applicable decision criteria or supporting documents.

- Construct **indistinguishable-world examples** demonstrating that certain decision defects cannot be established using model queries alone (e.g., two worlds with identical $f$ and $x$ where the true record differs — query access cannot separate them).
- Where possible, prove the corresponding information-theoretic limitation and characterize how additional trusted evidence changes identifiability.

## Experiment 5: Robustness and abstention

**Question:** Can the explanation system maintain reliable evidence standards under uncertainty?

Evaluate robustness to:

- Semantically invalid edits
- Model-output variability
- Limited budgets
- Changing decision thresholds
- Misleading supporting documents

The method should be permitted to conclude:

> "The current evidence is insufficient to determine whether this decision is erroneous. Additional review may be necessary."

This behavior should be **evaluated rather than automatically penalized** as an incomplete explanation.

## Experiment 6: Generalization beyond Jev

- **Jev** is the primary experimental model because its structured output interface provides a particularly clean setting.
- To establish broader relevance, also evaluate selected **open-weight or API-accessible classification models** and **LLM-based decision pipelines**.
- Determines whether observed failures result from Jev-specific behavior or general properties of black-box explanation systems.

---

## Shared protocol notes

- Fix query budgets and evidence-check budgets per condition; report results across a budget sweep.
- Repeat stochastic queries to estimate output variability; report stability.
- Report perturbation-quality-conditioned effects (Risk 3).
- Pre-register primary comparisons (CAPE vs. strongest equal-budget baseline) before running Experiment 2 at scale.
