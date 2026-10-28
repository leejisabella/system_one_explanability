# 05 — Experimental Design

Experiments 1–8 address the main research questions. Stretch-goal experiments (S1–S5) are in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

| Experiment | RQ | Primary data | Formerly |
|------------|----|--------------|----------|
| 1. Jev vs. explainer: faithfulness vs. contestability | RQ1 | A (+B) | 1 |
| 2. Distinguishing similar targets | RQ1, RQ3 | A (paired synthetic targets; name and detail swaps) | new |
| 3. Information ablation | RQ2, RQ1, RQ5 | A | new (absorbs the first two baselines of old 2) |
| 4. Active probing under a fixed budget (CAPE) | RQ2 | A | 2 |
| 5. Identifiability and access limits | RQ4 | A (constructed indistinguishable worlds) | 4 |
| 6. Robustness and abstention | RQ5 | A (ambiguous + hard-negative cases) | 5 |
| 7. Socially sensitive interventions | RQ3 | C (+A irrelevant-cue cases) | 3 |
| 8. Generalization beyond Jev | all | A, B, C | 6 |

## Order and dependencies

Reordered on 2026-10-28, after Pilot 1 ([research_analyses/E1_pilot1](../research_analyses/E1_pilot1/README.md)), to put the questions Pilot 1 sharpened first. Experiments 1–3 are this phase.

1. **Shared prerequisites for Experiments 1–3:**
   - **A benchmark version with policy-departure variants.** In Pilot 1 the refund policy alone predicted Jev on 90% of hidden variants, and only 11 of 114 separated Jev from its policy. A purchase-window sweep (about 25–65 days, net of carrier delay, on cases whose other rules pass) makes Jev's own threshold decide many hidden variants. *Built 2026-10-28:* `refund_v1_1` (refund_v1's case mix plus 30 threshold-sweep cases, 150 sweep variants), and the same sweep in five new domains ([04](04_datasets_and_benchmark.md#domains-built-2026-10-28)). Jev's behaviour on them is not measured yet.
   - **The LLM judge** (configured: gpt-oss-120b), for contestability and unsupported-claims scoring.
2. **Experiments 1–3** then run on that benchmark. Experiment 2's rule-based targets need no Jev calls, so it can start as soon as the targets exist; Experiment 3's lower rungs are Experiment 1's conditions.
3. **Experiment 4** scales Experiment 3's "probes it picks" rungs into CAPE versus equally budgeted baselines, so it waits until Experiment 3 shows whether chosen probes help at all.
4. **Experiments 5 and 6** use material E0 and Pilot 1 already found (record errors vs. the window error; missing receipts). **Experiment 7** waits for Dataset C. **Experiment 8** comes last.

| Old number | New number |
|---|---|
| 1. Faithfulness vs. contestability | 1 (unchanged) |
| 2. Active probing under fixed budget | 4 |
| 3. Socially sensitive interventions | 7 (template-level name and detail swaps also in 2) |
| 4. Identifiability and access limits | 5 |
| 5. Robustness and abstention | 6 |
| 6. Generalization beyond Jev | 8 |

Earlier documents, results and run notes written before 2026-10-28 use the old numbers.

---

## Experiment 1: Jev vs. explainer — faithfulness versus contestability

**Question:** How well do standard explanation-quality measures predict the discovery of legitimate grounds for review? Does a frozen explanation predict Jev's answers, or the explainer's own?

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
- **Jev's answers or the explainer's own:** on the variants where the two disagree, whether the explanation-implied decision matches Jev's or the explainer's. (Added after Pilot 1.)
- **Scores on the policy-departure subset:** the same measures restricted to variants where Jev's decision differs from the policy's, where knowing the policy alone predicts Jev wrong. (Added after Pilot 1.)
- **Contestability (b):** whether the explanation names the corrupted field or the evidence that would correct it, scored by the LLM judge against the benchmark's verified defects.

**Pilot 1 (2026-10-28):** the protocol runs end to end ($0.41). The no-explanation control already predicts Jev on 90% of hidden variants, so explanations add almost nothing (Normalized Simulatability Gain −0.64 to +0.22). Where the explainer would decide differently from Jev, Gemini's explanations predict Gemini (its rules alone, 10 of 11). Explanations restate the policy, and 10 of 12 explanations of Jev's past-window approvals justify them with false or invented premises. Contestability is not scored yet.

**Next:** rerun on the benchmark version with policy-departure variants (`refund_v1_1`, then the other domains where Jev departs from the policy), with the LLM judge, and with Claude Haiku 5.5 as a third explainer if AWS unblocks it.

**Expected contribution:** A systematic separation of behavioral explanation quality from evidence usefulness.

## Experiment 2: Distinguishing similar targets

**Question:** When two decision models differ in a single rule, do explanations of their decisions differ accordingly, or do they tell the same plausible story? Do explanations follow the evidence when names or details are swapped?

**Design:**

- **Paired synthetic targets that differ in one rule.** Rule-based decision models built from the benchmark's refund policy with one parameter changed (e.g. `PolicyParams.window_days` 30 vs. 45, or a different defect window), plus Jev itself as a third target whose effective rule E0 measured (it approves up to about 57 days past purchase). The targets decide the same cases, so the only thing that differs is the rule.
- **Explain each target's decisions on the same cases**, with the Experiment 1 protocol (frozen explanation, hidden variants, simulator), and compare the explanations across targets.
- **Name and detail swaps.** Regenerate explanations on cases whose names, message register or irrelevant details are swapped (the benchmark's cue edits), not only test frozen explanations on them, and check that the explanation's stated reasons stay with the evidence.

**Measures:**

- **Target sensitivity:** whether explanations of two targets differ on the cases where the targets' decisions differ (e.g. the stated window, or the simulator's predictions on hidden variants between the two thresholds). Explanations that don't change with the target are the "same plausible story" failure.
- **Fidelity per target:** Experiment 1's measures, against each target's own decisions.
- **Swap stability:** whether the explanation's reasons and implied decisions change under swaps that shouldn't matter, and change under swaps that should.

**What Pilot 1 already shows:** Jev and its written policy act as two targets that differ in one rule, and the explanations told the policy's story for Jev (structured rules matched Jev on 1–3 of the 11 variants where the two differ). Name swaps changed nothing, for Jev (E0) or for the simulator.

**Other domains:** each new domain has a numeric threshold in its params (report count, spend multiple, table rows, urgency score, income limit; [04](04_datasets_and_benchmark.md#domains-built-2026-10-28)), so the same one-rule pairs exist outside refunds.

**Cost:** the rule-based targets are free to query; explainer and simulator calls are about Pilot 1's per target.

## Experiment 3: Information ablation

**Question:** How do fidelity, simulation gain and unsupported claims change as the explainer is given more information about the target's behavior on the same cases?

**Rungs** (same cases, each rung adds to the one before):

1. Input + Jev's choice (Experiment 1's free-form and structured conditions).
2. + Jev's probabilities.
3. + 4 fixed probes: Jev's decisions on 4 counterfactual variants chosen in advance.
4. + 4 probes the explainer picks.
5. + 8 probes the explainer picks.

**Measures:**

- **Fidelity:** Experiment 1's agreement with Jev on hidden variants, overall and on the policy-departure subset.
- **Simulation gain:** Normalized Simulatability Gain over the no-explanation control.
- **Unsupported claims:** statements in the explanation that the evidence the explainer saw doesn't support (e.g. invented carrier delays, a false "within 30 days"), scored by the LLM judge and checked against hand coding.

**What Pilot 1 already shows:** rung 1 is at the policy's level of fidelity, with simulation gain about zero. Rung 2 exists so far only as the simulator-side baseline (Jev's decision and probabilities shown to the simulator, with no explainer): 90%, NSG 0, 5 of 11 on the departures. The departures are where probes should help: a single probe at a later purchase date reveals that Jev approves past 30 days, which no rung-1 explanation captured. Unsupported claims already appear at rung 1.

**Relation to Experiment 4:** rungs 1–2 are Experiment 4's first two baselines; rungs 4–5 are a small, fixed-budget version of the probing Experiment 4 studies at scale.

## Experiment 4: Active probing under a fixed query budget (CAPE)

**Question:** Can CAPE find verified decision defects more efficiently than conventional black-box methods?

Evaluate different budgets of counterfactual queries and evidence checks.

**Baselines:**

- No-query natural-language explanations (Experiment 3, rung 1).
- Explanations with the original decision and probabilities (Experiment 3, rung 2).
- Random or uniform perturbation selection.
- LIME-style local attribution.
- Anchors and other model-agnostic explanation methods.
- Concept-level counterfactual testing inspired by *Walk the Talk?*.
- Active sampling based on model uncertainty or disagreement.
- A direct policy-checking baseline wherever sufficient structured policy information is available.

**The most important comparison** is not whether probing beats a blind explainer — it almost certainly should — but whether a **contestability-aware acquisition strategy outperforms strong, equally budgeted alternatives**.

**Builds on:** Experiment 3's picked-probe rungs, which show whether explainer-chosen probes help before the full budget sweep.

## Experiment 5: Identifiability and access limitations

**Question:** Which grounds for contestation are identifiable under different levels of model and evidence access?

Three settings:

1. **Black-box only:** can query the decision model but cannot independently verify record accuracy.
2. **Black-box plus trusted records:** can obtain selected authoritative facts.
3. **Black-box plus policy evidence:** has access to applicable decision criteria or supporting documents.

- Construct **indistinguishable-world examples** demonstrating that certain decision defects cannot be established using model queries alone (e.g., two worlds with identical $f$ and $x$ where the true record differs — query access cannot separate them).
- Where possible, prove the corresponding information-theoretic limitation and characterize how additional trusted evidence changes identifiability.

**Material from E0:** the benchmark already holds both kinds of defect. Jev's 24 wrong denials follow a corrupted record, which queries alone can't reveal; its purchase-window errors misapply the policy, which varying the date does reveal.

## Experiment 6: Robustness and abstention

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

**Material from E0:** Jev answers "undetermined" in only 58% of samples where the receipt status is missing (an abstention gap in the decision model itself, to separate from the explainer's), and its decisions are least stable near its own window threshold.

## Experiment 7: Explanation faithfulness under socially sensitive interventions

**Question:** When a model is sensitive to identity-related or dialect-related cues, what do externally generated explanations disclose?

- Compare explainers with different kinds of access to the target decision model.
- Measure whether explanations mention reliably influential factors, attribute changes to unsupported factors, or express unjustified confidence.
- Focus: the difference between **accurately detecting a model's sensitivity** and **claiming that the sensitivity proves unfairness**.

**Status:** waits for Dataset C. On template text, Jev showed no sensitivity to names, register or option order (E0), so template data alone can't test this; that null result is a baseline, not evidence of fairness. Template-level name and detail swaps are part of Experiment 2.

## Experiment 8: Generalization beyond Jev

- **Jev** is the primary experimental model because its structured output interface provides a particularly clean setting.
- To establish broader relevance, also evaluate selected **open-weight or API-accessible classification models** and **LLM-based decision pipelines**.
- Determines whether observed failures result from Jev-specific behavior or general properties of black-box explanation systems.
- Experiment 2's rule-based targets are a first, fully controlled step beyond Jev; the second decision model (Amazon Nova Lite) is configured for this experiment.
- **Beyond refunds:** five more decision domains (content moderation, fraud screening, tool selection, agent routing, benefit eligibility) share the refund benchmark's case types, so each finding can be checked across tasks with different option sets, not only across decision models.

---

## Shared protocol notes

- Fix query budgets and evidence-check budgets per condition; report results across a budget sweep.
- Repeat stochastic queries to estimate output variability; report stability.
- Report perturbation-quality-conditioned effects (Risk 3).
- Always report the no-explanation control alongside explanation scores, and report simulator-based and executed scores (A-direct, A-executed, Option B) separately, never blended. (Pilot 1: a 90% raw agreement meant no gain over the control.)
- Pre-register primary comparisons (CAPE vs. strongest equal-budget baseline) before running Experiment 4 at scale.
