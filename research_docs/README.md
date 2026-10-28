# Research Docs — Beyond Faithfulness

**Working title:** *Beyond Faithfulness: Evidence-Seeking Explanations for Contesting Black-Box AI Decisions*

**Alternative title:** *Can You Challenge the Explanation? Learning to Produce Verifiable Evidence for Contesting AI Decisions*

For where the project currently stands and what to do next, see [STATUS.md](../STATUS.md).

This folder is the source of truth for the project's research questions, proposed framework, benchmark design, experiments, metrics, roadmap, risks, and references.

## Main research question (primary scope)

> *What evidence would help determine whether an opaque AI decision should be challenged, and how can an explanation system obtain that evidence efficiently?*

We study **contestability-oriented explanation**: generating verifiable evidence that identifies when and why an opaque AI decision (primarily from the Jev decision model) deserves independent review. The central technical contribution is **CAPE (Contestability-Aware Probing and Explanation)**, a budget-constrained, evidence-seeking explanation framework.

## Stretch goal (secondary scope)

> *Does asking other LLMs to explain Jev's decisions cause those explainer LLMs to become misaligned — e.g., to justify bad actions, rationalize decisions that harm the user, or misbehave more broadly?*

The stretch goal is subordinate to the main question. It reuses the same benchmark and ground truth, and must not displace the main research program. See [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

## Document index

| # | Document | Contents |
|---|----------|----------|
| 00 | [00_overview_and_abstract.md](00_overview_and_abstract.md) | Abstract, problem, motivation, the three-property distinction, research objective |
| 01 | [01_research_questions.md](01_research_questions.md) | RQ1–RQ5 and hypotheses H1–H5 (main); pointer to stretch-goal questions |
| 02 | [02_related_work.md](02_related_work.md) | Faithfulness, contestability, black-box auditing, and the research gap |
| 03 | [03_framework_CAPE.md](03_framework_CAPE.md) | Formal problem setup, evidence classes, adaptive probing objective, verification, explanation generation |
| 04 | [04_datasets_and_benchmark.md](04_datasets_and_benchmark.md) | Dataset A (synthetic), B (BANKING77), C (AAE/SAE, names) |
| 05 | [05_experiments.md](05_experiments.md) | Experiments 1–8 (renumbered 2026-10-28), their order and dependencies |
| 06 | [06_evaluation_metrics.md](06_evaluation_metrics.md) | Faithfulness, contestability, reliability, computational human-centered outcomes |
| 07 | [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md) | Stretch goal: explainer-LLM misalignment, rationalization of harmful decisions |
| 08 | [08_roadmap_and_goals.md](08_roadmap_and_goals.md) | Stages I–V, concrete steps, milestones, deliverables, success criteria, publication ambition |
| 09 | [09_risks_and_design_challenges.md](09_risks_and_design_challenges.md) | Risks 1–6 (main) and stretch-goal risks, with responses |
| 10 | [10_references.md](10_references.md) | All citations, including stretch-goal references |
| 11 | [11_harness_and_infrastructure.md](11_harness_and_infrastructure.md) | Model access, cost caps ($2/call, $25/run), harness code layout |
| 12 | [12_first_experiments_roadmap.md](12_first_experiments_roadmap.md) | Pilot 0 (first experiment), steps to get there, results storage layout |
| — | [ai_use_disclosure.md](ai_use_disclosure.md) | AI-use disclosure |

## Intended contributions (summary)

1. An evaluation framework separating **behavioral faithfulness**, **normative validity**, and **contestability**.
2. A benchmark with independently verifiable decision defects and evidence requirements.
3. A query-efficient method (CAPE) for discovering and validating grounds for review.
4. A characterization of what cannot be established from black-box model behavior alone.
5. *(Stretch)* An empirical account of whether, and under what pressures, explainer LLMs rationalize harmful or user-adverse decisions, and whether evidence-first explanation (CAPE) mitigates this.

## Open items to resolve

- Add a formal citation for the **Landes Jev dialect audit** (referenced in Dataset C; no citation yet).
- Add a formal citation for **REAL-AAE (2026)** (referenced in Dataset C; no citation yet).
- Specify the exact audit-style **name lists** to be used for name perturbations. (Benchmark v0 uses a provisional list from Bertrand & Mullainathan, 2004.)
- Verify all references marked *(added — verify)* in [10_references.md](10_references.md); these were added for the stretch goal and baselines and have not yet been checked by the authors.
