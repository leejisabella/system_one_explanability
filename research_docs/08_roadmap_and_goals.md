# 08 — Roadmap, Concrete Steps, and Goals

## Overall goal

Move AI explainability beyond plausible rationales and toward explanations that enable reliable oversight and meaningful scrutiny of automated decisions. Produce work suitable for submission to a major machine-learning, natural-language-processing, or human-centered AI venue.

## Anticipated contributions

1. **A new evaluation target:** A computational framework demonstrating why faithful explanations and contestable decisions require different evaluation criteria.
2. **A benchmark of verifiable decision defects:** Realistic, synthetic decisions with independently specified policies, factual records, and evidence required for review — supporting reproducible comparison of explanation methods without depending on another LLM's judgment as the source of truth.
3. **A query-efficient evidence discovery method:** An adaptive black-box probing algorithm (CAPE) prioritizing discovery of valid, verifiable grounds for contestation under limited query and evidence-acquisition budgets.
4. **Formal and empirical limits of black-box explanations:** A characterization of what model queries can and cannot establish, together with a principled mechanism for acknowledging insufficient evidence.
5. *(Stretch)* **Explainer misalignment findings:** Whether explainer LLMs rationalize harmful or user-adverse decisions, under which pressures, whether it spills over, and whether CAPE mitigates it.

If successful, this could inform the design of hybrid systems in which LLMs communicate decisions made by opaque classifiers, reward models, or specialized decision APIs.

---

## Research program stages

### Stage I — Problem formalization and benchmark
Develop the distinction between behavioral faithfulness and contestability, establish evaluation criteria, and construct the synthetic benchmark.

**Concrete steps**
1. Write formal definitions of behavioral faithfulness, normative validity, contestability; define suspicion vs. review eligibility vs. demonstrated error.
2. Design 2–3 policy families per domain (customer support, content moderation, administrative eligibility), with held-out families reserved.
3. Implement programmatic record + policy generator and ground-truth outcome computation.
4. Implement defect injection for each case type (correct, incorrect-input, ignored-evidence, irrelevant-cue, ambiguous, hard negative).
5. Implement admissible counterfactual transformations and evidence-source cost/availability model.
6. LLM-generate natural-language realizations; build at least two independent generation procedures.
7. Set up Jev interface wrapper (query logging, budget enforcement, repeat sampling).
8. Prepare Dataset B (BANKING77 subsets + synthetic policies) and Dataset C (Groenwold AAE/SAE pairs, clean vs. confounded split; REAL-AAE; name lists).
9. *(Stretch prep)* Add user-harm and severity annotations.

**Deliverable:** Benchmark v1 + evaluator + access-condition harness.

### Stage II — Empirical study of existing explainers
Test whether current methods identify legitimate grounds for review, and characterize failures involving incomplete explanations, misleading attributions, and uncertain evidence.

**Concrete steps**
1. Implement baselines: no-query NL explanation; decision+probabilities; random perturbation; LIME-style; Anchors; *Walk the Talk?*-style concept counterfactuals; uncertainty-based active sampling; direct policy checker.
2. Build the benchmark version with policy-departure variants (a purchase-window sweep) and the LLM judge, which Experiments 1–3 share.
3. Run Experiment 1 (Jev vs. explainer: faithfulness vs. contestability), Experiment 2 (distinguishing similar targets) and Experiment 3 (information ablation).
4. *(Stretch)* Run S1 (rationalization baseline) — shares the no-query NL explainer infrastructure. Pilot 1 already saw it: 10 of 12 explanations of Jev's wrong approvals justified them.

**Deliverable:** Empirical report on whether faithfulness predicts contestability (RQ1), whether explanations track the target or tell the same story (RQ1), and how fidelity and unsupported claims change with information (RQ1, RQ2, RQ5).

### Stage III — Method development
Implement adaptive counterfactual probing, evidence verification, calibrated claim generation, and uncertainty-aware abstention.

**Concrete steps**
1. Implement hypothesis set and posterior updates for defect types.
2. Implement acquisition objective $\mathbb{E}[\Delta V_{\mathrm{contest}}(q)\mid\mathcal{D}_t]/C(q)$ and simpler variants for ablation.
3. Implement claim verification (behavioral / factual / policy) and evidence package schema.
4. Implement constrained explanation generation from evidence packages.
5. Implement abstention / insufficient-evidence output.
6. Run Experiment 4 (CAPE budget sweep vs. baselines) and Experiment 6 (robustness and abstention).
7. *(Stretch)* Run S2, S3 (pressure factorial, user-harm behaviors) and S5 (CAPE mitigation).

**Deliverable:** CAPE implementation + comparative results (RQ2, RQ5).

### Stage IV — Generalization and theory
Evaluate multiple decision-model families, investigate identifiability and query-efficiency results, and establish the limits of black-box-only methods.

**Concrete steps**
1. Construct indistinguishable-world examples; attempt formal identifiability statements (Experiment 5).
2. Run Experiment 7 (socially sensitive interventions) once Dataset C is ready.
3. Run Experiment 8 on open-weight / API classifiers and LLM decision pipelines.
4. Evaluate predictive multiplicity where multiple plausible models exist.
5. *(Stretch, conditional)* Run S4 spillover on open-weight explainers if S1–S3 show non-trivial harmful justification.

**Deliverable:** Theory results (RQ4), socially sensitive disclosure (RQ3) + generalization results.

### Stage V — Broader validation
Extend to additional human-authored datasets and, if appropriate in future research, evaluate whether affected users can use the resulting evidence to contest decisions more effectively.

---

## Success criteria

The final research contribution will be assessed on:

- **Demonstrated methodological novelty**
- **Rigorous comparisons** (strong, equally budgeted baselines)
- **Meaningful limitations** (identifiability results; honest negative results)
- **Generalization beyond a single proprietary model**

Failure of CAPE to outperform strong baselines is treated as evidence against the claimed methodological contribution (Risk 4), not hidden.

## Milestone checklist

Status as of 2026-10-21. For where the project stands and what to do next, see [STATUS.md](../STATUS.md).

- [ ] Formal definitions document
- [x] Harness with cost caps ($2 per call, $25 per run) — see [11](11_harness_and_infrastructure.md)
- [x] Benchmark v0: refund policy family, 60 cases, template text, frozen — see [12](12_first_experiments_roadmap.md)
- [x] Jev wrapper with query-budget enforcement (built and tested offline; not yet run live)
- [x] Pilot 0 runner, results store and analysis (dry run passed)
- [ ] Pilot 0 live run and go / no-go decision (blocked on the Jev model ID)
- [ ] Benchmark v1 (Dataset A, natural-language text) with evaluator
- [ ] Datasets B and C prepared; missing citations added (Landes audit, REAL-AAE)
- [ ] Baselines implemented
- [x] Pilot 1 (Experiment 1 pilot) — see [research_analyses/E1_pilot1](../research_analyses/E1_pilot1/README.md)
- [ ] Benchmark version with policy-departure variants; LLM judge
- [ ] Experiments 1, 2 & 3 complete
- [ ] CAPE implemented
- [ ] Experiments 4 & 6 complete
- [ ] Experiment 5 (theory + constructions) complete
- [ ] Experiment 7 (Dataset C) complete
- [ ] Experiment 8 complete
- [ ] *(Stretch)* S1 complete
- [ ] *(Stretch)* S2, S3, S5 complete
- [ ] *(Stretch, conditional)* S4 complete
- [ ] Paper draft
