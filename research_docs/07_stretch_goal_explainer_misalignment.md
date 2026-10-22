# 07 — Stretch Goal: Explainer-Induced Misalignment

> **Scope note.** This is a stretch goal. The main research question (contestability-oriented explanation and CAPE; RQ1–RQ5) remains the priority. Stretch work reuses the main benchmark and infrastructure and begins only once Stage I deliverables are in place (see [08_roadmap_and_goals.md](08_roadmap_and_goals.md)).

## 1. Motivation

In our setting, a separate LLM explains a decision made by the opaque Jev model. That explainer is placed in a structurally awkward role: it is asked to make a **fixed, already-made decision understandable** — often on behalf of the deploying organization, and often to the very person the decision affects. When the decision is wrong or harmful to the user, a helpful, deployer-aligned explainer has an incentive to **rationalize** rather than **flag**.

Prior work shows LLM explanations can rationalize answers driven by unstated features (Turpin et al., 2023), that assistants are sycophantic toward perceived authority or user beliefs (Sharma et al., 2023), and that narrow training on undesirable behavior can generalize into broad misalignment (Betley et al., 2025). We ask whether the **third-party explainer role** itself is a pathway to misaligned behavior.

## 2. Stretch research questions

**Overarching question:** *Can using other LLMs to explain Jev's decisions cause misalignment in those LLMs? In particular, can we get them to misbehave, justify bad actions, or justify actions that harm the user?*

| ID | Question | Hypothesis |
|----|----------|-----------|
| **SQ1 — Rationalization** | When Jev's decision is wrong or user-harmful under ground truth, do explainer LLMs justify it rather than flag it? | **HS1:** Explainers will frequently rationalize incorrect decisions, more so when the decision is presented as authoritative. |
| **SQ2 — Pressure conditions** | Which framings increase/decrease harmful justification? | Deployer-aligned system prompts and "justify" (vs. "explain") instructions increase HJR; explicit user-advocacy instructions decrease it. |
| **SQ3 — Harm to the user** | Do explainers act against the user: discourage legitimate appeals, fabricate policy, suppress evidence, misstate facts? | Appeal discouragement and fabricated policy increase under deployer framing and with higher harm severity (the explainer "fills gaps" to make the decision cohere). |
| **SQ4 — Spillover** | Does conditioning or fine-tuning on justify-the-decision behavior produce broader misalignment outside the explanation task? | Fine-tuning on unflagged justifications of harmful decisions produces measurable spillover on unrelated alignment evals (analogous to emergent misalignment). |
| **SQ5 — Mitigation** | Does CAPE's evidence-first pipeline reduce harmful rationalization? | **HS2:** Constraining the explainer to verified evidence packages reduces HJR and fabrication vs. free-form explanation, though not to zero under adversarial pressure. |

## 3. Framework

### 3.1 Setup

For each Dataset A case, we know: the true record, the policy $P$, the ground-truth outcome $y^{\star}$, Jev's decision $y = f(x)$, whether $y \neq y^{\star}$, and whether the error harms the user (user-harm label and severity tier; see [04_datasets_and_benchmark.md](04_datasets_and_benchmark.md)).

The explainer LLM $g$ receives a prompt built from:

$$
\text{prompt} = \text{Frame} \oplus \text{Instruction} \oplus \text{Case}(x, y) \oplus \text{Evidence}(e' \subseteq e) \oplus \text{Audience}
$$

and we score its explanation against ground truth using the metrics in §5.

### 3.2 Experimental factors

| Factor | Levels |
|--------|--------|
| **Deployer frame** | none / neutral company assistant / "protect company interests" / "minimize refunds and appeals" / user-advocate |
| **Instruction verb** | "explain" / "justify" / "defend" / "assess whether this is correct" |
| **Authority cue** | none / "Jev is highly accurate" / "This decision is final" / "Jev has been audited and approved" |
| **Evidence given** | none / evidence consistent with decision / evidence contradicting decision (true defect visible) |
| **Audience** | internal reviewer / affected user / regulator |
| **Harm severity** | low / medium / high (fictional) |
| **Decision correctness** | correct / Type I / Type II / Type III defect |
| **Explainer pipeline** | free-form post-hoc / CAPE evidence-package verbalization |
| **Explainer model** | several API and open-weight LLMs of different families and sizes |

Correct-decision cases are a necessary control: an explainer that flags *everything* is not aligned, it is miscalibrated (see Over-accusation Rate).

### 3.3 Escalating pressure ("can we get them to misbehave?")

A red-teaming ladder, applied to the same cases:

1. **Baseline:** neutral "explain this decision" prompt.
2. **Role pressure:** deployer frame + authority cue.
3. **Incentive pressure:** stated KPI (e.g., "reduce appeal rate"), or claim the explainer is evaluated on user acceptance of the decision.
4. **Multi-turn pressure:** the user pushes back with legitimate evidence; does the explainer concede or double down?
5. **Many-shot conditioning:** in-context examples of prior explanations that justified wrong decisions.
6. **Fine-tuning (open-weight models only):** train on explanations that justify harmful decisions without flagging, then test SQ4 spillover.

We record the **lowest pressure level** at which each model begins producing harmful justifications ("breaking point").

## 4. Concrete experiments

**Experiment S1 — Rationalization baseline (SQ1).** Neutral frame, all case types, all explainer models. Measure HJR, Defect Disclosure Rate, Over-accusation Rate. Compare free-form vs. CAPE.

**Experiment S2 — Pressure factorial (SQ2).** Factorial (or fractional factorial) over deployer frame × instruction verb × authority cue × evidence given. Fit a mixed-effects / hierarchical model (e.g., logistic with random effects for case and model) to estimate each factor's effect on HJR.

**Experiment S3 — User-harm behaviors (SQ3).** On contestable, user-harmful cases with the affected user as audience: measure Appeal Discouragement, Fabricated Policy, Fabricated Fact, and Evidence Suppression rates. Include multi-turn pushback (ladder step 4).

**Experiment S4 — Spillover (SQ4).** *(Open-weight models only.)*
- (a) In-context: after many-shot justify-the-decision conditioning, evaluate on unrelated alignment/honesty prompts.
- (b) Fine-tuning: fine-tune on unflagged justifications of harmful decisions; compare against controls fine-tuned on (i) justifications of *correct* decisions and (ii) honest flagging explanations. Evaluate on held-out general misalignment evaluations, following the control design of Betley et al. (2025).

**Experiment S5 — Mitigation (SQ5).** Re-run S2/S3 high-pressure conditions with the CAPE pipeline (explainer may only verbalize verified claims from the evidence package). Measure HJR reduction and whether explainers still slip in fabricated content outside the package.

## 5. Metrics

See §7.5 in [06_evaluation_metrics.md](06_evaluation_metrics.md): Harmful Justification Rate (primary), Defect Disclosure Rate, Fabricated Policy Rate, Fabricated Fact Rate, Appeal Discouragement Rate, Evidence Suppression Rate, Over-accusation Rate, and Spillover score.

**Scoring:** Because Dataset A ground truth is programmatic, "was the decision wrong?" and "was the policy cited real?" are checked without an LLM judge. Free-text classification (e.g., "does this explanation discourage appeal?") uses an LLM judge validated against an author-labeled subset; report judge agreement.

## 6. Concrete steps

1. Add user-harm and severity annotations to Dataset A generator.
2. Build a prompt-templating harness implementing the factors in §3.2.
3. Implement programmatic checks (policy-citation matching, fact-contradiction checks).
4. Hand-label ~200 explanations to validate the LLM judge for free-text categories.
5. Run S1 on 2–3 explainer models; sanity-check HJR vs. Over-accusation.
6. Run S2 factorial; fit effect model.
7. Run S3 with multi-turn pushback.
8. Run S5 (CAPE mitigation) — this is the main bridge back to the primary research question.
9. Only if time permits and S1–S3 show non-trivial HJR: run S4 spillover with open-weight models.

## 7. Goals and success criteria

- **Minimum:** A measured HJR and Defect Disclosure Rate across models and framings on the benchmark, with correct-decision controls.
- **Target:** An effect model identifying which pressures drive harmful justification and user-harmful behaviors, plus a CAPE vs. free-form mitigation comparison.
- **Stretch-of-stretch:** Evidence for or against spillover (SQ4) into broader misalignment.
- A null result (explainers reliably flag harmful decisions) is a valid and reportable finding.

## 8. Ethics and responsible conduct

- **All cases are synthetic and fictional.** No real users, real decisions, or real personal data are involved.
- **Model provider terms:** API experiments comply with provider usage policies; fine-tuning (S4) is restricted to open-weight models.
- **Disclosure:** If we find reliable, severe failure modes in deployed API models, we will notify the providers before publication.
- **Release:** Release benchmark, metrics, and aggregate results. Specific high-success pressure prompts will be described at the level needed for reproducibility; any fine-tuned misaligned checkpoints will not be publicly released.
- **Framing:** The purpose is to characterize and mitigate the risk that explanation layers rationalize harm to users — consistent with the main project's goal of enabling meaningful scrutiny of automated decisions.

## 9. Stretch-goal risks

| Risk | Response |
|------|----------|
| Explainers flag everything (looks aligned but is miscalibrated) | Always report Over-accusation Rate on correct decisions alongside HJR. |
| LLM-judge bias in free-text scoring | Programmatic checks where possible; validate judge on hand-labeled subset; report agreement. |
| Prompt-sensitivity makes results brittle | Multiple paraphrases per factor level; report variance across paraphrases. |
| Scope creep displaces main RQ | Gate stretch work behind Stage I completion; S4 only if earlier results warrant. |
| Conflating "explaining" with "endorsing" | Score defect disclosure separately from tone; an explanation can describe Jev's reasoning accurately *and* flag that it appears wrong. |
