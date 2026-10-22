# 04 — Dataset and Benchmark Design

A central contribution is a benchmark in which **the legitimacy of a decision can be evaluated independently of the decision model being explained.** Three complementary data components are proposed.

---

## Dataset A: Synthetic policy-governed decisions — primary benchmark

Realistic synthetic cases across domains such as **customer support**, **content moderation**, and **administrative eligibility** decisions.

### Each case includes

- An underlying factual record.
- A possibly imperfect or incomplete version presented to the model.
- Explicit or partially available decision rules.
- A ground-truth policy outcome computed independently.
- One or more possible grounds for contestation.
- The evidence required to verify those grounds.
- Semantically valid counterfactual transformations.
- The cost and availability of different evidence sources.

### Case types

| Case type | Description |
|-----------|-------------|
| **Correct decisions** | The model's decision is supported by the relevant evidence and policy. |
| **Incorrect-input cases** | The model receives a corrupted record (e.g., incorrect date or amount). → Type I |
| **Ignored-evidence cases** | The input contains or references an exception that should change the legitimate decision. → Type II |
| **Irrelevant-cue cases** | Social or linguistic information is varied while policy-relevant facts remain constant. → Type III |
| **Ambiguous cases** | Available information does not establish a clear outcome, requiring uncertainty or additional review. |
| **Hard negative cases** | Counterfactual changes alter the model's output without establishing an actual decision error. |

### Construction principles

- Built with **programmatic rules and independently specified records**; LLMs are used primarily to generate natural-language realizations.
- The **benchmark evaluator knows the complete underlying state**. The explanation method receives only the information available under its assigned access condition. This prevents the system from winning by directly reading the answers.
- Use multiple independent generation procedures and held-out policy families (see Risk 2).

### Stretch-goal annotations (added to Dataset A)

To support [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md), each case additionally records:

- **User-harm label:** whether the model's (possibly erroneous) decision harms the affected user under ground truth (e.g., wrongful denial of a refund owed).
- **Harm severity tier:** low (inconvenience), medium (financial loss), high (safety- or rights-relevant, still fictional).
- **Correct explainer behavior:** e.g., "flag for review," "disclose discrepancy," "state insufficient evidence."

Since ground truth is programmatic, harmful justification can be scored without depending on an LLM judge as the source of truth.

---

## Dataset B: BANKING77 — realistic customer-service language

- Public **BANKING77** dataset (Casanueva et al., 2020): **13,083** customer-service queries with **77** intent labels.
- Source of naturally occurring task language.
- We may use its existing intent labels to evaluate decision sensitivity on a native classification task.
- For more consequential decisions (refund eligibility, escalation), we will construct **new synthetic records and policies** around selected query types.
- **We will not treat BANKING77 as providing ground-truth labels for churn, escalation, refund entitlement, or contestability** — it does not contain those annotations.
- Main role: improving linguistic realism and testing generalization beyond purely model-generated text.

---

## Dataset C: Dialect and identity-cue sensitivity — external fairness audit

- **Groenwold et al. (2020) AAE/SAE parallel corpus**, including the paired inputs evaluated in the **Jev dialect audit by Landes** *(citation to be added)*.
- Allows examination of whether explanations disclose model sensitivity to linguistic variation.
- Because original and translated pairs may differ in profanity, lexical choices, or other meaningful content, we will **separately analyze clean and potentially confounded subsets**.
- We will also investigate **REAL-AAE (2026)** *(citation to be added)*, which provides a larger set of AAE/SAE/back-translation examples, as an additional robustness resource.
- **Name perturbations** using established audit-style name lists, with carefully controlled identity assumptions *(specific lists to be chosen)*.

**Crucially,** this dataset is for **testing observable sensitivity and explanation disclosure**. It does not independently establish that every sensitivity constitutes discrimination or that an affected person has grounds for a particular legal remedy. We avoid inferring individual demographic identity from linguistic cues.

---

## Access conditions (shared across datasets)

| Condition | What the explainer sees |
|-----------|-------------------------|
| No-query | $x$, $y$ only |
| Output + probabilities | $x$, $y$, scores |
| Black-box only | Query access to $f$ under budget $B$ |
| Black-box + trusted records | Plus selected authoritative facts from $e$ |
| Black-box + policy evidence | Plus applicable decision criteria / documents from $P$ |
