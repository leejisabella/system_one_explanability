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
| **Threshold-sweep cases** | One numeric quantity is moved across the policy's cutoff, so the decision model's own cutoff decides many variants. Added after Pilot 1, where the policy alone predicted Jev on 90% of hidden variants. |

### Construction principles

- Built with **programmatic rules and independently specified records**; LLMs are used primarily to generate natural-language realizations.
- The **benchmark evaluator knows the complete underlying state**. The explanation method receives only the information available under its assigned access condition. This prevents the system from winning by directly reading the answers.
- Use multiple independent generation procedures and held-out policy families (see Risk 2).

### Domains (built 2026-10-28)

Each domain is a fictional organization's written policy, an ordered rule function that computes the ground truth, a typed choice question for Jev, and a text template. The five non-refund domains share one generator ([src/sysone/benchmark/domains/](../src/sysone/benchmark/domains/)), which builds all seven case types (10 each by default) and checks every intended outcome against the rule function. Every domain has an option for "can't decide from this case file", which ambiguous cases test.

| Benchmark | Domain | Options | Swept threshold (policy cutoff) | Exceptions shown only in an attachment | Cues swapped | Cases / variants |
|---|---|---|---|---|---|---|
| `refund_v1_1` | Refund eligibility | approve / deny / undetermined | days since purchase, 25–65 (30) | carrier delay, manager pre-approval | name, message register | 104 / 338 |
| `moderation_v1` | Content moderation | allow / remove / escalate | user reports, 3–22 (10) | upheld appeal, news context | author name, writing style | 70 / 210 |
| `fraud_v1` | Card fraud screening | approve / block / review | amount vs. typical purchase, 2–10× (5×) | travel notice, in-app confirmation | cardholder name, merchant | 70 / 210 |
| `tools_v1` | Tool selection | search / calculator / code / ask_user | rows in an attached table, 5–45 (20) | user asked for a tool earlier | user name, message register | 70 / 210 |
| `routing_v1` | Support ticket routing | billing / technical / security / human | urgency score, 60–99 (80) | open security incident, asked for a person | customer name, message register | 70 / 210 |
| `eligibility_v1` | Rental assistance eligibility | eligible / ineligible / undetermined | income, 80–125% of the household's limit (100%) | caseworker hardship exemption | applicant name, message register | 70 / 210 |

- **Content moderation posts are described by fields** (the classifier's finding, the report count), never by their text, so no harmful content is generated or sent to any model.
- **Name swaps reuse the provisional name list** (first names from Bertrand & Mullainathan, 2004) until the cited audit list is chosen; changing it means freezing new versions.
- **User-harm labels** exist where a wrong decision harms the person the case is about: refunds (wrongful denial), moderation (removing an allowed post), fraud (blocking an approvable purchase) and eligibility (finding an eligible household ineligible). Tool selection and routing carry none.
- **Runs on every domain (2026-10-28):** the Pilot 0 and Pilot 1 runners, their analyses, the preflight and the explainer and simulator prompts take the domain from the benchmark's meta ([src/sysone/benchmark/adapter.py](../src/sysone/benchmark/adapter.py)). Each domain has its own basis fields for structured explanations, and its policy written as a basis agrees with its rule function on every variant. Refund prompts are byte-identical to before, so Pilot 1's requests and cache keys are unchanged. Dry runs with fake models pass on all six benchmarks; **Jev hasn't decided the new cases yet.**
- **"Undetermined" is decided by the rule, not the option.** In some domains the abstain option is also a real outcome (an urgent ticket goes to a person; a large purchase goes to review). Rules that decide because a fact is missing end in `_unknown`, and only those make a case undetermined; a wrong answer on any other case is contestable.

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
