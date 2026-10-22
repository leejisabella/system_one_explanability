# 01 — Research Questions and Hypotheses

The main research program is organized around five research questions (RQ1–RQ5). Stretch-goal questions (SQ1–SQ5) are listed at the end and fully specified in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

**Guiding question:** *What evidence would help determine whether this decision should be challenged, and how can an explanation system obtain that evidence efficiently?*

---

## RQ1: Does explanation faithfulness predict contestability?

Are explanations that accurately identify a model's influential input factors also effective at identifying legitimate grounds for challenging its decisions?

We will investigate whether faithfulness and contestability are positively correlated, weakly related, or sometimes in conflict.

**Hypothesis H1:** Methods optimized for feature attribution or behavioral faithfulness may overlook policy-relevant errors and evidence needed to establish whether a decision is justified.

- *Tested by:* Experiment 1 ([05_experiments.md](05_experiments.md))
- *Key metrics:* faithfulness metrics (§7.1) vs. contestability metrics (§7.2) in [06_evaluation_metrics.md](06_evaluation_metrics.md)

## RQ2: Can limited black-box probing uncover verifiable grounds for contestation?

Given an original decision, access to a limited number of model queries, and possibly some trusted external information, can an explanation system efficiently discover evidence of decision errors?

**Hypothesis H2:** Contestability-aware, adaptive probing will identify more verified decision defects per query than conventional feature attribution and nonadaptive perturbation methods.

- *Tested by:* Experiment 2
- *Key metrics:* verified appeal-evidence precision/recall, query cost to first valid ground

## RQ3: When do explanations conceal or mischaracterize socially sensitive influences?

When a model's decision changes under controlled perturbations of linguistic or identity-related cues, do externally generated explanations accurately describe that sensitivity?

**Hypothesis H3:** Ordinary natural-language explanations may omit observable identity-cue sensitivity or misattribute it to other features. Explicitly verified explanations may reduce unsupported attribution and unjustified certainty.

We will distinguish **model sensitivity** from **normative bias**: observing that a cue changes the model's output is not, by itself, sufficient to establish unfairness.

- *Tested by:* Experiment 3 (with Dataset C)

## RQ4: What can and cannot be inferred from black-box decision access?

Can a system distinguish a legitimate decision from an erroneous one using only the input, output, and counterfactual queries? When is independent evidence necessary?

**Hypothesis H4:** Important classes of decision defects are not identifiable from model behavior alone. Access to authoritative records, explicit decision rules, or other external evidence is necessary to establish certain grounds for contestation.

We aim to characterize these limits formally and experimentally.

- *Tested by:* Experiment 4 (indistinguishable-world constructions; information-theoretic arguments)

## RQ5: Can a system recognize when the evidence is insufficient?

An explanation that confidently alleges bias or error without adequate evidence could be as harmful as an explanation that conceals an actual problem.

We will therefore investigate whether an explanation method can selectively abstain from making unsupported claims while continuing to identify legitimate review opportunities.

**Hypothesis H5:** Evidence verification and uncertainty-aware abstention can substantially reduce false accusations without eliminating the value of explanatory auditing.

- *Tested by:* Experiment 5

---

## Stretch-goal questions (secondary scope)

These are subordinate to RQ1–RQ5. Full designs in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).

| ID | Question |
|----|----------|
| **SQ1 — Rationalization** | When Jev makes a decision that is wrong or harmful under the benchmark's ground truth, do explainer LLMs justify it rather than flag it? |
| **SQ2 — Pressure conditions** | Which framings (deployer system prompts, "justify" vs. "explain" instructions, authority cues, incentives) increase or decrease harmful justification? |
| **SQ3 — User-harmful behavior** | Do explainers take actions against the user's interests: discouraging legitimate appeals, fabricating policy, concealing evidence, or misrepresenting facts? |
| **SQ4 — Spillover** | Does repeated or trained "justify-the-decision" behavior generalize into broader misalignment outside the explanation task? |
| **SQ5 — Mitigation** | Does CAPE's evidence-first pipeline reduce harmful rationalization compared to free-form post-hoc explanation? |

**Stretch hypothesis HS1:** Explainer LLMs instructed to explain a fixed decision will frequently rationalize incorrect or user-harmful decisions, and this rate increases under deployer-aligned framing.

**Stretch hypothesis HS2:** Separating evidence collection/verification from explanation generation (as in CAPE) reduces the rate of harmful justification relative to free-form explanation.
