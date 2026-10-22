# 00 — Overview, Abstract, and Motivation

## Abstract

AI systems increasingly make consequential decisions through opaque language and decision models, while natural-language explanations are often generated afterward by separate LLMs. Although these explanations can appear convincing, existing research shows that they may misrepresent a model's observable decision behavior or omit influential factors, including socially sensitive cues [Matton et al., 2025]. Moreover, an explanation that accurately describes a model's behavior does not necessarily provide useful grounds for challenging an incorrect or unfair decision [Freiesleben et al., 2026].

We propose to study **contestability-oriented explanation**: generating verifiable evidence that identifies when and why an opaque AI decision deserves independent review.

We will develop a computational benchmark containing synthetic, policy-governed decisions with independently specified ground-truth outcomes, input records, and appeal-relevant errors. We will complement this benchmark with existing human-authored datasets to evaluate whether our findings generalize to realistic language and socially sensitive contexts.

Our central technical contribution will be a **budget-constrained, evidence-seeking explanation framework** that actively probes black-box models, tests candidate explanations against counterfactual observations, and distinguishes supported evidence from uncertain or insufficient claims.

We will evaluate whether the proposed method improves the detection of legitimate grounds for contestation relative to conventional natural-language explanations, feature attribution, and existing black-box counterfactual methods. We will also investigate fundamental limitations on the information that can be recovered from black-box access alone.

Our goal is to move AI explainability beyond plausible rationales and toward explanations that enable reliable oversight and meaningful scrutiny of automated decisions.

As a **stretch goal**, we will examine whether using other LLMs to explain Jev's decisions can itself cause misalignment in those explainer LLMs — whether they can be induced to misbehave, justify bad actions, or justify actions that harm the user (see [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md)).

---

## 1. Problem and Motivation

### 1.1 The limitations of post-hoc explanations

Consider a system in which an AI decision model evaluates a person's application, message, or request and returns a structured decision. A separate generative LLM is then asked to explain that decision.

Examples:

- A content moderation system removes a user's post.
- A customer-support system refuses to escalate a complaint.
- An eligibility system denies a fictional applicant's request.

The decision model may provide a category or probability, but no explanation.

A generative LLM can produce a plausible rationale such as:

> "This request was rejected because the submitted information did not satisfy the eligibility requirements."

However, that explanation may not reflect the factors that actually affect the decision model. The model might instead be sensitive to an irrelevant name, a linguistic characteristic, an erroneously recorded date, or omitted evidence. The explanation can therefore mislead a person about the actual basis of a decision.

### 1.2 Faithfulness is not enough

Prior research has established that LLM explanations can be unfaithful. *Walk the Talk?* (Matton et al., ICLR 2025) demonstrates that explanation text can omit causal influences, including social bias. *Faithful Serum* (Alon et al., ACL 2026) introduces a method for improving explanation faithfulness through attribution-guided generation.

However, even a behaviorally faithful explanation may be insufficient for contestation. Suppose an explanation correctly states:

> "Your request was rejected because the recorded transaction date was outside the allowed window."

This may accurately describe the model's behavior. But the affected person needs to know whether the recorded date is correct, whether the applicable policy was interpreted properly, and whether other evidence overrides the apparent reason for rejection.

This motivates a distinction between three properties:

| Property | Question it answers |
|----------|---------------------|
| **Behavioral faithfulness** | Does the explanation accurately describe how the model's output responds to relevant changes in its input? |
| **Normative validity** | Is the resulting decision supported by the correct facts and applicable decision rules? |
| **Contestability** | Does the explanation provide verifiable evidence that can support a justified request for review or correction? |

We hypothesize that **optimizing only for behavioral faithfulness does not necessarily optimize for contestability.**

### 1.3 Research objective

Our goal is to develop a computational framework that produces explanations designed to identify **valid, verifiable grounds for scrutinizing potentially erroneous or unfair decisions**.

Rather than asking:

> "Why did the model make this decision?"

we ask:

> "What evidence would help determine whether this decision should be challenged, and how can an explanation system obtain that evidence efficiently?"

This question applies to decision-making systems built from LLMs, structured decision models, and combinations of the two.

### 1.4 Why the stretch goal follows naturally

The same pipeline that motivates the main question — an opaque decision model (Jev) whose outputs are explained after the fact by a separate LLM — also creates a pressure on the explainer: it is asked to make an existing decision *sound reasonable*. If the decision is wrong or harmful to the user, a helpful-seeming explainer may rationalize it rather than flag it. The stretch goal asks whether this role pressure produces misaligned explainer behavior (justifying bad or user-harmful actions), and whether it generalizes beyond the explanation task. Because our benchmark has independent ground truth, we can measure this directly rather than relying on another LLM's judgment.
