# 02 — Related Work and Research Gap

The project connects three research areas that have largely developed separate evaluation objectives. A fourth area (§2.4) supports the stretch goal. Full citations are in [10_references.md](10_references.md).

## 2.1 Explanation faithfulness

- **Walk the Talk?** *Measuring the Faithfulness of Large Language Model Explanations* (Matton, Ness, Guttag & Kıcıman, ICLR 2025) defines explanation faithfulness through concept-level interventions and uses a Bayesian hierarchical model to estimate feature effects. It shows that LLM-generated explanations can hide influential features, including social bias.
- **Faithful Serum** (Alon, Zimerman & Wolf, ACL 2026) uses attribution information from a model's internal representations to improve textual explanation faithfulness.
- **A Positive Case for Faithfulness** (Mayne et al., ICML 2026) introduces **Normalized Simulatability Gain (NSG)**, measuring whether explanations help observers predict a model's behavior on related examples.

These studies provide important definitions and methods, but **accurate prediction of model behavior is not equivalent to identifying valid grounds for overturning a decision.**

## 2.2 Algorithmic contestability

- **Explainable AI Isn't Enough! Rethinking Algorithmic Contestability** (Freiesleben, Meding & König, AAAI/ACM AIES 2026) distinguishes contestability from algorithmic recourse and identifies evidence involving incorrect feature values, neglected overriding evidence, and predictive multiplicity. (These map directly onto our evidence Types I–III in [03_framework_CAPE.md](03_framework_CAPE.md).)
- **Counterfactual Explanations and the Scope of Contestability** (Huang & Grote, 2026) analyzes the limitations of counterfactual explanations and proposes limited multi-query interaction as a promising approach.

These studies motivate contestability as an objective, but leave opportunities for algorithmic development, benchmark construction, and comparative evaluation of evidence discovery.

## 2.3 Black-box explanation and model auditing

- Existing methods such as **LIME** (Ribeiro et al., 2016), **SHAP** (Lundberg & Lee, 2017), **Anchors** (Ribeiro et al., 2018), and black-box **counterfactual generation** (e.g., Wachter et al., 2017) estimate influential factors or locate changes that alter model predictions.
- Recent work also develops efficient, rigorous model-agnostic explanation algorithms (Marques-Silva, Lefebre-Lobaina & Martinez, IJCAI 2025).
- Dialect auditing: Wu et al. (2026) audit and mitigate LLMs silently correcting African American English via activation steering; Groenwold et al. (2020) study AAVE in transformer-based generation.

Our research differs by targeting **verified contestation evidence under limited information and query budgets**, rather than feature importance, response prediction, or decision reversal alone.

## 2.4 (Stretch) Unfaithful rationalization, sycophancy, and misalignment

Relevant to whether explainer LLMs can be pushed to justify bad or user-harmful decisions:

- **Unfaithful rationalization:** Turpin et al. (2023) show chain-of-thought explanations can systematically rationalize answers driven by biasing features without mentioning them. Lanham et al. (2023) measure CoT faithfulness.
- **Sycophancy:** Sharma et al. (2023) show RLHF-trained assistants tend to agree with user/authority beliefs over truth. Perez et al. (2022) use model-written evaluations to surface such behaviors.
- **Generalization of narrow behavior into broad misalignment:** Betley et al. (2025), *Emergent Misalignment*, show that narrow fine-tuning (e.g., on insecure code) can produce broadly misaligned behavior — directly motivating SQ4 (does training/conditioning on "justify the decision" spill over?).
- **Strategic or pressured behavior under deployment framing:** Greenblatt et al. (2024), *Alignment Faking in Large Language Models*; Lynch et al. (2025), *Agentic Misalignment* — models behave differently under goal conflicts and deployer framings.
- **Safety-training failure modes:** Wei et al. (2023), *Jailbroken*, characterize competing objectives and mismatched generalization — relevant to the conflict between "be helpful to the deployer by explaining" and "don't harm the user."

*Note:* §2.4 references were added for the stretch goal and must be verified by the authors (see [10_references.md](10_references.md)).

## 2.5 Proposed research gap

We aim to investigate whether evidence relevant to contestation can be discovered more efficiently and reliably than through conventional explanation methods.

The intended contributions are **not** a new definition of contestability or another demonstration of explanation unfaithfulness. Instead, we propose:

1. An evaluation framework that separates behavioral faithfulness, normative validity, and contestability.
2. A benchmark with independently verifiable decision defects and evidence requirements.
3. A query-efficient method for discovering and validating grounds for review.
4. A characterization of what cannot be established from black-box model behavior alone.

**Stretch-goal gap:** Prior work on unfaithful CoT and sycophancy studies models explaining *their own* answers or agreeing with users. Less studied is the **third-party explainer role**: an LLM asked to explain *another* system's fixed, possibly harmful decision to the affected user, with deployer incentives pulling toward justification. Our benchmark's independent ground truth makes it possible to measure harmful justification objectively.
