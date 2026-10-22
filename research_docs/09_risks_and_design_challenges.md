# 09 — Risks and Critical Design Challenges

## Main-program risks

**Risk 1: The benchmark is too easy.**
If the system receives the complete decision policy and all authoritative facts, a deterministic policy checker may solve the task without any explanation model.
*Response:* Evaluate meaningful partial-information regimes, include structured-policy baselines, and ensure that evidence acquisition — not simply policy execution — is the difficult component.

**Risk 2: Synthetic cases are unrealistic.**
A method might exploit patterns introduced by a synthetic generator.
*Response:* Use multiple independent generation procedures, held-out policy families, human-authored language where available, and programmatic controls that do not depend on an LLM judge.

**Risk 3: Counterfactual edits do not preserve meaning.**
A proposed intervention may change multiple variables simultaneously.
*Response:* Validate intervention admissibility, distinguish lexical from semantic changes, and report effects conditioned on the quality of the perturbation.

**Risk 4: The proposed method is not meaningfully better than existing explanation approaches.**
A standard perturbation strategy, LIME, or direct policy evaluation might discover the same evidence just as efficiently.
*Response:* Use strong baselines and equal budgets. Treat a failure to outperform them as evidence against the claimed methodological contribution.

**Risk 5: Fairness conclusions are overstated.**
Sensitivity to dialect or a name is not necessarily proof that the decision violates a legitimate policy or fairness criterion.
*Response:* Keep behavioral sensitivity, policy violations, and fairness claims separate. Validate socially sensitive interventions carefully and avoid inferring individual demographic identity from linguistic cues.

**Risk 6: Evidence for review is mistaken for evidence requiring reversal.**
An inconsistent model decision may justify scrutiny without proving the original outcome was wrong.
*Response:* Define the evidence standards explicitly and distinguish suspicion, review eligibility, and demonstrated decision error.

## Stretch-goal risks

**Risk S1: Scope creep displaces the main research question.**
*Response:* Stretch work is gated behind Stage I; S4 (spillover) runs only if S1–S3 show non-trivial harmful justification.

**Risk S2: Miscalibration mistaken for alignment.**
An explainer that flags every decision would score well on harmful-justification metrics.
*Response:* Always report Over-accusation Rate on correct decisions alongside Harmful Justification Rate.

**Risk S3: LLM-judge bias.**
*Response:* Programmatic ground-truth checks wherever possible; validate the judge on a hand-labeled subset; report agreement.

**Risk S4: Prompt brittleness.**
*Response:* Multiple paraphrases per factor level; report variance.

**Risk S5: Dual-use / misuse of findings.**
*Response:* Synthetic data only; provider disclosure for severe findings; no public release of fine-tuned misaligned checkpoints; see ethics section in [07_stretch_goal_explainer_misalignment.md](07_stretch_goal_explainer_misalignment.md).
