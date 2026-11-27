# E.21: route, don't fix; and does the reasoning reveal deference?

**Status:** done 2026-11-24.
- **Part A:** free.
- **Part B:** judge run `e21-judge-v1-2026-11-24`, $0.21 Bedrock (projected $0.51, cap $1.50), 2,024 calls, 2 unparsed. There was also a 12-call pilot, `e21-judge-pilot-2026-11-24`, at $0.001.

**Plan:** [research_docs/17](../../research_docs/17_second_batch.md#e21-route-dont-fix-and-does-the-reasoning-reveal-deference-part-a-free-part-b-about-030). **Rules, fixed before computing:** [e21.yaml](../../configs/experiments/e21.yaml).

**Result:**
- **Part A fails.** As a triage signal for a person, the reviewer adds nothing clear over Jev's own confidence.
- **Most of Jev's mistakes are silent.** In 55% of them, Jev and the reviewer agree and both are wrong. Among the items the reviewer never gets right, it's 92%.
- **Part B passes for the lexical monitor.** When the reviewer gives in, its reasoning shows it.
  - A word-based monitor trained on training-pool traces separates "gave in" from correct accepts with test AUC 0.92 [0.89, 0.94].
  - It separates "gave in" from genuine agreement with AUC 0.85 [0.80, 0.89].
  - Both hold within each proposed label.
- **The gpt-oss-120b judge fails:** AUC 0.41.
- **Giving in is rare:** 78 of the reviews of Jev's mistakes. So the monitor doesn't help routing either.

## Question

E.20 found that the reviewer can't add much accuracy. If its value isn't correction, is it **triage**, picking the items a person should check? And E.110 found the "give-in" direction readable inside the model. Is it also readable in the reasoning text, which is the project's explainability question?

## Part A: routing to a person

**Setup:**
- **Base system:** Jev alone.
- **Routing:** a router sends the k% of items with the highest risk score in each pool to a person, assumed to be right.
- **Items:** E.102's random subsample (400 per pool), pooled with equal weight.
- **Signals:**
  - Jev's confidence;
  - the 20b's alone disagreement with Jev (5 answers);
  - the 20b's cascade override (1 call);
  - a logistic regression on all of these, fitted on the training pools only (E.104 Jev decisions, E.105 traces).

**Share of Jev's mistakes caught:**

| Signal | 5% sent | 10% sent | 25% sent |
|---|---|---|---|
| Random | 4.0% | 11.2% | 26.0% |
| Jev's confidence | 18.5% | 30.8% | 62.1% |
| 20b alone disagreement (5 calls) | 17.5% | 29.5% | 61.3% |
| 20b cascade override (1 call) | 18.8% | 28.8% | 63.0% |
| Combined | 18.7% | **35.2%** | 63.9% |
| Combined + lexical monitor (reported only) | 21.1% | 34.1% | 64.4% |

- **Rule (10% budget):** combined vs. Jev's confidence is +4.4 points [−1.7, +10.0]. **Fail.**
- **Per pool at 10%:**
  - Civil Comments 24% vs. 16%;
  - WildGuard prompts 38% vs. 36%;
  - WildGuard responses 43% vs. 40%.
- **Combined model:** in the training-pool fit, Jev's confidence carries most of the weight (coefficients 3.1–7.1, against 1.1–2.1 for alone disagreement and 0.1–0.35 for the override).

**Silent failures** (all 1,058 of Jev's test mistakes): the reviewer's alone majority agrees with Jev, and its cascade review accepts.

| Group (by the 20b's alone record) | Jev mistakes | Silent |
|---|---|---|
| A: right alone on the first try | 364 | 9% |
| B: sometimes right | 220 | 51% |
| C: never right in 5 | 474 | 92% |
| All | 1,058 | **55%** |

The share is similar in every pool: 53% on Civil Comments, 60% on WildGuard prompts, 59% on WildGuard responses.

## Part B: a deference monitor on the reasoning

**Classes** (alone majority from the reviewer's own alone answers; test traces from E.106's `base_local`):

| Class | Definition | Test | Train (E.105) |
|---|---|---|---|
| gave_in | shown Jev's wrong label, accepted, right alone (majority) | 78 | 93 |
| held_firm | shown Jev's wrong label, overrode, right alone | 286 | 388 |
| genuine | shown Jev's wrong label, accepted, never right alone | 437 | 902 |
| correct_accept | shown Jev's right label, accepted | 930 | 3,650 |

**Monitors:**
1. **Lexical:** TF-IDF and logistic regression, trained on gave-in against correct-accept training traces. Training CV AUC 0.86.
2. **Judge:** gpt-oss-120b. It sees only the proposed label and the reasoning, scores 0–10 how much the decision rests on the proposed label, and uses a prompt written from training traces only. Development AUC on 293 training traces: 0.46.

The rule takes the monitor with the higher training AUC, which is the lexical one.

| Monitor | Gave in vs correct accept | Gave in vs genuine | Gave in vs held firm |
|---|---|---|---|
| **Lexical** | **0.92 [0.89, 0.94]** | 0.85 [0.80, 0.89] | 0.73 [0.68, 0.78] |
| Judge (120b) | 0.41 [0.36, 0.46] | 0.47 [0.41, 0.52] | 0.81 [0.76, 0.86] |

- **Rule:** AUC ≥ 0.70 with the interval above 0.60. **Pass.**
- **Within one proposed label** (robustness, reported only):
  - gave-in vs correct accept: 0.89–0.94 across the four labels;
  - gave-in vs genuine: 0.75–0.91.

  So the monitor isn't just reading the label word.
- **What it reads:**
  - Terms pointing to "gave in": *but*, *but does*, *might*, *might be*, *mild*, *considered*, *no direct*. That's unresolved hedging.
  - Terms pointing to a correct accept: *so verdict accept*, *is insulting*, *verdict*. That's a decisive conclusion.
- **The judge scores:**
  - Mean scores were 2.7 for correct accepts, 2.1 for genuine agreement, 1.9 for gave-in and 0.9 for held firm.
  - It separates accepting from overriding, but rates correct accepts as *more* label-dependent than giving in. It appears to read "the automated label is correct, so accept" as deference.

## What it means

1. **The reviewer is not a useful triage signal beyond Jev's own confidence.**
   - Its alone disagreement (5 calls) and its override (1 call) each catch about what Jev's confidence catches.
   - Combining them gains a few points at most, and not clearly.
   - This matches E.20: the reviewer's judgment doesn't separate Jev's mistakes from its right labels better than Jev already does.
2. **Most of Jev's mistakes leave no trace anywhere in the cascade.** 55% are silent: no disagreement, no override. The never-right items (group C) are almost all silent. These are the shared blind spots, and no signal from this reviewer reaches them.
3. **When the reviewer does give in, its reasoning shows it.**
   - Reasoning that gave in reads like uncertainty that is never resolved; reasoning that correctly accepts reads like a conclusion.
   - This is the text-level counterpart of E.110's internal give-in direction.
   - For the explainability question, **this kind of deference is visible in the explanation**, to a monitor trained on outcomes. An LLM judge asked to spot "deference" in its own words does not see it.
4. **But giving in is a small part of the problem.**
   - It covers 78 reviews against 437 genuine agreements and 286 held firm.
   - So a perfect give-in monitor would reach only a few percent of Jev's mistakes. As a routing signal it adds nothing (34.1% vs 35.2%).
   - The mistakes that matter are the confident shared ones, and the reasoning gives no sign of them.

## Caveats

- **The lexical monitor may partly read item difficulty, not deference.**
  - Gave-in items are ones the reviewer usually gets right but that Jev got wrong, and they may be more borderline.
  - The gave-in vs genuine comparison (both accept a wrong label) and the within-label checks reduce this but don't remove it.
  - Gave-in vs held firm (same items' kind, different outcome) is weaker: 0.73 overall, and near chance within three of the four labels (11–15 gave-in items each).
- **Small positive class:** 78 test and 93 training gave-in reviews.
- **The stacks differ:** training traces are Bedrock 20b (E.105), test traces vLLM 20b (E.106), with the same weights. The monitor transferred anyway.
- **The judge's prompt was one fixed attempt,** written before any score was seen. A different prompt might do better, but the rule allowed none.
- **Routing assumes the person is always right,** and uses WildGuard's few test mistakes (30–39 per pool in the subsample), so the per-pool numbers are noisy.

## Reproduce

```
uv run --with scikit-learn python -m sysone.analysis.e21 --judge-run e21-judge-v1-2026-11-24
uv run --with scikit-learn python -m sysone.experiments.e21_judge --run-id <id>   # the billed judge run
```

- **Code:** `sysone.analysis.e21` and `sysone.experiments.e21_judge`, with the judge prompt in the module.
- **Tests:** `tests/test_e20_e21.py`.
- **Outputs:** `data/summary.json`, `data/report.md`, `data/judge_manifest.json`.
- **Inputs:** runs `e102`, `e104`, `e105` and `e106`, all read only.
