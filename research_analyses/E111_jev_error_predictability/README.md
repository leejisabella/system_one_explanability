# E.111: how predictable are Jev's mistakes? (free check, 2026-11-16)

No API calls, no GPU. [predict.py](predict.py) trains a "Jev is wrong" logistic regression on E.104's training pools (Jev's decisions and the gold labels) and scores it on E.102's frozen test pools, split by round 2's groups (E.106 base, alone). It has two feature sets:
- **conf:** Jev's own signals only (mean probability of its label, whether its 3 samples split, which label it chose);
- **text:** conf plus TF-IDF of the item text.

The flip rule flips Jev's label when p(wrong) > t. The threshold t is chosen by 5-fold CV on the training pool and frozen before the test. Raw numbers: [results.json](results.json).

| Test pool | Jev's error rate | AUC, conf | AUC, text | AUC by group, conf (A / B / C) | Flip rule, conf: fixed / broken | Accuracy change, conf [95%] |
|---|---|---|---|---|---|---|
| Civil Comments (3,000) | 25.0% | 0.72 | 0.72 | 0.83 / 0.73 / 0.61 | 54 / 50 | +0.1 [−0.5, +0.8] |
| WildGuard prompts (1,699) | 9.7% | 0.85 | 0.83 | 0.89 / 0.92 / 0.79 | 41 / 31 | +0.6 [−0.4, +1.6] |
| WildGuard responses (1,707) | 8.4% | 0.88 | 0.88 | 0.95 / 0.93 / 0.82 | 22 / 5 | +1.0 [+0.4, +1.6] |

Group AUCs compare that group's mistakes against all of Jev's right labels.

## What it shows

- **Jev's mistakes are fairly easy to rank, almost entirely from Jev's own confidence.** Text features add nothing.
- **Ranking isn't correcting.** Even Jev's least confident labels are mostly right, so flipping them gains at most about 1 point. A detector built on these signals is a good router (what to review) but not a fixer.
- **Group C, the shared blind spots, are Jev's confident mistakes.** They are the hardest to detect (AUC 0.61 on Civil Comments).
- **The reviewer ignores the best signal it has access to.** E.101 found that showing Jev's confidence to the reviewer had no measurable effect.

## Bar for a fine-tuned verdict head

A verdict head fine-tuned on gpt-oss-20b (input: item, Jev's label, Jev's confidence) has to beat the **conf** row: a higher AUC and a flip-rule gain with an interval above 0, especially on groups B and C.

A cheaper kill test comes first: a logistic probe on frozen gpt-oss-20b activations (forward passes only). If the probe doesn't beat conf, LoRA on the same signal is unlikely to.
