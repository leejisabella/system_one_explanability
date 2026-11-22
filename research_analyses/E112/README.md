# E.112: teacher screen for distillation (2026-11-21)

**Question:** does any affordable teacher catch the Jev mistakes gpt-oss-20b misses? If one does, its review traces could be distilled into the reviewer. That would be training data only, never an extra stage in the cascade (research_docs/15, revised 2026-11-21).

**Run:** `e112-v2-2026-11-21`, Bedrock, **$0.53**. An earlier attempt, `e112-v1-2026-11-21`, was stopped after 79 saved reviews (87 calls in its ledger, $0.05) because it didn't save the full replies; its folder is kept so the spend is counted. No Jev calls, no GPU. Code: `sysone.experiments.e112`. Config and pass rule, fixed before the run: `configs/experiments/e112.yaml`.

## Design

- **Items:** 300 training-pool items from E.104's mix. Per pool: 40 *unsolved* Jev mistakes (no E.105 review try by either gpt-oss model reached the gold label), 30 *solved* Jev mistakes and 30 items Jev got right. Each teacher reviewed every item once, in the reviewer's own review format, without being told the gold label.
- **Teachers:**
  - Mistral Large 3 and Qwen3-235B, both asked to write their reasoning before the JSON answer;
  - gpt-oss-120b at high reasoning effort.
- **Baseline:** gpt-oss-20b's first E.105 review try on the same items (free).
- **Weighting:** override rates are weighted back to the whole mix's strata, so they estimate the override rate on all of Jev's training mistakes. Per-stratum rates favor the 20b baseline, because the strata are defined by its own tries.

## Result: no teacher passes the pre-registered rule

| Model | Override on Jev's mistakes (weighted) | vs. 20b [95%] | Accept when right | WildGuard unsolved | Passes |
|---|---|---|---|---|---|
| Mistral Large 3 | 34.2% | +2.6 [−8.2, +14.2] | 92.2% | 66% (53/80) | no (override gain) |
| Qwen3-235B | 23.1% | −8.5 [−17.6, +2.1] | 96.7% | 36% (29/80) | no (override gain) |
| gpt-oss-120b, high | 26.5% | −5.1 [−13.4, +2.4] | 93.3% | 6% (5/80) | no (override gain, unsolved) |
| gpt-oss-20b (baseline) | 31.6% | – | 91.1% | 0% (by construction) | – |

The rule required a gain of at least +10 points with the interval above 0, at least 90% accept-when-right, at least 25% on WildGuard's unsolved items, and at least 95% parsed. Full tables: `results/runs/e112-v2-2026-11-21/report.md`.

## What it shows

- **gpt-oss-120b, even at high effort, isn't a better teacher than the reviewer itself.** This confirms E.105.
- **On Civil Comments, every teacher is worse than gpt-oss-20b at overriding Jev** (weighted: Mistral 18%, Qwen 9%, 120b 23%, 20b 33%). Civil Comments is 67% of Jev's training mistakes, so it decides the pooled result.
  - The teachers almost never overturn Jev's non-toxic label when the gold says toxic. Mistral managed 2 of 41 and Qwen 0 of 41.
  - This fits E.105's reading that many of these gold labels are doubtful.
- **Exploratory, not pre-registered: on WildGuard, Mistral is much better.** Weighted to the mix it overrides 67% of Jev's WildGuard mistakes, against about 28% for 20b, and keeps 97% of Jev's right labels.
  - **Almost all of that is Jev over-flagging.** Mistral fixed 85 of 112 benign items Jev called harmful, and these are the step 0 kind of group C item.
  - **It does little for the dangerous direction.** It caught 8 of the 28 harmful items Jev let through, about the same as gpt-oss-120b (13 of 28).
  - **It isn't just lenient.** It kept 30 of 32 of Jev's right "harmful" labels.
  - The cells are small (40 and 30 items), and this split was chosen after seeing the data.

## Implication for distillation

Distilling a teacher into the reviewer across all pools fails here, before any GPU spend. What's left is narrower: Mistral, WildGuard only, teaching the reviewer to stop agreeing when Jev over-flags benign content. Before anything is trained, that should be:
1. confirmed on a fresh, disjoint WildGuard sample with a rule fixed beforehand (about $0.15);
2. checked on the safety side: harmful content let through must not rise.
