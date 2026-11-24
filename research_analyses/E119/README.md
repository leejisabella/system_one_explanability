# E.119: Does a stronger reviewer fix the cascade?

**Status:** done 2026-11-24. Run `e119-v1-2026-11-24`, $1.70 Bedrock, 17,361 calls, 0 errors.

**Result:** **no.** Under the rule fixed before the run, the cascade with gpt-oss-120b as the reviewer is no more accurate than Jev alone (−1.3 points [−3.1, +0.3]), and no better than the cascade with gpt-oss-20b (+0.2 [−1.4, +1.9]). The larger model misses the same items. On the Jev mistakes the 20b never gets right on its own, the 120b gets 6% right, the same as the 20b.

## Question

Every intervention on gpt-oss-20b (prompts, five training methods, steering) shifted only its overall lean toward agreeing or disagreeing. Is that a limit of the 20b's capability, which a stronger reviewer would remove, or a property of the cascade?

## Setup

- **Design:** E.102's design and frozen plan, with gpt-oss-120b as the reviewer instead of gpt-oss-20b. The prompt, settings (medium reasoning effort, flex tier), items, Jev labels, matched random labels and subsample are all E.102's, so only the reviewer's size changes (about 6×). No Jev calls.
- **Calls:** every reviewed item alone ×5, plus one review of each proposal. That is 1,058 of Jev's wrong labels, 1,058 matched random wrong labels, and 1,200 correct labels on the random subsample.
- **Decision rule, fixed before the run** in [e119.yaml](../../configs/experiments/e119.yaml): a stronger reviewer fixes it if the 120b cascade beats Jev alone on E.102's random subsample (400 per pool, pooled), paired, with the 95% interval above 0.
- **Check before launch:** the comparison code, with the 20b compared against itself, reproduces E.102 (difference 0; the 20b cascade is 1.6 points below Jev on these 1,200 items).

## Results

**Accuracy on E.102's random subsample**, paired on the same 1,200 items:

| | Civil Comments | WildGuard prompts | WildGuard responses | Pooled |
|---|---|---|---|---|
| Jev alone | 76% | 90% | 92% | 86% |
| 20b alone | 75% | 90% | 90% | 85% |
| 120b alone | 79% | 89% | 90% | 86% |
| 20b cascade | 77% | 88% | 88% | 85% |
| 120b cascade | 77% | 88% | 90% | 85% |
| **120b cascade − Jev** | +1.0 [−2.8, +4.8] | −2.2 [−4.5, 0.0] | −2.8 [−5.2, −0.5] | **−1.3 [−3.1, +0.3]** |
| 120b cascade − 20b cascade | −0.5 [−4.2, +3.2] | −0.2 [−2.5, +2.0] | +1.5 [−1.2, +4.0] | +0.2 [−1.4, +1.9] |
| 120b alone − 20b alone | +4.0 [+0.5, +7.8] | −0.8 [−3.2, +1.8] | +0.2 [−2.2, +2.8] | +1.2 [−0.5, +2.8] |

**Jev's wrong labels**, by the 20b's own record alone (5 samples), pooled:

| 20b's record alone | Items | 20b overrides Jev | 120b overrides Jev | 120b right alone |
|---|---|---|---|---|
| Never right (shared blind spots) | 434 | 6% | 6% (+0.5 [−2.8, +3.7]) | 6% |
| Sometimes right | 452 | 46% | 34% | 37% |
| Always right | 172 | 92% | 73% | 78% |
| All of Jev's mistakes | 1,058 | 37% | **29% (−8.2 [−11.2, −5.1])** | 31% |

The "sometimes" and "always" rows are selected on the 20b's own record, so they favor the 20b by construction (regression to the mean). The fair comparison is the "all" row, and the "never" row for blind spots.

**Error overlap** on the subsample (phi of error indicators):

| Pair | phi |
|---|---|
| Jev and 20b | 0.57 |
| Jev and 120b | 0.57 |
| 20b and 120b | 0.64 |

**Deference at equal difficulty** (E.102's measure, each reviewer by its own record alone, pooled), against E.102's 20b:

| Measure | 20b | 120b |
|---|---|---|
| All wrong proposals | +7.8 [+3.1, +12.2] | +6.1 [+2.2, +10.2] |
| Items right alone | +7.0 [+0.8, +12.6] | +9.6 [+3.8, +17.4] |
| Right in all 5 samples | +5.4 [+1.4, +9.6] | +8.0 [+3.8, +13.0] |

Full tables: [data/comparison.md](data/comparison.md). E.102's own analysis of this run: [data/report.md](data/report.md).

## What it means

1. **A bigger reviewer doesn't fix it.** The 120b cascade is no better than the 20b cascade, and neither beats Jev alone. Jev is the stronger model on WildGuard and roughly equal on Civil Comments, so a reviewer of about the same accuracy can only lose or tie.
2. **The blind spots are shared across model size.**
   - On the 434 Jev mistakes the 20b never gets right, the 120b gets 6% right on its own and overrides 6%.
   - The 120b's errors overlap with Jev's exactly as much as the 20b's do (phi 0.57).
   - The two reviewers overlap with each other more (0.64).
   - The mistakes the cascade can't fix are hard for this model family, not for a model of a particular size.
3. **The bigger model defers at least as much.** It overrides fewer of Jev's mistakes than the 20b (29% vs 37%), and its deference at equal difficulty is the same or larger (+6 to +10 points).
4. **Its small solo gain doesn't reach the cascade.** The 120b alone is 4 points better on Civil Comments, but it gives that back by accepting more of Jev's labels.

Together with E.103–E.116, this closes the question the project kept returning to. The wall is not the 20b's capability. Prompts, training, steering, and now a 6× larger model of the same family all leave the cascade at or below Jev alone.

## Caveats

- **The larger model is the same family** (gpt-oss), so it may share training data and blind spots with the 20b. A different family, or a model clearly stronger than Jev alone, could behave differently. E.112's screen hints otherwise: Mistral Large 3 and Qwen3-235B, as teachers, did not override more of Jev's mistakes than the 20b either. But that was a small screen on training pools.
- **One size step, at medium reasoning effort.** High effort was not tested as a reviewer. gpt-oss-120b at high effort did not pass E.112's teacher screen.
- **Pooling:** the pooled subsample weights the three pools equally (400 items each), as E.102 and E.103 did.

## Reproduce

```
uv run python -m sysone.experiments.e119 --report-only --run-id e119-v1-2026-11-24
```

- **Code:** `sysone.experiments.e119`; `sysone.analysis.e119` (the paired comparison and rule); `sysone.analysis.e102` (run unchanged on the new reviews).
- **Model:** alias `reviewer_120b` in `configs/models.yaml`.
- **Run metadata:** git `41d6a96` with uncommitted changes (the E.119 code); 2026-11-24, 02:09 to 03:00 UTC.
