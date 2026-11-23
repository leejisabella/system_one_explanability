# E.114: send only Jev's low-confidence labels to the reviewer (2026-11-21)

**Status:** done 2026-11-21. The plan below was written before any number was computed; the results follow it.

**Question:** E.106's cascade sends every item to the reviewer (gpt-oss-20b) and ends 3.5 points *below* Jev alone (82.7% against 86.2% on the 1,200-item subsample), because the reviewer breaks more of Jev's right labels than it fixes wrong ones. E.111 found Jev's own confidence ranks its mistakes well (test AUC 0.72 / 0.85 / 0.88). If only Jev's least confident labels go to the reviewer:
1. how many reviewer calls are saved, and
2. does the cascade stop losing to Jev alone (or beat it)?

**Cost:** $0. No API calls, no GPU: a re-analysis of existing outputs.

## Data (all existing)

- Jev's test decisions and confidence: `results/runs/e102-v1-2026-11-04/plan.jsonl`.
- The reviewer's outputs: `results/runs/e106-v1-2026-11-04/reviews.jsonl`, reviewer `base_local` (untrained gpt-oss-20b), the same outcomes E.106 used for its cascade accuracy. That's the `jev_label` review on Jev's wrong items and the `correct_label` review on Jev's right items (where the label shown equals Jev's).
- Cut points: Jev's decisions on E.104's training pools (`results/runs/e104-v1-2026-11-04/jev_decisions.jsonl`, combined as E.111 does). No test item is used to set them.

## Plan, fixed before computing

- **Policies:** review the q% of items with the lowest Jev confidence, for q = 10, 25 and 50, plus q = 100 (E.106's cascade) and q = 0 (Jev alone). The cut point for each q is the q-th percentile of Jev's confidence on that pool's training items. An item is reviewed if its confidence is **below** the cut. Ties make the real share differ from q, so the share actually reviewed is reported.
- **Routed item:** the reviewer's final label is used. **Other items:** Jev's label stands.
- **Primary measure:** pooled accuracy on the 1,200-item random subsample (400 per pool, as in E.106), as a paired change against Jev alone, with a 95% bootstrap interval (2,000 resamples, seed 114).
- **Rule, written before the run:** a policy "beats Jev alone" only if the interval is above 0. It "stops the loss" if the interval includes 0, while q = 100 stays below.
- **Secondary:**
  - per pool;
  - fixed and broken counts;
  - on WildGuard, harmful items let through, against Jev alone;
  - the same policies for the round-1 trained reviewers (`sft_review`, `sft_plain`, `dpo_review`);
  - a precision check that uses all of Jev's test mistakes (1,058) instead of the subsample's 166. It is a two-part estimate per pool: gain = error rate × share of wrong labels routed and fixed − (1 − error rate) × share of right labels routed and broken. Error rates come from the full test pool. The broken share comes from the subsample's right items, the only right items with a review.
- **Not done:** choosing the best q on test data. Any q found best here is reported as descriptive.

## Result

Run with `uv run python research_analyses/E114_confidence_routing/route.py`; full numbers are in `results.json`. A check: at q = 100 the script reproduces E.106's cascade accuracy for all four reviewers (base 82.7%, sft_review 85.0%, sft_plain 85.5%, dpo_review 86.0%).

**Cut points** (Jev's confidence, from the training pools):

| Pool | 10% | 25% | 50% |
|---|---|---|---|
| Civil Comments | 0.593 | 0.723 | 0.900 |
| WildGuard prompts | 0.687 | 0.890 | 0.990 |
| WildGuard responses | 0.747 | 0.930 | 0.990 |

### Primary: the untrained reviewer (`base_local`), pooled, 1,200 items

Jev alone: 86.2%. It made 166 mistakes in the subsample.

| Policy | Share reviewed | Cascade accuracy | vs. Jev alone, points [95%] | Fixed / broken | Jev's mistakes routed | Harmful let through (of 255) |
|---|---|---|---|---|---|---|
| Jev alone (q = 0) | 0% | 86.2% | – | – | 0 / 166 | 30 |
| Lowest 10% | 10.1% | 85.7% | −0.5 [−1.6, +0.6] | 19 / 25 | 43 / 166 | 30 |
| Lowest 25% | 26.1% | 84.2% | −1.9 [−3.7, −0.1] | 44 / 67 | 97 / 166 | 29 |
| Lowest 50% | 49.0% | 83.2% | −2.9 [−4.8, −0.8] | 57 / 92 | 140 / 166 | 28 |
| All (E.106's cascade) | 100% | 82.7% | −3.5 [−5.5, −1.4] | 62 / 104 | 166 / 166 | 28 |

The two-part check, which uses all 1,058 of Jev's test mistakes, agrees: −0.0, −1.7, −2.7 and −3.4 points.

**By the rule fixed beforehand:**
- **No policy beats Jev alone.**
- **Reviewing the lowest 10% stops the loss:** the interval includes 0 while q = 100 stays below, and it makes about 90% fewer reviewer calls. Reviewing 25% or more still loses to Jev.
- **Routing works as a filter but doesn't fix the cascade.** The lowest 10% holds 26% of Jev's mistakes (43 of 166), and the lowest 50% holds 84%. Even on these items, though, the reviewer breaks more right labels than it fixes (25 against 19).
- **Harmful content let through doesn't rise** under any policy (30 for Jev, 28–30 routed).

### Secondary (descriptive)

**By pool, base reviewer, points against Jev alone:**

| Pool | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Civil Comments | +0.5 [−1.3, +2.3] | +1.5 [−1.8, +5.0] | −0.5 | −1.3 |
| WildGuard prompts | −1.0 [−2.8, +0.8] | −2.3 [−4.8, +0.3] | −3.0 | −4.0 |
| WildGuard responses | −1.0 [−3.3, +1.3] | −5.0 [−7.8, −2.0] | −5.3 | −5.3 |

- **On WildGuard, routing to the reviewer never helps.** Jev is the better model there, and any review costs accuracy.
- **On Civil Comments, a little routing is roughly neutral to slightly positive** (point estimates +0.5 and +1.5, intervals include 0). Choosing "route only Civil Comments" would be picking a policy after seeing test data, so it isn't claimed.

**Round-1 trained reviewers, pooled, points against Jev alone:**

| Reviewer | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| sft_review | +0.2 [−0.9, +1.3] | +0.3 [−1.3, +2.0] | −0.5 [−2.2, +1.4] | −1.2 [−3.2, +0.8] |
| sft_plain | −0.1 [−1.2, +1.1] | 0.0 [−1.7, +1.6] | −0.1 [−1.9, +1.8] | −0.7 [−2.7, +1.5] |
| dpo_review | +0.1 [−1.0, +1.2] | +0.8 [−0.8, +2.4] | +0.4 [−1.3, +2.2] | −0.2 [−2.1, +1.8] |

- **None beats Jev alone.** The best point estimate is DPO at 25%: +0.8 points. Its interval includes 0, and on Civil Comments alone it is +3.3 [0.0, +6.5].
- Round-1 training mostly made the reviewer accept more, so it breaks fewer of Jev's right labels. Routing compounds that, but not significantly.

## What it means

- **Confidence routing is a cost result, not an accuracy result.** It confirms E.111. Reviewing only Jev's lowest-confidence 10% gives the same accuracy as Jev alone (within about ±1.6 points) at about a tenth of the reviewer calls, and it avoids the full cascade's 3.5-point loss.
- **The cascade's problem isn't which items reach the reviewer.** Even on Jev's least confident labels, the reviewer breaks more of the right ones than it fixes of the wrong ones. Better routing can't fix that; only a better reviewer can (round 2's E.108–E.110, never run), or reviewing only where the reviewer is stronger than Jev (Civil Comments, untested on fresh data).
- **For a deployed cascade at today's reviewer quality:** route the lowest ~10% or don't review at all. Reviewing everything is the worst option.
- **Caveats:**
  - one reviewer family and one upstream model;
  - 166 Jev mistakes in the subsample, so intervals are about ±1–2 points;
  - the cut points come from training data, but the policies were run on the same test items E.106 already reported, so this is a re-analysis, not a fresh test.
