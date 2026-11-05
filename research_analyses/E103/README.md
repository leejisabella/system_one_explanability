# E.103: Cheap alternatives to post-training (the Part 2 baseline)

**Status:** done 2026-11-04 (`e103-v1-2026-11-04`, $2.25). **Answer: no prompt change fixes the problem.** We tried four ways of asking the reviewer differently. None of them made it catch meaningfully more of Jev's mistakes, and none made the Jev → reviewer cascade more accurate. Blind-first does shrink the measured pull toward Jev, but not by catching more of Jev's mistakes. What E.102 found is not an artifact of prompt wording, so Part 2 really does need training.

## The short version, in plain terms

**What we tested.** E.102 showed that the reviewer (gpt-oss-20b) goes along with Jev's mistakes far more often than with random mistakes. Mostly that's because the two models share blind spots, but partly it's because the reviewer is swayed by Jev. Before spending effort on post-training, we asked a simple question: can a cheaper fix do the job? Each of four prompt changes ("protocols") re-reviewed exactly the same items as E.102:

| Protocol | What the reviewer sees differently |
|---|---|
| **Standard** | Nothing: E.102's own reviews, kept as the comparison |
| **Blind-first** | It decides on its own first. Then it's shown Jev's decision and asked to review it. |
| **Warning** | "Note: this automated model is often wrong." |
| **Calibrated warning** | "Note: on content like this, this automated model is wrong about N% of the time", with Jev's real error rate as N (25%, 10%, 8%) |
| **Consider the opposite** | Before deciding, make the strongest case for the other label |

**What we found.**
1. **No protocol catches more of Jev's mistakes.** On Civil Comments the reviewer overrides 37–43% of Jev's wrong labels under every protocol, against 40% with the standard prompt. Compared on the same items, every change is within about ±3 points, and none clearly differs from zero.
2. **No protocol makes the cascade more accurate.** Cascade accuracy stays within about ±3 points of the standard prompt's on every pool, and no difference is clear of zero.
3. **Blind-first shrinks the pull toward Jev, but not in the way we hoped.** Measured at equal difficulty, the extra pull toward Jev falls from about +7–8 points to about +2–3 (+2.7 on items right alone, +1.9 on all wrong labels). But the reviewer accepts Jev's wrong labels *as often as before* (63%). What changed is that it now accepts *random* wrong labels more often (55% → 61%). Having committed to an answer, the reviewer mostly sticks to it, and sometimes that answer was already the wrong one. So the gap closes from the wrong side.
4. **Warnings don't work, and the honest one backfires.** "Often wrong" changed almost nothing. "Wrong about 25% of the time" *increased* the pull toward Jev on Civil Comments: the reviewer got better at rejecting random mistakes and worse at rejecting Jev's. One reading is that "wrong 25% of the time" also says "right 75% of the time".
5. **The shared blind spots don't move at all.** On items the reviewer never gets right on its own, it accepts 91–100% of Jev's wrong labels under every protocol. Changing the wording can't help the reviewer see what it can't see.
6. **Harmful content still gets through.** Of the harmful prompts and responses Jev called harmless, every protocol still lets about half or more past the reviewer: 47–64%, against 62–68% with the standard prompt. The calibrated warning on responses is the best case: 22 of 47 let through, against 29.

**An analogy.** In E.102, a junior doctor diagnosed first, and a senior doctor reviewing the diagnosis often agreed with the junior's mistakes. Mostly that was because the senior would have made the same mistakes; partly the senior was swayed. E.103 tried telling the senior "the junior is often wrong", or asking for a diagnosis before showing the junior's. Neither made the senior catch more mistakes. Writing their own answer down first did make the senior less swayed by the junior, but only because they stuck to their first guess, right or wrong. The fix has to make the senior a better doctor on these cases, not a more skeptical reader of the junior's notes.

## What this means for our hypotheses

| # | Hypothesis (fixed before the run, [research_docs/13](../../research_docs/13_cascade_deference.md#hypotheses-decided-before-the-run-1)) | Result | Verdict |
|---|---|---|---|
| 1 | Blind-first reduces deference at equal difficulty (pooled, items right alone), without lowering accept-when-right for Jev's right labels by more than 3 points. | Deference +7.0 [+0.8, +12.6] → **+2.7 [−3.5, +8.7]**; all wrong labels +7.8 → **+1.9 [−2.0, +6.0]**. Accept-when-right changes −1.3, −1.1 and +0.3 points (paired). | **Partly supported.** The pull toward Jev shrinks, and right labels are still accepted. But the drop comes from accepting random mistakes *more*, not Jev's less, and the two intervals overlap, so the reduction itself isn't shown. |
| 2 | Warnings trade one error for another: more overrides of Jev's wrong labels, fewer of its right labels accepted. Neither raises cascade accuracy by more than 2 points. | Override changes −3.7 to +6.9 points; accept-when-right −3.0 to +0.8; the largest cascade change is +1.2. The calibrated warning *raises* deference on Civil Comments (+12.1 [+6.0, +18.2]). | **First half not supported, second half supported.** There's no trade-off because there's hardly any effect. The calibrated warning moves deference the wrong way. |
| 3 | No protocol removes the shared blind spots: at least 80% of Jev's wrong labels accepted on items never right alone. | 91–100% under every protocol and pool. | **Supported.** |
| 4 | No protocol makes the cascade beat its best single model. | Cascade accuracy is 74–77% / 88–89% / 88–90% against Jev's 76% / 90% / 92%. | **Supported, with a qualifier.** No protocol beats Jev on WildGuard. On Civil Comments, four protocols (and the standard prompt) edge past Jev by 0.5–1.5 points (305–309 vs. 303 of 400), within noise. |

## What this means for testing and for Part 2

1. **Part 2 is needed, and it's not a wording problem.** Four quite different prompts, including the one the feedback expected to "remove most of the deference for free", leave the reviewer's outcomes where they were. Any Part 2 gain therefore has to come from training.
2. **The baseline Part 2 has to beat** is the standard prompt; no cheap protocol beats it.

   | Pool | Jev's wrong labels overridden | Jev's right labels accepted | Cascade accuracy | Harmful let through |
   |---|---|---|---|---|
   | Civil Comments | 40% | 87% | 77% | – |
   | WildGuard prompts | 30% | 95% | 88% | 68% |
   | WildGuard responses | 29% | 93% | 88% | 62% |

   Beside that sits a second no-training bar for the cascade as a whole: a majority vote of Jev, gpt-oss alone and Qwen alone (77% / 92% / 93%). It's stronger on WildGuard; on Civil Comments it ties the standard cascade (76.75% vs. 77.25%).
3. **Train for being right, not for disagreeing with Jev.** The large part of the problem is the shared blind spots, where the reviewer is wrong even without Jev. A reviewer trained only to resist Jev would act like the warnings here: it would reject more random mistakes and still miss Jev's. Part 2's training data should therefore be the items Jev finds hard, with the correct labels and with Jev's wrong label shown. Success should be measured as **Jev's wrong labels overridden while Jev's right labels are still accepted**, not as deference alone.
4. **Report deference alongside override and accept rates.** Blind-first shows that the deference measure can improve while nothing useful changes. The measure compares Jev's mistakes with random ones, and making the random side worse also "fixes" it. From here on, every E.1xx should report deference next to Jev's wrong labels overridden and its right labels accepted.
5. **Held-out upstream models are ready.** Llama 4 Maverick, Qwen3-32B and Nova Lite decided every item in all three pools. They make about as many mistakes as Jev, and their mistakes overlap with Jev's about as much as gpt-oss's do (section 5). So Part 2 can test against new upstream models at a similar accuracy, as the feedback asked.
6. **Reasoning traces are now saved** on every review (13,264 of 13,264), for the trace-level checks in research_docs/13.

## Setup

| | |
|---|---|
| Run | `e103-v1-2026-11-04`, commit `82c4b70` (clean). Reviews 09:19–10:05 UTC, upstream models until 11:24 UTC (2026-11-04) |
| Source | E.102's frozen plan (`e102-v1-2026-11-04`): the same items, Jev's wrong labels, matched random wrong labels and the 400-item subsamples. **No new Jev calls.** |
| Reviewer | gpt-oss-20b (Bedrock, flex tier, reasoning effort medium), with the same prompt as E.102 plus each protocol's change (byte-identical to E.102 otherwise, checked in tests) |
| Blind-first | A 3-turn conversation. The reviewer's E.102 alone prompt comes first, then its first E.102 alone answer as an assistant turn (the answer only, no reasoning). Then Jev's decision and the review question. |
| Difficulty | Fixed at the base reviewer's 5 alone samples from E.102, so every protocol is measured on the same scale |
| Upstream models | Llama 4 Maverick (`explainer_open`), Qwen3-32B without thinking (`reviewer_qwen`) and Nova Lite (`decision_alt`). Each decided all 6,406 pool items once, with the reviewer's alone prompt. |
| Volume | 13,264 reviews: 4 protocols × 3,316, all on flex, 1 unparsed. 19,218 upstream decisions, all answered. |
| Cost | **$2.25 on Bedrock** (projected $3.38, cap $6): reviews $1.09, upstream $1.16 (Llama $0.76, Qwen $0.22, Nova $0.18). TypeSafe $0. |
| Hiccups | Llama was throttled on 1,803 requests. These were not billed, and a slower second pass (2 workers) filled them in. Because of that, `manifest.json`'s `config` shows the resume settings; the run's design is the committed [configs/experiments/e103.yaml](../../configs/experiments/e103.yaml). Nova's content filter blocked 7 replies on harmful WildGuard items (4 prompts, 3 responses), and 1 Civil Comments reply used an unrecognized spelling (`non-toxic`): 8 unparsed in all. 20 Llama replies were cut off at 256 tokens, 3 of them unparseable. |

## Results

### 1. Deference at equal difficulty, by protocol

![Pooled deference at equal difficulty, per protocol](figures/fig1_deference_by_protocol.png)

The extra share of wrong labels the reviewer accepts when they come from Jev rather than at random, with both compared at the same difficulty (percentage points, 95% interval; E.102's measure):

| Protocol | Pooled: items right alone | Pooled: right 5 of 5 | Pooled: all wrong labels | Civil Comments, right alone | WildGuard prompts, right alone | WildGuard responses, right alone |
|---|---|---|---|---|---|---|
| Standard (E.102) | +7.0 [+0.8, +12.6] | +5.4 [+1.4, +9.6] | +7.8 [+3.1, +12.2] | +3.6 [−2.6, +9.9] | +13.8 [−10.1, +34.0] | +23.1 [+3.7, +37.5] |
| **Blind-first** | **+2.7 [−3.5, +8.7]** | +3.6 [+0.2, +7.4] | **+1.9 [−2.0, +6.0]** | +2.1 [−3.8, +8.2] | +0.5 [−22.3, +21.4] | +9.3 [−15.8, +33.4] |
| Warning | +7.1 [+1.2, +13.4] | +5.3 [+1.2, +9.9] | +5.9 [+0.9, +11.2] | +7.5 [+1.6, +13.5] | +10.5 [−13.2, +32.8] | +0.8 [−23.1, +21.9] |
| Calibrated warning | +11.5 [+5.9, +17.2] | +6.3 [+2.1, +10.6] | +11.3 [+5.8, +16.9] | **+12.1 [+6.0, +18.2]** | +4.3 [−19.9, +25.7] | +15.7 [−2.3, +30.7] |
| Consider the opposite | +6.6 [+1.0, +12.1] | +5.4 [+1.5, +9.8] | +4.2 [−0.6, +8.8] | +6.2 [+0.6, +12.0] | −1.2 [−25.1, +20.5] | +17.8 [+1.4, +32.2] |

**Interpretation.** Only blind-first brings deference near zero: its intervals for items right alone and for all wrong labels include zero. (Consider-the-opposite's all-wrong-labels interval also includes zero, but its estimate is +4.2.) The warnings and consider-the-opposite leave deference where it was, and the calibrated warning increases it. On their own, the WildGuard estimates are too uncertain to separate protocols; the random arm has few hard items there, as in E.102.

### 2. Where the gap goes

![Jev's wrong labels accepted vs. random wrong labels at the same difficulty](figures/fig2_where_the_gap_goes.png)

The two numbers behind each pooled deference estimate (all wrong labels, all three pools, 1,058 of Jev's wrong labels):

| Protocol | Jev's wrong labels accepted | Random wrong labels accepted, at Jev's difficulty | Gap |
|---|---|---|---|
| Standard (E.102) | 63% | 55% | +7.8 |
| Blind-first | 63% | **61%** | +1.9 |
| Warning | 62% | 56% | +5.9 |
| Calibrated warning | 64% | **52%** | +11.3 |
| Consider the opposite | 60% | 56% | +4.2 |

**Interpretation, plainly.** Under every protocol, Jev's wrong labels get through 60–64% of the time. The protocols differ only in how they treat the *random* mistakes.
- **Blind-first** makes the reviewer accept more random mistakes. It closes the gap, but no fewer of Jev's mistakes get through.
- **The calibrated warning** makes it reject more random mistakes, but not Jev's. It widens the gap.

A random mistake is often obviously wrong, and a warning helps the reviewer reject it. Jev's mistakes look plausible, and a warning doesn't help with those.

**Why blind-first accepts more random mistakes.** The reviewer mostly keeps the answer it committed to:

| Blind-first: final answer keeps the first answer | Civil Comments | WildGuard prompts | WildGuard responses |
|---|---|---|---|
| First answer **agreed** with Jev's wrong label (so it accepts it) | 84% (386/461) | 91% (109/120) | 91% (86/95) |
| First answer **agreed** with a random wrong label | 79% (135/170) | 80% (16/20) | 81% (13/16) |
| First answer was **right**, Jev's label wrong: kept its own | 80% (230/289) | **66% (29/44)** | **69% (34/49)** |
| First answer was **right**, random label wrong: kept its own | 92% (535/580) | 97% (139/144) | 97% (124/128) |
| First answer was **wrong**, Jev's label right: kept its own (wrong) answer | 80% (80/100) | 80% (36/45) | 73% (29/40) |

The first answer is a single E.102 alone sample, and it's sometimes wrong even on items the reviewer usually gets right. Committing to it locks in those slips. With the standard prompt, the reviewer could still have corrected itself when shown a wrong label.

Two more things show up in this table:
- **Even after committing to the right answer, the reviewer switches to Jev's wrong label more often than to a random one:** 20% vs. 8% on Civil Comments, and about a third vs. 3% on WildGuard. That is the deference that remains. Part of it is still difficulty: items where Jev is wrong tend to be harder, even when the reviewer's one sample was right.
- **Blind-first is stubborn in both directions.** When its own first answer was wrong and Jev was right, it kept its wrong answer 73–80% of the time.

### 3. The trade-off: catching Jev's mistakes vs. keeping its right answers

![Jev's wrong labels overridden vs. right labels accepted, per protocol and pool](figures/fig3_tradeoff.png)

| | Civil: wrong overridden | Civil: right accepted | Prompts: wrong overridden | Prompts: right accepted | Responses: wrong overridden | Responses: right accepted |
|---|---|---|---|---|---|---|
| Standard (E.102) | 40% (301/750) | 87% (263/303) | 30% (50/164) | 95% (344/361) | 29% (42/144) | 93% (345/370) |
| Blind-first | 41% | 85% | 24% | 94% | 30% | 94% |
| Warning | 42% | 85% | 27% | 94% | 33% | 94% |
| Calibrated warning | 37% | 84% | 32% | 96% | 36% | 94% |
| Consider the opposite | 43% | 85% | 30% | 94% | 35% | 92% |

Paired changes against the standard prompt, on the same items (points, 95%):

| | Civil: wrong overridden | Civil: right accepted | Civil: cascade | Prompts: wrong overridden | Prompts: cascade | Responses: wrong overridden | Responses: cascade |
|---|---|---|---|---|---|---|---|
| Blind-first | +0.5 [−2.7, +3.7] | −1.3 [−5.0, +2.0] | −0.5 [−4.0, +3.2] | −6.1 [−13.4, +1.2] | −0.2 [−3.0, +2.2] | +0.7 [−6.2, +8.3] | −0.8 [−3.5, +2.0] |
| Warning | +1.7 [−1.7, +5.2] | −1.7 [−5.6, +2.0] | −1.0 [−4.8, +3.0] | −3.7 [−11.6, +4.3] | −0.8 [−3.8, +2.0] | +4.2 [−2.8, +11.8] | +1.2 [−1.0, +3.5] |
| Calibrated warning | −2.8 [−6.0, +0.5] | −3.0 [−6.6, +0.7] | −3.2 [−6.8, +0.5] | +1.2 [−6.1, +9.2] | +1.0 [−1.5, +3.2] | **+6.9 [0.0, +14.6]** | 0.0 [−2.8, +2.8] |
| Consider the opposite | +2.4 [−0.7, +5.6] | −1.3 [−4.3, +1.7] | 0.0 [−3.2, +3.2] | 0.0 [−7.9, +7.3] | 0.0 [−2.5, +2.8] | +5.6 [−2.1, +13.9] | −0.5 [−3.2, +2.0] |

Accept-when-right on WildGuard changes by −1.4 to +0.8 points for every protocol (`data/vs_standard.csv`).

**Interpretation.** Every point sits within a few points of the standard prompt (the largest is blind-first's −6.1 on WildGuard prompts), and no interval excludes zero except the calibrated warning on responses, which only touches it. The prompts didn't make the reviewer contrarian, as hypothesis 2 feared. They just didn't do much. Consider-the-opposite costs the most (27% more output tokens than standard on Civil Comments) and buys nothing measurable.

### 4. The shared blind spots

Jev's wrong labels that the reviewer accepted on items it never gets right on its own (0 of 5):

| Protocol | Civil Comments (292 items) | WildGuard prompts (75) | WildGuard responses (67) |
|---|---|---|---|
| Standard (E.102) | 93% | 96% | 97% |
| Blind-first | 94% | 99% | 100% |
| Warning | 91% | 97% | 99% |
| Calibrated warning | 94% | 92% | 91% |
| Consider the opposite | 92% | 93% | 93% |

**Interpretation.** These are 39–47% of all of Jev's mistakes (E.102), and no prompt reaches them. It's the clearest evidence that the main problem is what the reviewer knows, not how it is asked.

### 5. Pipelines and upstream models

Accuracy on the 400-item random subsample of each pool. These pipelines need no new calls.

| | Civil Comments | WildGuard prompts | WildGuard responses |
|---|---|---|---|
| Jev alone | 76% [71, 80] | 90% [87, 93] | 92% [90, 95] |
| gpt-oss alone | 75% [71, 79] | 89% [85, 91] | 90% [87, 93] |
| Jev → reviewer, standard | 77% | 88% | 88% |
| Jev → reviewer, best protocol | 77% (standard / consider the opposite) | 89% (calibrated) | 90% (warning) |
| **Majority vote: Jev, gpt-oss alone, Qwen alone** | **77% [72, 81]** | **92% [89, 94]** | **93% [90, 95]** |

The vote is the best pipeline on both WildGuard pools, ahead of Jev alone by 1.5 points on prompts and 0.75 on responses: within noise. On Civil Comments it is just behind the standard and consider-the-opposite cascades (307 vs. 309 of 400) and ahead of Jev alone (303). Unlike the cascade, it doesn't let one model's answer steer another's. It also never does worse than Jev alone here.

The upstream models on the whole pools (error correlation with Jev on the pool, and with gpt-oss alone on the subsample):

| | Civil: error rate | Civil: phi with Jev / gpt-oss | Prompts: error rate | Prompts: phi | Responses: error rate | Responses: phi |
|---|---|---|---|---|---|---|
| Jev (E.102) | 25% | – / 0.47 | 10% | – / 0.71 | 8% | – / 0.58 |
| Llama 4 Maverick | 28% | 0.59 / 0.48 | 12% | 0.53 / 0.52 | 9% | 0.59 / 0.49 |
| Qwen3-32B | 26% | 0.62 / 0.46 | 12% | 0.58 / 0.55 | 9% | 0.57 / 0.47 |
| Nova Lite | 24% | 0.54 / 0.36 | 13% | 0.50 / 0.50 | 10% | 0.51 / 0.45 |

**Interpretation.** The three upstream models are about as accurate as Jev, and their mistakes overlap with Jev's (0.50–0.62) about as much as gpt-oss's do. Every model shares the same hard items, so Jev isn't special in that respect. For Part 2, that makes them fair held-out tests: a reviewer trained on Jev's mistakes can be tested on Llama's, Qwen's and Nova's at similar accuracy. Nova Lite overlaps least with gpt-oss on Civil Comments (0.36). Nova also misses the most harmful responses (23% of them).

### 6. Safety: harmful content that gets through

![Harmful items Jev missed that the reviewer also let through, per protocol](figures/fig4_harmful_let_through.png)

| Harmful items Jev called harmless, let through by the reviewer | WildGuard prompts (60) | WildGuard responses (47) |
|---|---|---|
| Standard (E.102) | 41 (68%) | 29 (62%) |
| Blind-first | 37 (62%) | 30 (64%) |
| Warning | 35 (58%) | 27 (57%) |
| Calibrated warning | 38 (63%) | **22 (47%)** |
| Consider the opposite | 36 (60%) | 26 (55%) |

**Interpretation.** This is where "misalignment" is the right word (research_docs/13): harmful requests and harmful answers get past both filters. Every protocol lets fewer through than the standard prompt on prompts, but with 60 and 47 items the differences are 1–7 items and within noise. Even at its best, the cascade misses about half of what Jev misses.

### 7. Cost

| Protocol | Cost of 3,316 reviews | Mean output tokens, Civil Comments | Latency per review |
|---|---|---|---|
| Standard (E.102) | – | 486 | – |
| Blind-first | $0.23 | **386** | **1.3 s** |
| Warning | $0.26 | 491 | 1.6 s |
| Calibrated warning | $0.27 | 494 | 1.6 s |
| Consider the opposite | $0.32 | **615** | 2.1 s |

Blind-first is the cheapest review: the reviewer writes 11–21% fewer output tokens once it has committed (386 vs. 486 on Civil Comments, 254 vs. 285 on prompts, 239 vs. 302 on responses). Note that it also needs the reviewer's alone answer first, so a real blind-first cascade costs about two reviewer calls per item.

## Caveats

- **The blind-first reduction isn't tested directly.** Its deference interval includes zero and the standard's doesn't, but the two intervals overlap. Either way, the reduction doesn't come from catching more of Jev's mistakes (section 2).
- **Blind-first reuses an E.102 answer.** The committed answer is the reviewer's first E.102 alone sample, not a fresh one. This is the same as running both turns live, except that the first turn's reasoning isn't carried over. A live blind-first, where the model sees its own reasoning, might anchor even more strongly.
- **The protocols are single-turn wording changes.** Debate, several reviewers or self-consistency voting weren't tested. The majority vote in section 5 is the only multi-model pipeline.
- **WildGuard per-pool estimates are uncertain**, as in E.102: few hard random mistakes, and 60 and 47 harmful items.
- **Difficulty is still measured roughly** (5 alone samples), so the remaining deference is an upper bound, as in E.102.
- **The upstream models answered once each** (no repeated samples), with Qwen and Llama not reasoning. Their error rates are on whole pools; their overlap with gpt-oss is on the subsample.

## Reproduce

```bash
uv run python scripts/preflight_e103.py                                       # every protocol and upstream, a few cents
uv run python -m sysone.experiments.e103 --run-id e103-v1-2026-11-04           # resumes; everything is saved
uv run python -m sysone.analysis.e103 results/runs/e103-v1-2026-11-04 results/runs/e102-v1-2026-11-04
uv run --group analysis python research_analyses/E103/make_figures.py
```

`data/` holds the run's `summary.json` and `report.md`, and every table above:
- `deference_by_protocol.csv`, `protocols.csv`, `vs_standard.csv`, `blind_first_own_answer.csv`
- `no_call_pipelines.csv`, `upstream.csv`
