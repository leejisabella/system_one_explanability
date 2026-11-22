# Part 2, round 2 (E.108–E.113): a negative result

**Status:** round 2 was designed on 2026-11-05 and closed on 2026-11-21. **Round 2's training never ran.** E.108 (consistency training and balanced-proposal SFT), E.109 (GRPO) and E.110 (activation steering) are built and tested offline, but nothing was trained, sampled live or run on a GPU (on hold since 2026-11-05, the user's decision). What did run were four cheap checks: step 0, E.111, E.112 and E.113. None of the four found a cheap route to the mistakes the reviewer can't fix alone (E.113 produced no data at all), and on 2026-11-21 the user stopped distillation. This page summarizes them. Each write-up has the full numbers and methods; the links are in [the last section](#the-write-ups).

## The question

Round 1 ([E1xx_summary](../E1xx_summary/README.md)) found that most of the Jev mistakes the reviewer (gpt-oss-20b) lets through are shared blind spots, and that SFT (E.106) and DPO (E.107) on the reviewer's own correct reasoning didn't catch more of them. Both failed all five pre-registered criteria.

Round 2 split Jev's 1,058 test mistakes by how the untrained reviewer does on them alone (5 tries each, from E.106's base run):

| Group | Share of Jev's mistakes | Civil / WG prompts / WG responses | What it is |
|---|---|---|---|
| A. Right alone on the first try | 34% (364) | 281 / 41 / 42 | Knows the answer, gives in to Jev's wrong label 25% of the time |
| B. Wrong on the first try, right at least once in 5 | 21% (221) | 155 / 42 / 24 | Inconsistent |
| C. Never right alone (0 of 5) | 45% (473) | 314 / 81 / 78 | Shared blind spots |

The training methods (E.108–E.110) aim at groups A and B, which the reviewer can already solve. If it stopped giving in on group A, it would catch about 46% of Jev's mistakes instead of 38% ("about +9 points pooled", [research_docs/15](../../research_docs/15_part2_round2.md)). Group C is out of reach for self-training: there is no correct answer of the reviewer's own to learn from.

So round 2 asked two things before spending on a GPU:
- **Is group C a benchmark ceiling, or real mistakes worth reaching?** (step 0)
- **Is there a cheap signal that reaches the Jev mistakes the reviewer can't fix alone?** From Jev's own confidence (E.111), or from a stronger teacher whose traces could be distilled into the reviewer (E.112, E.113).

## The answer in four points

1. **WildGuard's group C isn't just label noise, but nothing we have confirmed reaches it.** In one model reader's reading of 30 WildGuard group C items, 10 were clear model mistakes, exactly at the pre-registered threshold. All 10 were benign items both models flagged as harmful (step 0). E.112's exploratory Mistral result on WildGuard over-flagging is the only lead, and it is untested on fresh items. On Civil Comments the ceiling assumption stands, based on E.105's reading of 30 never-solved items; step 0 didn't re-check it.
2. **Jev's mistakes can be ranked, but ranking isn't fixing.** Jev's own confidence ranks its mistakes well (test AUC 0.72 / 0.85 / 0.88), and the item text adds nothing to the AUC. Flipping its least confident labels gains about 1 point or less (point estimates +0.1 to +1.0; upper bounds up to about +1.6). Group C is the hardest to detect in every pool (E.111).
3. **No affordable teacher passes the pre-registered rule.** Mistral Large 3, Qwen3-235B and gpt-oss-120b at high effort don't override clearly more of Jev's mistakes than gpt-oss-20b. On Civil Comments all three do worse (E.112). The screen is small, so this is a failed screen, not evidence of no effect: Mistral's interval (−8 to +14 points) can't exclude the 10-point gain the rule asked for.
4. **Claude can't be used as a teacher here.** Both smoke-test agents were stopped by Anthropic's safeguards before producing any output, and Claude Sonnet 5 wasn't available on Bedrock (agreement NOT_AVAILABLE) or in the two Vertex locations checked (E.113).

**What it means:** the cheap routes to the mistakes the reviewer can't fix on its own all came up empty before any GPU spend. This is not evidence that training can't help: the round-2 training methods were never run, so none of round 2's criteria R1–R6 was ever evaluated. See [what it rules out](#what-it-rules-out-and-what-it-doesnt).

## Step 0: is WildGuard's group C a benchmark ceiling?

Laptop plus one Claude agent as reader; no billed API calls, $0; 2026-11-05. Round 2's plan treated group C as a ceiling of the benchmark, not a target. The evidence was from Civil Comments only: in E.105, about half of 30 never-solved items had a doubtful gold label and none was a clear model mistake. On WildGuard, group C is about half of Jev's mistakes, so we checked there first.

- **Method:** 30 WildGuard group C test items (15 prompts, 15 responses), drawn at random by hash from E.106's base run. A separate Claude agent, which wasn't told the rule, read each item's text, gold label, Jev's label and the reviewer's 5 answers alone with their reasoning. It classified each as a doubtful gold label, genuinely ambiguous, or a clear model mistake, as E.105 did for Civil Comments.
- **Rule, fixed beforehand:** if 10 or more of 30 are clear model mistakes, the ceiling assumption doesn't hold for WildGuard.

| | Doubtful gold label | Genuinely ambiguous | Clear model mistake |
|---|---|---|---|
| WildGuard prompts (15) | 5 | 5 | 5 |
| WildGuard responses (15) | 4 | 6 | 5 |
| **All (30)** | **9** | **11** | **10** |

- **The assumption doesn't hold on WildGuard.** 10 of 30 is exactly at the threshold.
- **All 10 mistakes are over-flagging:** benign items both models called harmful. The triggers were a harm keyword in a harmless setting (a wildfire in Minecraft, Einstein's "brainwave data", profanity in film quotes), past stereotypes quoted in order to criticize them, and, on responses, judging the request instead of what the response delivers.
- **In that reading, the other 20 were doubtful gold labels (9) or genuinely ambiguous (11).** Every item with a "harmful" gold label looked wrong or contestable (4 doubtful, 1 ambiguous), for example a request for age-appropriate reading recommendations. Some "unharmful" gold labels looked too lenient, mostly jailbreak-framed prompts.
- **Decided 2026-11-05:** no gold data from a stronger model. WildGuard's group C stays out of training and is reported separately, as a real gap rather than a ceiling. (This was revisited on 2026-11-21; see E.112.)
- **Caveats:** the verdict sits exactly on the threshold, so one reading the other way changes it. There was one reader, a model rather than a person, and 30 of the 159 WildGuard group C items.

## E.111: can Jev's own signals find and fix its mistakes?

Free: no API calls, no GPU; 2026-11-16. A "Jev is wrong" logistic regression is trained on E.104's training pools and scored on E.102's frozen test pools, split by groups A/B/C.
- **conf:** Jev's own signals only (its confidence, whether its samples split, which label it chose).
- **text:** conf plus TF-IDF of the item text.
- **Flip rule:** flip Jev's label when the predicted chance it's wrong is above a threshold. The threshold was chosen by cross-validation on the training pool only, then frozen before the test pools.

| | Civil Comments | WildGuard prompts | WildGuard responses |
|---|---|---|---|
| Jev's test error rate | 25.0% | 9.7% | 8.4% |
| AUC, conf | 0.72 | 0.85 | 0.88 |
| AUC, text | 0.72 | 0.83 | 0.88 |
| AUC by group, conf (A / B / C) | 0.83 / 0.73 / 0.61 | 0.89 / 0.92 / 0.79 | 0.95 / 0.93 / 0.82 |
| Flip rule, conf: labels flipped (fixed / broken) | 104 (54 / 50) | 72 (41 / 31) | 27 (22 / 5) |
| Accuracy change, conf, points [95%] | +0.1 [−0.5, +0.8] | +0.6 [−0.4, +1.6] | +1.0 [+0.4, +1.6] |

Group AUCs compare that group's mistakes against all of Jev's right labels, not mistakes within the group. The groups are defined by the untrained reviewer's (gpt-oss-20b's) alone tries in E.106's base run, not by Jev.

- **Jev's mistakes are fairly easy to rank, almost entirely from its own confidence.** Text features don't raise the AUC (within ±0.02 of conf in each pool).
- **Ranking isn't correcting.** Flipping the least confident labels gains about 1 point or less (point estimates +0.1 to +1.0; upper bounds up to about +1.6). Only the WildGuard responses interval excludes 0. With text features the flip gain on WildGuard prompts is slightly higher (+0.8 [−0.2, +1.7] against +0.6), within noise. On Civil Comments, the flips break about as many labels as they fix.
- **Group C is the hardest to detect in every pool** (AUC 0.61 / 0.79 / 0.82): these are the mistakes Jev makes with confidence.
- **So a detector on these signals is a good router, not a fixer.** A fix has to come from understanding the items. That motivated the teacher screen.
- **The reviewer ignores this signal.** E.101 found that showing Jev's confidence to the reviewer had no measurable effect.
- **A bar for later work:** any trained verdict head on gpt-oss-20b has to beat the conf row, with a higher AUC and a flip-rule gain whose interval is above 0, especially on groups B and C.

## E.112: is there an affordable teacher to distill from?

Bedrock, 2026-11-21. On that date the user allowed distillation from a stronger teacher, as training data only, never as an extra stage in the cascade. The teacher reviews training-pool items in the reviewer's format without the gold label, and only traces that reach gold on their own would be kept. E.112 is the cheap screen before building any of that.

- **Items:** 300 training-pool items from E.104's mix. Per pool: 40 *unsolved* Jev mistakes (no E.105 try by either gpt-oss model reached gold), 30 *solved* Jev mistakes, and 30 items Jev got right.
- **Teachers:** Mistral Large 3, Qwen3-235B (the same family as the held-out Qwen3-32B), and gpt-oss-120b at high reasoning effort. Each reviewed every item once.
- **Baseline:** gpt-oss-20b's first E.105 review try on the same items, at no cost.
- **Weighting:** override rates are weighted back to the whole mix (2,025 Jev mistakes, 67% of them Civil Comments). The per-stratum rates favor the 20b baseline, because its own tries define the strata. They also count against gpt-oss-120b: the unsolved stratum excludes items gpt-oss-120b (E.105's teacher, at medium effort) solved in E.105, so its WildGuard-unsolved score is partly low by construction too.
- **Pass rule, fixed before the run; all four must hold:** override gain over 20b of at least 10 points with the paired 95% interval above 0; accept-when-right at least 90%; gold reached on at least 25% of WildGuard's unsolved items; at least 95% of replies parsed.

### Pre-registered result: no teacher passes

| | Override of Jev's mistakes (weighted) | vs. 20b, points [95%] | Accept when right | WildGuard unsolved reaching gold | Parsed | Passes |
|---|---|---|---|---|---|---|
| Mistral Large 3 | 34.2% | +2.6 [−8.2, +14.2] | 92.2% | 66% (53/80) | 100.0% | No (override gain) |
| Qwen3-235B | 23.1% | −8.5 [−17.6, +2.1] | 96.7% | 36% (29/80) | 100.0% | No (override gain) |
| gpt-oss-120b, high | 26.5% | −5.1 [−13.4, +2.4] | 93.3% | 6% (5/80, partly by construction) | 99.7% | No (override gain, WildGuard unsolved) |
| gpt-oss-20b (E.105, first try) | 31.6% | – | 91.1% | 0% (0/80, by construction) | 100.0% | – |

- **No teacher overrides clearly more of Jev's mistakes than the reviewer itself.** Mistral and Qwen pass the other three criteria and fail only on override gain. Qwen's and gpt-oss-120b's point estimates are below 20b.
- **gpt-oss-120b, even at high effort, isn't a better teacher than the reviewer.** This confirms E.105, but read it from the weighted override gain (−5.1, interval includes 0), not from the WildGuard-unsolved criterion, which is partly selected against gpt-oss-120b (unsolved items are ones no E.105 try by either gpt-oss model solved).
- **The screen is small:** 300 items, one try per teacher, weighted intervals about ±8 to ±11 points. It can fail a teacher on the pre-registered rule, but it can't rule out a 10-point gain for Mistral.
- **On Civil Comments every teacher is worse than 20b** (weighted override: Mistral 18%, Qwen 9%, gpt-oss-120b 23%, 20b 33%). Mistral and Qwen almost never overturn Jev's "non-toxic" when gold says toxic (2 and 0 of 41). gpt-oss-120b does so on 9 of 41, against 14 of 41 for 20b. This fits E.105's reading that many of these gold labels are doubtful. Because Civil Comments is 67% of the weighted mix, it decides the pooled failure.

### Exploratory, not pre-registered: Mistral on WildGuard

This split was chosen after seeing the data, and the cells are small (40 and 30 items). The counts are raw screen-sample counts, not weighted to the mix, except the first row. The teacher counts here and in the Civil Comments bullet above were counted from the run's raw replies (`results/runs/e112-v2-2026-11-21/reviews.jsonl`), and the 20b counts from its first E.105 review try (`results/runs/e105-v1-2026-11-04/traces.jsonl`); most aren't in E.112's README or report.

| WildGuard only | Mistral Large 3 | gpt-oss-20b |
|---|---|---|
| Jev's WildGuard mistakes overridden (weighted) | 67% | about 28% |
| Benign items Jev called harmful, fixed (over-flagging) | 85 of 112 | 17 of 112 |
| Harmful items Jev let through, caught | 8 of 28 | 11 of 28 |
| Jev's right "harmful" labels kept | 30 of 32 | 32 of 32 |

- **Mistral's whole WildGuard gain is in the over-flagging direction,** the same kind of group C item step 0 found. In the dangerous direction, harmful content Jev let through, it is no better than 20b (8 against 11 of 28, a difference within noise on 28 items). gpt-oss-120b caught 13 of 28 and Qwen 8 of 28.
- **What's left of distillation would be narrow:** Mistral, WildGuard only, teaching the reviewer to stop agreeing when Jev over-flags benign content. It would first need confirming on a fresh, disjoint sample under a rule fixed beforehand, plus a check that harmful content let through doesn't rise.

## E.113: Claude as a blind teacher

2026-11-21, $0. The plan was to have Claude subagents review the same 300 items as E.112, blind, so their traces could be screened the same way.
- **Built (no API calls):** [build_blind.py](../E113_claude_teacher/build_blind.py) wrote one file per item under a random ID, holding exactly the review prompt gpt-oss-20b and the API teachers saw. The gold label, item ID, pool, stratum and whether Jev was right never went into the blind folder. [audit.py](../E113_claude_teacher/audit.py) checks each reviewer's tool calls for leakage.
- **What happened:** both smoke-test agents were stopped by Anthropic's safeguards (category `reasoning_extraction`) before producing any output. No data was produced, and we made no attempt to work around it.
- **Claude Sonnet 5 through a cloud provider wasn't available either:** on Bedrock the model agreement shows NOT_AVAILABLE, and on Vertex the quota is 0 in `global` and the model isn't found in `us-east5`.
- **Decided 2026-11-21 (the user):** distillation is stopped, after E.112 found no passing teacher and E.113's Claude agents were stopped by safeguards. The hold on live runs and GPU stays in place.
- **The run folder** `results/runs/e113-v1-2026-11-21/` holds only the blinding key (`key.json`); there is no ledger and no reviews.

## Round 2's criteria, never scored

Fixed on 2026-11-05, before any round-2 model was trained ([research_docs/15](../../research_docs/15_part2_round2.md#pre-registered-criteria-for-round-2)). They apply to E.108's two arms and E.109, paired with the base on the same test items. A model works if R1, R2 and R3 all pass; R4–R6 say how far it generalizes and whether it's useful.

| # | Rule | Base value |
|---|---|---|
| R1 (primary) | Override of Jev's wrong labels on solvable items (groups A + B) rises, and accept-when-right falls by at most 2 points | Override 61.7% (361/585); accept 89.9% |
| R2 (primary) | Group A: Jev's wrong labels accepted falls | 25.0% (91/364) |
| R3 (primary) | Overrides more on solvable items than E.106's `sft_plain` control | Measured in the same run |
| R4 | R1's override change above 0 in each of the three pools | – |
| R5 | R1 holds on the held-out upstream models (Llama 4 Maverick, Qwen3-32B, Nova Lite) | Measured in the same run |
| R6 | Cascade accuracy beats Jev alone | Base −3.5 points; Jev 86.2% |

## What it rules out and what it doesn't

**It rules out, at this budget and on these samples:**
- **Distilling a stronger teacher into the reviewer across all pools, as screened.** No affordable teacher cleared E.112's pre-registered screen, and this attempt to use Claude as a teacher produced no data (E.112, E.113). Mistral's interval (−8 to +14 points) can't exclude a useful gain, so this is a failed screen at this sample size, not evidence of no effect. Distillation was stopped by the user's decision, not ruled out by the data.
- **Fixing Jev's mistakes from Jev's own confidence.** It is a good router, but flipping labels gains about 1 point or less (E.111).
- **Treating group C as uniformly a ceiling.** By the pre-registered rule, which step 0 met exactly (10/30, one model reader), WildGuard's group C is treated as a real gap rather than a ceiling. Nothing we have confirmed reaches it; E.112's exploratory Mistral result is the only lead (step 0, E.112).

**It does not rule out:**
- **Training on groups A and B.** E.108's consistency training and balanced-proposal SFT target the items the reviewer can already solve. They were never run, so round 2 says nothing about whether they work. The +9-point group-A ceiling still stands as a possible gain. Even at that ceiling, the plan expects the cascade to stay below Jev alone (R6; base −3.5 points), because group C still counts.
- **GRPO (E.109) and activation steering (E.110).** Built and tested offline (357 tests passed as of 2026-11-05, before `tests/test_e112.py` was added), but none of the GPU code has run on a GPU. Where the code lives is [below](#the-untested-code).
- **A narrower, WildGuard-only fix for over-flagging.** E.112's one lead is exploratory and needs a fresh test.
- **Other teachers or people.** E.112 screened three affordable models on 300 items; it doesn't show that no source of correct reasoning exists.
- **The held-out-domain study** (train without WildGuard responses and test on them, on the existing `part2_no_responses_v1` split). It was deferred, not rejected, until a round-2 method works.

**Not pursued by decision on 2026-11-05, so neither tested nor ruled out by evidence:** deference DPO with redesigned pairs, self-consistency voting, uncertainty-gated escalation to a third model, training toward group C, more prompt variants, and tuning round 1. The reasons are in [research_docs/15](../../research_docs/15_part2_round2.md#what-we-decided-not-to-do-and-why).

**And it carries round 1's caveats:** one reviewer, one upstream model, three moderation tasks.

## What's left

These are options, not decisions. None runs without the user's go-ahead.

| Option | Cost | What it would test | Note |
|---|---|---|---|
| Route by Jev's confidence | Free, local data only | Sending only Jev's low-confidence labels to the reviewer, to cut reviewer calls and cost per item | Won't make the cascade more accurate (E.111) |
| Change the WildGuard reviewer prompt | Small Bedrock spend | A clearer policy rule against over-flagging benign content, with no training | On fresh items, pass rule fixed beforehand, and harmful content let through must not rise. Round 1's prompt variants (E.103) changed nothing, but those weren't aimed at over-flagging |
| Probe gpt-oss-20b's frozen activations | GPU, forward passes only | Whether the reviewer's internals hold a signal beyond Jev's confidence; it has to beat E.111's conf baseline | A kill test for the detection route only: if a probe can't beat Jev's confidence, a fine-tuned verdict head or LoRA on the same signal probably won't either (E.111). It says nothing about whether E.108–E.110 reduce deference on groups A and B |
| E.108, then E.109 and E.110 | ~$0.20 Bedrock (E.108's dry run: 2,806 rows, 524 needing new samples, at most about 2,700 calls; [E108](../E108/README.md#building-the-data-not-run-yet)), then roughly 12–24 GPU hours across sessions | Round 2 as designed, on groups A and B, against criteria R1–R6 | Independent of the probe. Whether to spend GPU money on it first is a judgment call, not something the data implies |

The GPU estimates in the plan are 4–5 GPU hours for E.108 (~$10–20 of credits), 6–15 for E.109 (~$20–60, likely 2 × H100) and 2–4 for E.110, at the Spot H100 rate seen in round 1 (about $2.50–4 an hour). E.110's figure is soft: about 7,000 generations may take longer.

### The untested code

The full map and runbook are in [research_docs/15](../../research_docs/15_part2_round2.md#code-map-and-runbook).

| Piece | Where | Tests |
|---|---|---|
| Groups A/B/C, criteria R1–R6 | `sysone.analysis.e106` (`base_groups`, `evaluate_round2`); `configs/experiments/e106.yaml` | `tests/test_part2_round2.py` |
| E.108 data | `sysone.experiments.e108`, `configs/experiments/e108.yaml` | `tests/test_part2_round2.py` |
| E.108 training and evaluation | `configs/posttrain/{consist,balanced}_review.yaml`, `scripts/run_e108.sh` | `tests/test_part2_round2.py` |
| E.109 GRPO | `sysone.posttrain.grpo`, `configs/posttrain/grpo_review.yaml`, `scripts/run_e109.sh` | `tests/test_grpo.py` |
| E.110 steering | `sysone.interp.steering`, `scripts/run_e110.sh` | `tests/test_steering.py` |

**Open prerequisites before any of it runs:** E.108's Bedrock sampling (~$0.20) is the next live step; the Spot quota for 2 × H100 must be checked before E.109; and the vLLM version for GRPO isn't settled (it isn't in `pyproject.toml`). All are under the hold.

## Budget

All round-2 spend is Bedrock API calls. No TypeSafe, GCP or GPU spend in round 2.

| Step | Where | Cost |
|---|---|---|
| Step 0 | Laptop plus one Claude agent as reader; no billed API calls | $0 |
| E.108–E.110 | Not run (on hold) | $0 |
| E.111 | Laptop, no API calls | $0 |
| E.112, `e112-v1-2026-11-21` (stopped: it didn't save the full replies; 79 reviews saved, 87 calls in the ledger) | Bedrock | $0.0506 (ledger) |
| E.112, `e112-v2-2026-11-21` (900 teacher reviews on 300 items) | Bedrock | $0.5316 |
| E.113 | Claude subagents, stopped before any output; no ledger | $0 |
| **Round 2 total** | | **$0.58** |

- **E.112 by teacher (v2):** Mistral $0.26, Qwen $0.15, gpt-oss-120b $0.12. The run's preflight projected about $0.67 (the config comment says ~$0.5), with a worst case of about $2.7 under the run cap.
- **v1's two figures:** its manifest records $0.046; its ledger sums to $0.0506. The table uses the ledger, which is what the project caps count.
- **Project totals from every ledger under `results/runs`:** Bedrock $7.57 of the $50 cap, GCP $0.20 of $50, TypeSafe $2.10 of $5. Round 1's GPU work was about 5.5 GPU hours plus the disk (roughly $13–22 of education credits; E1xx_summary's figure, which STATUS rounds to 5.4). No GPU resources remain. The provider consoles are the authoritative record.

## Caveats that apply throughout

- **One reader for step 0,** and it was a model, not a person; its verdict sits exactly on the threshold.
- **Small screen samples:** E.112 used 300 items with one try per teacher; E.111's flip gains are around a point with intervals of similar size.
- **E.112's strata are defined by the gpt-oss models' own E.105 tries,** so per-stratum rates aren't neutral between 20b, gpt-oss-120b and the other teachers.
- **The WildGuard split in E.112 is exploratory,** chosen after seeing the data.
- **One reviewer, one upstream model, three moderation tasks,** as in round 1. Costs are from the ledgers; the provider consoles are the authoritative record.

## The write-ups

| Experiment | Write-up | One line |
|---|---|---|
| Design | [research_docs/15](../../research_docs/15_part2_round2.md) | Round 2's plan: groups A/B/C, E.108–E.110, criteria R1–R6, what we decided not to do |
| Step 0, E.108 | [E108](../E108/README.md) | WildGuard's group C: 10 of 30 clear model mistakes, all over-flagging; E.108's data built but not sampled |
| E.111 | [E111_jev_error_predictability](../E111_jev_error_predictability/README.md) | Jev's mistakes rank well from its confidence; flipping them gains about 1 point or less |
| E.112 | [E112](../E112/README.md) | No affordable teacher passes; exploratory: Mistral fixes WildGuard over-flagging only |
| E.113 | [E113_claude_teacher](../E113_claude_teacher/) | Claude blind-teacher agents stopped by safeguards before any output; no data |
| Round 1 | [E1xx_summary](../E1xx_summary/README.md) | E.101–E.107: shared blind spots, not deference; SFT and DPO fail all five criteria |
