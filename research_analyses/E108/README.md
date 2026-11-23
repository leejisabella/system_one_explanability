# E.108: consistency training and balanced-proposal SFT

**Status:** done 2026-11-23. **Answer: neither arm works.** Both fail R1, R2 and R3, the three primary criteria fixed on 2026-11-05 before training. The design is in [research_docs/15](../../research_docs/15_part2_round2.md). Step 0 (the ceiling check) and the data build come first below; [the result](#result-2026-11-23-neither-arm-works) follows them.

## Step 0: is WildGuard's group C a benchmark ceiling?

**The answer: no, not by the rule we fixed in advance.** 10 of the 30 items are clear model mistakes, exactly at the threshold (10 or more), so the ceiling assumption doesn't hold for WildGuard.

**What we did.**
- **The sample:** 15 WildGuard prompts and 15 WildGuard responses from group C, drawn at random by hash. These are items where Jev is wrong and the untrained reviewer is wrong in all 5 of its tries alone.
- **The reading:** each item was classified by an independent read: a separate Claude agent, which wasn't told the rule.
  - The agent saw the text, the gold label, Jev's label and the reviewer's 5 answers with their reasoning.
  - The readings, with a short neutral note per item, are in [data/group_c_readings.csv](data/group_c_readings.csv).
  - The reading sheet holds WildGuard's harmful text, so it stays in the gitignored run folder: `results/runs/e106-v1-2026-11-04/step0/`.

| | Doubtful gold label | Ambiguous | Clear model mistake |
|---|---|---|---|
| WildGuard prompts (15) | 5 | 5 | 5 |
| WildGuard responses (15) | 4 | 6 | 5 |
| **All (30)** | **9** | **11** | **10** |

**What the readings show.**
1. **The model mistakes run one way: the models flag benign items as harmful.** In all 10 the gold label is "unharmful" and both models said "harmful". The triggers were:
   - a harm keyword in a harmless setting: a wildfire in Minecraft, Einstein's "brainwave data", profanity in film quotes;
   - past stereotypes shown in order to criticize them;
   - on responses, judging the request instead of what the response actually delivers, including garbled or generic responses that give no real help.
2. **Every item whose gold label is "harmful" looks wrong or contestable** (4 doubtful, 1 ambiguous). Examples: a request for age-appropriate reading recommendations, and a response that refuses the harmful request and helps instead.
3. **Some "unharmful" gold labels look too lenient,** mostly on jailbreak-framed prompts and on responses that comply with a harmful goal.

**What this means.**
- **On WildGuard, part of group C is learnable,** and it's mostly one kind of mistake: over-flagging. That's different from Civil Comments, where E.105's reading found no clear model mistakes.
- **The criteria aren't changed.** Round 2 scores groups A and B either way, as fixed in research_docs/15.
- **WildGuard's group C is reported as a real gap,** not as a ceiling, in every round-2 write-up.
- **Decided 2026-11-05: no new training data for these items.** A stronger model could have written gold-reaching reasoning for them, as training data only, but we chose to leave WildGuard's group C out of training and report it separately.

**Caveats.**
- **The result sits exactly on the threshold.** One reading the other way changes the verdict.
- **There was one reader,** and it was a model, not a person. The classifications are easy to check: the sheet is in the run folder.
- **30 items** out of the 159 WildGuard group C items.

## Building the data (run 2026-11-22)

`sysone.experiments.e108` selects the rows, samples new review traces where E.105 has no correct one, and exports both arms plus E.109's GRPO prompts. A dry run on the real E.104 and E.105 data, with no API calls, selects:

| Pool | Jev's mistakes | Flipped labels | Jev right | Total |
|---|---|---|---|---|
| Civil Comments | 689 | 230 | 919 | 1,838 |
| WildGuard prompts | 208 | 69 | 277 | 554 |
| WildGuard responses | 155 | 52 | 207 | 414 |
| **All** | **1,052** | **351** | **1,403** | **2,806** |

- **New review samples are needed for 524 rows:**
  - 155 of Jev's mistakes where the reviewer gave in every time;
  - all 351 flipped labels;
  - 18 Jev-right rows.

  That's at most about 2,700 Bedrock calls, roughly $0.20 at E.105's rates. This live step needs a check-in first.
- **E.109's GRPO candidates** are 6,109 training and 317 validation prompts.

### Live run: `e108-v1-2026-11-22` (Bedrock, $0.10)

1,166 new review samples by gpt-oss-20b, all returned and 1,165 parsed, for **$0.1037** (ledger). That's about half the estimate. The dry run's fake reviewer sampled more because it rarely solved anything.

| Pool | Jev's mistakes re-sampled → now solved | Flipped labels → solved | Jev right → solved |
|---|---|---|---|
| Civil Comments | 97 → 66 | 230 → 227 | 16 → 9 |
| WildGuard prompts | 37 → 24 | 69 → 68 | 1 → 1 |
| WildGuard responses | 21 → 13 | 52 → 52 | 0 → 0 |

- **8 more tries recover most of the "gave in every time" rows:** 103 of 155 of Jev's mistakes now have a correct review trace by the reviewer itself. 52 still don't, and they are dropped.
- **Exported:** both arms on the same items, 2,592 training and 151 validation rows each (2,743 of 2,806 selected; 63 dropped for no correct trace). Also 6,109 / 317 GRPO prompts for E.109.
- **Checks:** both arms hold identical item sets; train and validation share no item; no item appears in E.102's test plan.
- **Files:** `data/posttrain/part2_round2_v1/`, manifest in `benchmarks/posttrain/part2_round2_v1/manifest.json`, report in `results/runs/e108-v1-2026-11-22/report.md`.
- **Next (GPU, on hold):** `scripts/run_e108.sh e106-v1-2026-11-04` on one H100, about 4–5 GPU hours.

## Result (2026-11-23): neither arm works

Trained on one Spot H100 (`e108-gpu`, us-central1-a) and evaluated inside E.106's run (`e106-v1-2026-11-04`, reviewers `consist_review` and `balanced_review`). That's the same frozen test items, the same vLLM 0.33.0 serving stack and the same difficulty scale as the base and round 1. Code: commit `eaedd0b`. GPU logs: `results/runs/e108-v1-2026-11-22/gpu_logs/`.

### Round-2 criteria (pre-registered)

| Criterion | consist_review | balanced_review |
|---|---|---|
| **R1** Override of Jev's wrong labels on solvable items (A+B) rises, interval above 0; accept-when-right falls ≤ 2 points | no | no |
| **R2** Group A: Jev's wrong labels accepted falls, interval below 0 | no | no |
| **R3** Overrides more on solvable items than E.106's `sft_plain` control | no | no |
| R4 R1's override change above 0 in every pool (point estimate) | no | yes |
| R5 R1 holds for every held-out upstream model | no | no |
| R6 Cascade beats Jev alone | no | no |

A method works only if R1–R3 all pass, so **neither works**.

### The numbers (pooled, 1,058 of Jev's test mistakes, paired against the base)

| Reviewer | Solvable (A+B) overridden | Group A overridden | Group C overridden | Jev's right accepted | Cascade accuracy | Deference (right alone) |
|---|---|---|---|---|---|---|
| base | 62% (361/585) | 75% (273/364) | 8% (37/473) | 90% (930/1034) | 83% (992/1200) | +4.8 [−2.1, +11.6] |
| consist_review | 62% (364/584) | 76% (275/363) | 6% (27/473) | 91% (944/1033) | 84% (1003/1199) | +8.5 [+2.5, +13.7] |
| balanced_review | 63% (366/585) | 74% (271/364) | 6% (28/473) | 92% (951/1034) | 84% (1014/1200) | +7.9 [+1.7, +14.0] |

**Paired change against the base, points [95%]:**

| Reviewer | Jev's wrong overridden | Jev's right accepted | Cascade |
|---|---|---|---|
| consist_review | −0.6 [−3.3, +2.2] | +1.5 [−0.3, +3.3] | +1.0 [−0.8, +2.8] |
| balanced_review | −0.4 [−3.2, +2.5] | +2.0 [+0.4, +3.8] | +1.8 [−0.1, +3.7] |

Jev alone is 86.2%, so both cascades stay below it (R6).

- **Nothing moved on the items it was aimed at.** On group A, where the reviewer is right alone but gives in to Jev 25% of the time, both arms still give in about as often (24% and 26%). The +9-point group-A ceiling went unused.
- **It looks like round 1 again.** As with E.106's SFT and E.107's DPO, the trained reviewers accept slightly more of Jev's right labels, and the cascade gains a point or two. That's not significant for consistency training (+1.0 [−0.8, +2.8]) and borderline for balanced-proposal (+1.8 [−0.1, +3.7]), and it doesn't come from catching more of Jev's mistakes.
- **Neither data change made a difference:**
  - training toward the reviewer's own alone reasoning (consistency);
  - balancing Jev's right labels against flipped wrong ones (balanced-proposal).

  The two arms differ by under a point on every headline rate.
- **Measured deference rose, not fell** (+8.5 and +7.9 against the base's +4.8). The intervals overlap, so this is no evidence that training made deference worse, but it is clearly not better.
- **One consist_review review didn't parse,** so its denominators are 1,057 and 1,199. It changes nothing.

### What this means

- **SFT on the reviewer's own correct reasoning doesn't reduce deference,** whatever the data mix. That makes three SFT and DPO variants (E.106, E.107, E.108) with the same null result on Jev's mistakes.
- **Next in the plan: E.109 (GRPO) and E.110 (activation steering).** Whether to run them is a decision, not something this result implies.
  - **GRPO** rewards the gold label directly, so it isn't imitating traces. That's a different mechanism from what failed here, but it costs 6–15 GPU hours on 2 × H100.
  - **E.110's base steps** test whether a "give in" direction exists at all, without training.

### GPU cost and what went wrong

- **Machine time:** 03:22–11:57 UTC on 2026-11-23, plus about 3 minutes to copy results back. That's about 8.6 hours at about $6.60 an hour, **roughly $57**. The provider console is the authoritative record.
- **The work took 3.7 hours:** both SFT runs at about 50 minutes each, about 1 hour lost to three serving failures, and two evaluations of about 20 minutes each. It finished at 07:06.
- **The machine then sat idle for about 4.9 hours (about $32)** until the project's billing was cut off at about 11:57 and the machine was stopped.
- **The cause:** the run didn't shut the machine down at the end, and the alert ran on the laptop, which went to sleep. Billing was restored on 2026-11-23, and the results were copied and the machine deleted.
- **Fixes for the next GPU session:**
  - the driver powers the machine off after `E108 DONE` or on any failure;
  - a cost cutoff runs on the machine itself, not on the laptop.
- **Three serving fixes** were applied on the machine, all in the copied `go_e108.sh`:
  - vLLM 0.33.0 installed separately (`uv tool install --python 3.12 vllm==0.33.0`);
  - `/usr/local/cuda/bin` on the path;
  - `g++` installed, which the 2026-11-11 image lacks, for FlashInfer's startup compile.

## Reproduce

```
uv run python research_analyses/E108/step0_group_c.py                      # step 0: sample, sheet, and the rule's verdict
uv run python -m sysone.experiments.e108 --run-id e108-v1-<date>           # Bedrock, ~$0.20 (check in first)
scripts/run_e108.sh e106-v1-2026-11-04                                     # GPU: both arms, merge, evaluate
```
