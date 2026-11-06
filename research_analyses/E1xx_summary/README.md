# Cascade deference (E.101–E.107): what we found

**Status:** Part 1 done (E.101–E.103) and Part 2's first two training methods done (E.104–E.107), 2026-11-03 to 2026-11-05. **Part 2's training strategy is going back to the drawing board.** This page summarizes all seven experiments in one place. Each experiment's own write-up has the full numbers, intervals and methods; the links are in [the last section](#the-write-ups).

## The question

A cheap, fast moderation model (**Jev**, from TypeSafe) labels each item first. A second model, the **reviewer** (gpt-oss-20b, a reasoning model), sees the same item and policy plus Jev's label, and accepts or overrides it. This "cascade" only helps if the reviewer catches Jev's mistakes.

- **Part 1:** is the reviewer more likely to let Jev's mistakes through than random mistakes of the same kind? If so, why?
- **Part 2:** can we post-train the reviewer so that it catches more of Jev's mistakes?

**Datasets:**
- **Civil Comments**: toxicity, Jigsaw test-set comments only, since Jev may have trained on the train split.
- **WildGuardTest**: harmful prompts and harmful responses.

**Two terms used throughout:**
- **Deference:** the reviewer being swayed by Jev's label beyond what the item's difficulty explains.
- **Shared blind spots:** items that Jev and the reviewer both get wrong, even when the reviewer decides alone.

## The answer in five points

1. **The reviewer lets far more of Jev's mistakes through than random ones.** On Civil Comments it accepts 60% of Jev's wrong labels, against 22% of random wrong labels. On WildGuard, about two thirds of the harmful content Jev misses also gets past the reviewer, against about a tenth when a random error missed it (E.102).
2. **Mostly, that's because both models have the same blind spots, not because the reviewer is swayed.** Jev's mistakes fall on items the reviewer gets wrong on its own. Compared at equal difficulty, real deference is about **+7 points** (pooled, 95% interval +0.8 to +12.6). That's real, but it's the smaller part of the problem (E.101, E.102).
3. **Cheap prompt changes don't fix it.** Blind-first, two kinds of warning, and consider-the-opposite none clearly changed how often the reviewer overrides Jev's mistakes (within about ±3 points on Civil Comments, and no change clear of zero). On items the reviewer never gets right alone, it accepts 91–100% of Jev's wrong labels under every prompt (E.103).
4. **Training the reviewer on its own correct reasoning didn't fix it either.**
   - **SFT made things slightly worse:** the reviewer gave up its own right answer to Jev's wrong label more often, 32% against 25%.
   - **DPO brought the main rates back near the base,** but went no further. Whether it reduced deference depends on the measure.
   - **No method caught more of Jev's mistakes** than the untrained reviewer. All three trained models fail all five pre-registered criteria (E.106, E.107).
5. **The common reason is in how the training data is made.** SFT, DPO and RL all learn from the reviewer's own answers, so they need it to reach the right label at least sometimes. The shared blind spots are by definition the items where it never does. About a third of Jev's mistakes are like that (E.105). A strategy that can reach them needs a training signal that doesn't depend on the reviewer already being able to get there.

**An analogy we've used throughout.** A junior doctor diagnoses and a senior doctor reviews. When the senior agrees with the junior's mistakes:
- **Mostly,** it's because the senior would have made the same mistakes on those patients anyway.
- **Partly,** the senior is swayed by the junior.

Telling the senior "the junior is often wrong" didn't help. Neither did a refresher course built from the senior's own best write-ups: the cases the senior never gets right weren't in it.

## Part 1: is the reviewer more susceptible to Jev's mistakes?

### E.101: pilot (go/no-go)

300 items per dataset on Civil Comments and WildGuard prompts; $0.42.
- **GO on Civil Comments, NO-GO on WildGuard prompts.** Jev was wrong on 26% of Civil Comments but only 9% of WildGuard prompts, too few for the pilot to measure anything.
- **The main finding was a surprise: Jev's and the reviewer's errors overlap far beyond chance.** On Civil Comments, 53 of Jev's 78 errors were on items the reviewer also got wrong alone; independent errors would give about 19. The overlap holds on items the annotators agreed on, so it isn't label noise.
- **Showing Jev's confidence changed nothing measurable.** Only the proposed label moved the reviewer, so confidence was dropped from later experiments.
- **The cascade doesn't pay for itself here.** Reviewing takes 22–46% more reasoning tokens than deciding alone, and no confidence threshold makes the cascade beat its best single model.

### E.102: deference beyond shared difficulty (the powered study)

About 6,400 items across the three pools (3,000 Civil Comments, all 1,699 WildGuard prompts, all 1,707 WildGuard responses); $1.81.

| | Civil Comments | WildGuard prompts | WildGuard responses |
|---|---|---|---|
| Jev's error rate | 25% | 10% | 8% |
| Jev's wrong labels overridden by the reviewer | 40% | 30% | 29% |
| Random wrong labels (same kind and number) overridden | 78% | 88% | 91% |
| Harmful content Jev missed that the reviewer also let through | – | 68% (41/60) | 62% (29/47) |
| Same, when a random error missed it | – | 8% (5/60) | 11% (5/47) |

- **The gap is mostly shared blind spots.** On Civil Comments, 39% of Jev's mistakes are on items the reviewer gets wrong every time alone (5 of 5), against 11% of random mistakes.
- **Real deference, at equal difficulty: about +7 points** (pooled, [+0.8, +12.6]). The raw "flip rate" overstated it about threefold, so from E.103 on, the difficulty-matched estimate is the primary measure.
- **Removing deference entirely would help only a little.** On Civil Comments it would lower the share of Jev's mistakes that get through from 60% to about 54%. The rest is blind spots.
- **The cascade isn't better than its best single model.** It's as accurate as Jev on Civil Comments (77% vs. 76%), worse on WildGuard (88% vs. 90–92%), and costs 2.6–4.6 times as much.

### E.103: cheap fixes, before any training

The same items as E.102, under four alternative prompts; $2.25.

| Prompt | What changed | Effect |
|---|---|---|
| Blind-first | The reviewer answers alone before seeing Jev's label | Deference shrank (+7 → +3 points), but only because the reviewer accepted *random* mistakes more. It accepted Jev's mistakes just as often |
| Warning | "This model is often wrong" | Almost nothing |
| Calibrated warning | "Wrong about N% of the time" (Jev's real rate) | Increased deference on Civil Comments |
| Consider the opposite | Argue for the other label first | Almost nothing |

- **No prompt catches more of Jev's mistakes or makes the cascade more accurate.** On Civil Comments every change is within about ±3 points; on the smaller WildGuard pools the changes reach −6 points, all within noise.
- **On items the reviewer never gets right alone, it accepts 91–100% of Jev's wrong labels** under every prompt.
- **The lesson that carried into Part 2:** a single number can improve while nothing useful changes. Blind-first "reduced deference" by getting worse on random mistakes. From then on, every experiment reports deference next to override-when-wrong and accept-when-right.
- **Three held-out upstream models** (Llama 4 Maverick, Qwen3-32B, Nova Lite) labeled every item, to test later whether training transfers to new upstream models.

## Part 2: can post-training fix it?

### E.104–E.105: building the training data

Fresh training pools that overlap no test item: 5,545 Civil Comments, 5,000 WildGuard prompts and 5,000 WildGuard responses. $1.47 for E.104 and $2.76 for E.105.

- **E.104:** Jev labeled every training item three times. Its error rates matched the test pools (24.5% / 7.5% / 5.9%, against 25.0% / 9.7% / 8.4%). For every Jev mistake, the mix added two items Jev got right and one decided with no proposal.
- **E.105:** for every example, we kept a reasoning trace that reached the gold label, by rejection sampling.
  - The reviewer itself had up to 4 tries; gpt-oss-120b, the teacher, got 2 more on the ones it never solved.
  - Every example was traced twice: with Jev's proposal (the treatment arm) and without it (the control arm).
- **About half of Jev's mistakes dropped out.** 35% were never answered correctly in 12 tries, and more were solved in only one arm. So Jev's mistakes are 15% of the training examples, not the planned 25%.
- **The dropped items look like label disagreements, not blind spots the reviewer could learn.** On the 476 dropped Civil Comments items, the models gave the same answer on every try. 71% are labeled non-toxic but called toxic by the models. In a reading of 30 of them, about half had a doubtful gold label, and none was a clear model mistake.
- **We chose to train on the data as it was** (decided 2026-11-04).

### E.106: supervised fine-tuning (SFT)

LoRA on gpt-oss-20b, one epoch, on one GCP Spot H100. The untrained base and both trained arms were all evaluated on the same self-hosted stack, so any difference is training, not serving.

| Change from the base [95% interval] | SFT treatment (with Jev's proposal) | SFT control (no proposal) |
|---|---|---|
| Jev's wrong labels overridden | −2.0 [−4.9, +0.9] | −1.4 [−4.2, +1.2] |
| Jev's right labels accepted | **+2.3** [+0.7, +4.1] | **+2.7** [+1.0, +4.6] |
| Cascade accuracy | **+2.3** [+0.5, +4.2] | **+2.8** [+1.0, +4.7] |
| Switches to Jev's wrong label, on items it gets right alone (base 25%) | **32%** (+7.2 [+0.9, +13.6]) | 31% (+6.3 [−0.1, +12.8]) |

- **Training didn't help the reviewer catch Jev's mistakes.** It made the reviewer accept Jev a little more, right or wrong, and give up its own right answer more often.
- **Deciding alone, nothing changed:** about 81% agreement with Jev and 70–71% accuracy for all three models.
- **Showing Jev's proposal in training made no difference:** the two arms differ by under a point on the headline rates, and by 1–2 points, within noise, on the rest.
- **Cascade accuracy rose only because the reviewer agreed with Jev more.** It still doesn't beat Jev alone.
- **Both arms fail all five pre-registered criteria.**

### E.107: DPO (preferring right over wrong reasoning on the same item)

1,019 pairs of the reviewer's own right and wrong answers, 58% of them on Jev's mistakes, starting from the SFT treatment model.

| | Base | SFT treatment | **DPO** |
|---|---|---|---|
| Jev's wrong labels overridden | 37.6% | 35.6% | **37.5%** |
| Jev's right labels accepted | 89.9% | 92.3% | **93.4%** |
| Switches to Jev's wrong label, on items it gets right alone | 25% | 32% | **27%** |
| Jev's wrong labels accepted on items the base never gets right alone | 92% | 94% | **93%** |
| Cascade accuracy (Jev alone: 86.2%) | 82.7% | 85.0% | **86.0%** |

- **DPO brought overrides and switching back near the base,** but caught no more of Jev's mistakes. Deference at equal difficulty went the other way (+7.8 points, the highest of any model); none of these changes is clear of noise.
- **Its own training loss stayed at chance level** (0.69). The right and wrong answers in a pair are long and nearly identical, and one pass over 1,019 pairs with a small adapter didn't separate them.
- **It fails all five pre-registered criteria.**

## What we learned overall

1. **The core problem is shared blind spots, not deference.** Deference is real (+5 to +8 points) but small next to the share of Jev's mistakes the reviewer would make anyway. A reviewer that ignored Jev entirely would still pass about half of them.
2. **Neither prompting nor self-training reaches the blind spots.** Prompts can change how the reviewer reacts to Jev, not what it believes. Training on the reviewer's own correct answers can only reinforce what it can already do.
3. **Watch the headline numbers.** "Deference" improved under blind-first while nothing useful changed, and cascade accuracy improved under SFT while the reviewer got worse at its job. Always read override-when-wrong, accept-when-right and the switching rate together.
4. **Some "blind spots" are label problems.** On Civil Comments, many of Jev's "mistakes" that every model disagrees with have doubtful gold labels, with few annotators or a 5-to-5 split. This caps what any reviewer can show on that dataset.
5. **As built, the cascade doesn't pay off.** In every experiment, Jev → gpt-oss-20b is no more accurate than Jev alone beyond noise (on Civil Comments some versions edge past it by 1–2 points), at several times the cost and latency. A no-training majority vote of Jev, gpt-oss and Qwen does a little better on WildGuard (92% / 93%, against Jev's 90% / 92.5%) and ties on Civil Comments (77%) (E.103). All these gaps are within noise.

## Working assumption: the never-solved items are a benchmark ceiling

Decided 2026-11-05, before choosing Part 2's next strategy. On the test set, the base reviewer's results split Jev's 1,058 mistakes into three groups:

| Group | Share of Jev's mistakes | What it is |
|---|---|---|
| A. Right alone on the first try | 34% (364) | The reviewer knows the answer, but gives in to Jev 25% of the time (deference) |
| B. Wrong on the first try, right at least once in 5 | 21% (221) | The reviewer is inconsistent |
| C. Never right alone (0 of 5) | 45% (473) | Shared blind spots |

**We treat group C as a ceiling of the benchmark, not a target.** We take these items to be largely contested or wrong gold labels, so a reviewer shouldn't be trained to "catch" them. Training that pushes toward those labels would teach label noise. From here on, Part 2's methods aim at groups A and B, and its results are read with group C reported separately.

- **The evidence is on Civil Comments only.** In E.105, the models gave the same answer on every try for the dropped training items. In a reading of 30 of them, about half had a doubtful gold label and none was a clear model mistake.
- **Checked on WildGuard on 2026-11-05 (step 0): it doesn't hold there.** 10 of 30 test group C items were clear model mistakes, exactly at the pre-registered threshold; all were benign items both models flagged as harmful. So the assumption stands for Civil Comments only, and WildGuard's group C is a target again ([E108](../E108/README.md#step-0-is-wildguards-group-c-a-benchmark-ceiling)).
- **What it changes:** the objection that SFT, DPO and RL can't reach group C no longer rules them out, since group C isn't the goal. Fixing group A alone would raise override-when-wrong from 38% to about 46% at most.

## Directions for a new strategy

**Superseded 2026-11-05:** round 2 chose consistency training, balanced-proposal SFT, GRPO and activation steering, aimed at groups A and B. See [research_docs/15](../../research_docs/15_part2_round2.md). The table below is the earlier list of options, kept for the record.

These are options to weigh, not results. Each targets the blind spots, which the methods so far couldn't reach.

| Direction | Idea | Main risk |
|---|---|---|
| A stronger source of correct reasoning | A much stronger model, or people, write correct reasoning for the items the reviewer never solves; train on that | Reasoning the reviewer can't reproduce may not transfer. On mislabeled items, it trains in the label noise |
| Hints during RL | GRPO on the hard items with partial hints that are faded out, so the reviewer starts to earn reward where it never did | More engineering; the hints may leak into the learned behavior |
| A reviewer with different blind spots | Choose or combine reviewers by how little their errors overlap with Jev's (E.102 measured this: on Civil Comments, gpt-oss overlaps Jev less than Qwen does) | It may cost accuracy elsewhere; it doesn't fix any single model |
| Better labels first | Re-label the contested Civil Comments items before training or evaluating on them | Annotation cost; it changes the benchmark |
| A different cascade | Aggregate instead of review (the vote already matches or beats the cascade), or review only where the models disagree | It's no longer the Jev → reviewer setting Part 2 set out to fix |

## Cost

| Step | Where | Cost |
|---|---|---|
| E.101 | TypeSafe + Bedrock | $0.42 |
| E.102 | TypeSafe + Bedrock | $1.81 |
| E.103 | Bedrock | $2.25 |
| E.104 | TypeSafe | $1.47 |
| E.105 | Bedrock | $2.76 |
| Preflights | TypeSafe + Bedrock | under $0.05 |
| E.106–E.107 | GCP Spot H100, about 5.5 GPU hours plus the disk | roughly $13–22 (education credits; an estimate from uptime) |
| **Total** | | **about $8.80 in API calls, plus roughly $13–22 of GPU credits** |

The GPU machine and its disk were deleted on 2026-11-05, so nothing is billing now. The three trained LoRA adapters are kept locally in `checkpoints/`.

## Caveats that apply throughout

- **One reviewer model** (gpt-oss-20b) and one upstream model (Jev), on three moderation tasks. The held-out upstream models were used only as a check in Part 2.
- **Part 2 tested one configuration of each method:** one epoch, rank-8 LoRA on the linear layers only, and one seed. The expert-weight LoRA in OpenAI's recipe didn't fit on one H100. The results show these runs didn't work, not that SFT or DPO can't.
- **The difficulty-matched deference measure was added after E.102's run,** when the raw flip rate turned out to mix in difficulty. It was the primary measure from E.103 on.
- **GPU costs are estimates.** The education billing account shows the exact charges a day later.

## The write-ups

| Experiment | Write-up | One line |
|---|---|---|
| E.101 | [E101_pilot](../E101_pilot/README.md) | Pilot: errors are shared far beyond chance; GO on Civil Comments |
| E.102 | [E102](../E102/README.md) | Most of the gap is shared blind spots; deference about +7 points |
| E.103 | [E103](../E103/README.md) | No cheap prompt fixes it; Part 2's baseline is the standard prompt |
| E.104–E.105 | [E104_E105](../E104_E105/README.md) | Training data built; half of Jev's mistakes had no correct trace |
| E.106 | [E106](../E106/README.md) | SFT made the reviewer defer more; fails all five criteria |
| E.107 | [E107](../E107/README.md) | DPO landed back near the base, no better; fails all five criteria |

The designs are in [research_docs/13](../../research_docs/13_cascade_deference.md) (Part 1) and [research_docs/14](../../research_docs/14_part2_posttraining.md) (Part 2).
