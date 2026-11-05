# E.104 and E.105: Part 2's training data

**Status:** done 2026-11-04 (`e104-v1-2026-11-04`, $1.47 TypeSafe; `e105-v1-2026-11-04`, $2.76 Bedrock). **The training data is built and checked. Training as-is was decided 2026-11-04.**
- About half of Jev's mistakes dropped out of the training set, because no model ever reached the gold label on them.
- A check of those items found that they are not the reviewer's blind spots. They are confident disagreements with Civil Comments' annotators, often over a doubtful label.
- Training (E.106) waits for a GPU.

These two runs prepare data; they test no hypothesis. The design is in [research_docs/14](../../research_docs/14_part2_posttraining.md).

## The short version, in plain terms

**What we did.**
- **E.104:** Jev labeled three fresh training pools that overlap no test item (5,545 Civil Comments, 5,000 WildGuard prompts, 5,000 WildGuard responses), 3 times each. Its mistakes mark the items the reviewer probably gets wrong too.
- **The mix:** for every Jev mistake, we kept two items Jev got right and one item to be decided with no proposal.
- **E.105:** gpt-oss-20b, the reviewer itself, tried each example up to 4 times. We kept a reasoning trace that reached the gold label.
  - Where the reviewer never got there, gpt-oss-120b (the teacher) had 2 tries.
  - Every example was traced twice, with Jev's proposal (the treatment arm) and without it (the control arm). An item stays only if both arms have a correct trace, so the two arms train on exactly the same items and labels.

**What we found.**
1. **Jev's mistake rates on the training pools match the test pools.** Jev was wrong on 24.5% of Civil Comments, 7.5% of prompts and 5.9% of responses, against 25.0%, 9.7% and 8.4% on E.102's test pools.
   - If Jev had seen these items in its own training, it would be noticeably more accurate here.
   - The WildGuard rates are 2–2.5 points lower, so a small overlap with Jev's training data can't be ruled out. It isn't large.
2. **On items Jev gets right, the models almost always find the gold label too.** 98–99.5% of "Jev right" and "no proposal" items got a correct trace.
3. **On Jev's mistakes, about a third never got one.** Neither model reached the gold label in any of 12 tries (4 + 2, in each of the two arms) on:
   - 35% of Jev's Civil Comments mistakes;
   - 29% of its prompt mistakes;
   - 36% of its response mistakes.

   Counting the items solved in only one arm, which are dropped to keep the arms paired, 51% of Jev's mistakes made it into the training set. They were meant to be a quarter of the training examples and are 15%.
4. **The dropped items look like label disagreements, not blind spots.** On all 476 dropped Civil Comments items, every model try gave the same answer. A real blind spot would get the right answer at least now and then.
   - 71% of the dropped items are labeled non-toxic but the models call them toxic. In the kept items it's 31%.
   - The dropped labels rest on fewer people: 29% had 5 or fewer annotators, against 19% of the kept ones.
   - In a reading of 30 dropped items:
     - about 15 had a gold label that looks wrong or rests on a 5-to-5 split;
     - 9 were mild snark that the annotators let pass and the models call toxic;
     - 6 could go either way;
     - in none did the models clearly miss something.
5. **Almost every trace is the reviewer's own.** The teacher wrote 2–3% of the training traces, so the training data mostly reinforces reasoning the reviewer can already produce on the right path.

**An analogy.** To train a junior radiologist, we collect the scans their screener misread and ask the trainee to work out the right diagnosis, keeping their correct workings. On about a third of the misread scans, neither the trainee nor the senior consultant ever agreed with the answer key. When we pulled a sample, the answer key itself was often the problem: the scan had been graded by only four or five people, or the graders had split evenly. Teaching the trainee to match that key would teach them the graders' quirks, not better radiology.

## The decision: train as-is

We considered three options before any training ([discussion summary](#options-considered)). The choice was to train on the data as exported (option 1):
- the dropped items would mostly teach Civil Comments' lenient threshold, or repeat label noise, not resistance to Jev;
- the treatment and control arms still train on the same items, so the comparison stays fair;
- the training set stays at its full size (6,592 examples).

The remaining risk is that only 15% of examples teach "Jev is wrong, override it", so the reviewer could learn to agree with Jev more. E.106 measures exactly this, as override-when-wrong and accept-when-right (criterion 1).

## Results

### Jev on the training pools (E.104)

| Pool | Items | Jev's error rate (train) | On the test pool (E.102) | Mix rows (train / val) |
|---|---|---|---|---|
| Civil Comments | 5,545 | 24.5% | 25.0% | 5,149 / 279 |
| WildGuard prompts | 5,000 | 7.5% | 9.7% | 1,408 / 84 |
| WildGuard responses | 5,000 | 5.9% | 8.4% | 1,132 / 48 |

The mix is one "Jev wrong" : two "Jev right" : one "no proposal" per pool. The validation split keeps whole items apart from training.

### How many examples got a correct trace (E.105)

| Pool | Group | Items | Solved (some arm) | Never solved |
|---|---|---|---|---|
| Civil Comments | Jev wrong | 1,357 | 881 | **476 (35%)** |
| | Jev right | 2,714 | 2,673 | 41 (2%) |
| | No proposal | 1,357 | 1,327 | 30 (2%) |
| WildGuard prompts | Jev wrong | 373 | 264 | **109 (29%)** |
| | Jev right | 746 | 737 | 9 (1%) |
| | No proposal | 373 | 368 | 5 (1%) |
| WildGuard responses | Jev wrong | 295 | 188 | **107 (36%)** |
| | Jev right | 590 | 587 | 3 (1%) |
| | No proposal | 295 | 293 | 2 (1%) |

Per-arm counts, including who wrote each trace, are in `data/e105_report.md`.

### The exported datasets

| Dataset | File | Records (train / val) | Jev wrong / Jev right / no proposal |
|---|---|---|---|
| `part2_all_v1` | SFT with proposal (treatment) | 6,592 / 370 | 15% / 57% / 28% |
| | SFT without proposal (control) | 6,592 / 370 | the same items |
| | DPO pairs (correct vs. wrong trace) | 1,019 / 64 | 58% / 42% / – |
| `part2_no_responses_v1` (E.108) | SFT treatment / control | 5,604 / 328 each | 15% / 57% / 28% |
| | DPO pairs | 900 / 58 | 56% / 44% / – |

All five training configs pass `--check`: files match their manifest hashes and every record has the right shape. One SFT record and one DPO pair would be cut at 4,096 tokens.

The DPO pairs lean towards Jev's mistakes. A pair needs the reviewer to be right on some tries and wrong on others, which happens most on the items Jev finds hard.

### The dropped Civil Comments items

| Jev's mistakes | Items | Borderline (30–70% of annotators said toxic) | Gold says non-toxic | 5 or fewer annotators | Models gave the same answer every try |
|---|---|---|---|---|---|
| Kept | 881 | 56% | 31% | 19% | – |
| Dropped | 476 | 46% | 71% | 29% | **100%** |

The dropped items are *less* often borderline by the annotator share. That share is also less reliable when only 4 or 5 people voted: "0% toxic" from 4 annotators is common among the dropped items.

Our hand reading of 30 random dropped items (seed 105; every item, label and category in `data/civil_dropped_sample30.csv`):

| Category | Items | Examples |
|---|---|---|
| Gold label looks wrong, or is a 5-to-5 split; the models look right | 15 | Labeled non-toxic: "Trump is either a half-wit or a shrewd con", "Any person with half a brain knows…", "…the 99.9 percent of the freaks out there…". Labeled toxic at 5 of 10: a polite comment on a football player's salary, a question about gun deaths |
| Mild snark the annotators let pass; the models call it toxic | 9 | "put on your big boy pants", "Faux News is nothing but a wingnut propaganda machine" |
| Either label defensible | 6 | "WOW Jew hater nice", broad claims about Muslims, a sarcastic "…of giving a damn" |
| The models clearly miss something | 0 | – |

This is one reader's judgment, and toxicity is subjective. The point that doesn't depend on judgment is the unanimity: the models are never uncertain on these items.

## What this means for E.106

1. **The Civil Comments ceiling is lower than it looks.** E.102's Civil test pool almost certainly holds the same kind of items: Jev's "mistakes" that the reviewer will confidently "accept" because it, too, disagrees with the gold label.
   - Training on gold-label-reachable traces won't change those.
   - Criterion 2's Civil target (+2.4 points of override-when-wrong over the best E.103 protocol) may be hard to reach for this reason, and a shortfall there alone should be read with that in mind.
2. **Watch accept-when-right first.** With Jev's mistakes at 15% of the training examples, the first failure to look for is a reviewer that agrees with Jev more. Criterion 1 catches it.
3. **No change to the pre-registered criteria.** They were fixed before training, and this finding is reported beside them, not used to move them.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **1. Train as-is (chosen)** | Full size; every trace reached the gold label honestly; the arms stay paired | Jev's mistakes are 15% of examples; the hardest items are missing |
| 2. Rebalance to 1 : 2 : 1 | Matches the planned mix; a quarter of examples teach overriding Jev | Cuts the data to about 3,900 examples; shows Jev wrong far more than on WildGuard, which could push the reviewer to over-override |
| 3. Read a sample first | Free; tells label noise apart from blind spots | Small sample, subjective |
| (Rejected) Tell the teacher the gold label | Recovers the dropped items | Trains reasoning written to justify a fixed answer, including wrong labels |

We did option 3 first, and its result led to option 1.

## Setup

| | |
|---|---|
| E.104 | `e104-v1-2026-11-04`, commit `f89b4c7` (clean), 22:32–22:40 UTC. Jev `jev-1.14.0` (TypeSafe), 3 samples per item, 46,635 calls, 0 errors |
| E.105 | `e105-v1-2026-11-04`, commit `06941bb` (clean), 22:41 UTC – 01:08 UTC (2026-11-05). Reviewer gpt-oss-20b, teacher gpt-oss-120b (Bedrock, flex tier, reasoning effort medium). 38,036 traces |
| Rejection sampling | At least 2 reviewer tries per example, continuing up to 4 until there was a correct trace; then 2 teacher tries where the reviewer never reached the gold label |
| Datasets | `part2_all_v1` and `part2_no_responses_v1`; manifests and hashes in `benchmarks/posttrain/`; files in `data/posttrain/` (gitignored, regenerated by E.105's export step) |
| Incident | For 30 minutes (00:04–00:34 UTC), Bedrock stopped answering. All 8 workers hit the 300-second read timeout, and 8 calls failed. The run then carried on by itself; the cost was time only |

## Caveats

- **The 30-item reading is one person's judgment** on a small sample. The unanimity and the annotator counts are the stronger evidence.
- **We only looked closely at Civil Comments.** The WildGuard drops (29% and 36% of Jev's mistakes) weren't hand-checked. WildGuard labels come from a few annotators per item too, but its content is different.
- **Contamination is checked only indirectly**, by comparing error rates on training and test pools.

## Reproduce

```
uv run python scripts/preflight_part2.py
uv run python -m sysone.experiments.e104 --run-id e104-v1-2026-11-04
# set source_run in configs/experiments/e105.yaml
uv run python -m sysone.experiments.e105 --run-id e105-v1-2026-11-04
uv run python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml --check
uv run python research_analyses/E104_E105/make_figures.py   # data/ tables, no API calls
```
