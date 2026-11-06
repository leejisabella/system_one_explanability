# E.108: consistency training and balanced-proposal SFT

**Status (2026-11-05):** built and tested offline, nothing trained yet. The design is in [research_docs/15](../../research_docs/15_part2_round2.md). Step 0, the check on the ceiling assumption, is done; its result is below.

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
- **But WildGuard's group C becomes a target again,** and the write-ups report it as one.
- **This reopens one option:** a stronger model could write gold-reaching reasoning for these items, as training data only, never as a cascade stage. Whether to use it is a decision still to make.

**Caveats.**
- **The result sits exactly on the threshold.** One reading the other way changes the verdict.
- **There was one reader,** and it was a model, not a person. The classifications are easy to check: the sheet is in the run folder.
- **30 items** out of the 159 WildGuard group C items.

## Building the data (not run yet)

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

## Reproduce

```
uv run python research_analyses/E108/step0_group_c.py                      # step 0: sample, sheet, and the rule's verdict
uv run python -m sysone.experiments.e108 --run-id e108-v1-<date>           # Bedrock, ~$0.20 (check in first)
scripts/run_e108.sh e106-v1-2026-11-04                                     # GPU: both arms, merge, evaluate
```
