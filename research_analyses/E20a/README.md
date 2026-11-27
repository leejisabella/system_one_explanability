# E.20a: selective override, out of sample

**Status:** done 2026-11-24. Free: a re-analysis, with no API calls. It follows up [E.20](../E20/README.md). **Rule, fixed before computing:** [e20a.yaml](../../configs/experiments/e20a.yaml).

**Result: no.**
- **The rule's test:** the policy chosen on the training pools gains **−0.3 points over Jev alone [−1.8, +1.2]** on E.102's subsample (gpt-oss-20b on Bedrock, its first 2 alone answers).
  - With all 5 answers it gains +0.4 [−0.8, +1.6].
  - On the vLLM 20b it gains −0.2 with 2 answers and +0.4 with 5.
- **Versus the plain cascade:** it stops the cascade's loss (−1.7 for the same reviewer), but adds nothing over Jev.
- **E.20's ceiling doesn't carry over to new items.**
  - The best rule chosen *on the test items themselves* gains +1.2 [+0.3, +2.0].
  - But it overrides in different directions from the rule the training pools chose.

## Question

E.20 found that a policy using only the reviewer's own alone answers could in principle add +1.6 to +2.8 points over Jev. That figure is in-sample. Does any of it survive when the policy is chosen on other items?

## Setup

- **Policy:** for each pool and each label Jev proposed, one of three rules:
  - never override;
  - override if both of the reviewer's first two alone answers disagree with Jev (**unanimous**);
  - override if at least one of the two disagrees (**majority**, i.e. at least half).
- **Chosen on the training pools only.**
  - Inputs: E.104's Jev decisions and E.105's 20b alone tries. Most training items have exactly 2 tries, hence n = 2.
  - Override rates are measured within E.104's mix, whose items were chosen on Jev's correctness only. They are weighted by Jev's real error rate in each direction on the whole training pool.
- **Tested once** on E.102's subsample (400 per pool, pooled with equal weight), with the same reviewer and stack (Bedrock 20b) and the same n = 2.

## Results

**Chosen on training:**

| Pool | Jev proposed | Rule | Training gain: never / unanimous / majority (points) |
|---|---|---|---|
| Civil Comments | toxic | unanimous | 0 / +0.1 / −0.3 |
| Civil Comments | non_toxic | unanimous | 0 / +0.1 / −2.1 |
| WildGuard prompts | harmful | majority | 0 / +0.5 / +1.3 |
| WildGuard prompts | unharmful | never | 0 / −3.4 / −8.5 |
| WildGuard responses | harmful | unanimous | 0 / +0.2 / −0.4 |
| WildGuard responses | unharmful | never | 0 / −0.3 / −2.9 |

Even on training, the best estimated gain in any direction is +1.3 points (overriding Jev's "harmful" on WildGuard prompts). Elsewhere it is +0.5 or less.

**Test:**

| Variant | Gain over Jev [95%] | Fixed / broken: Civil, WG prompts, WG responses |
|---|---|---|
| **Bedrock 20b, 2 answers (the rule)** | **−0.3 [−1.8, +1.2]** | 32/35, 5/10, 4/0 |
| Bedrock 20b, 5 answers (unanimous = 5 of 5, majority = 3+ of 5) | +0.4 [−0.8, +1.6] | 23/20, 2/3, 3/0 |
| vLLM 20b, 2 answers | −0.2 [−1.6, +1.3] | 30/32, 4/8, 4/0 |
| vLLM 20b, 5 answers | +0.4 [−0.7, +1.5] | 22/16, 2/4, 1/0 |
| *In-sample best rule on the test items (optimistic)* | *+1.2 [+0.3, +2.0]* | *5/2, 8/4, 8/1* |

- **How often the rule overrides:** 67, 15 and 4 of the 400 items per pool. That is a sixth to a hundredth as often as the plain cascade.
- **The in-sample best rule overrides in other directions:**
  - Civil Comments: only Jev's "toxic", unanimous.
  - WildGuard prompts: only Jev's "unharmful", unanimous. Training chose the opposite direction, "harmful".
  - WildGuard responses: only Jev's "harmful", by majority.

## What it means

1. **A careful override policy stops the cascade's loss but doesn't beat Jev.**
   - Requiring the reviewer's alone answers to agree before overriding cuts the broken right labels a lot.
   - The fixes fall with them, so the result is a tie with Jev.
   - This is what E.114 found for routing by Jev's confidence: a cost lever, not an accuracy lever.
2. **E.20's ceiling is mostly in-sample optimism.** The gain is real only when thresholds and directions are fitted to the same items. Chosen elsewhere, the directions don't hold.
3. **On WildGuard, the training labels point the wrong way.**
   - The training pools' labels are GPT-4's; the test labels are human.
   - Training said to override Jev's "harmful" calls on prompts (Jev over-flags under GPT-4's labels). The test items favour overriding its "unharmful" calls.
   - That mismatch alone can explain the WildGuard part of the failure. It is the label caveat E.116 raised, and the reason E.23 (re-labelling) matters.
4. **This closes the last cheap lever from E.20.** The reviewer's own answers, used as carefully as we can choose out of sample, don't add accuracy to Jev.

## Caveats

- **The rule is coarse:** three rules on 2 answers, because most training items have only 2 alone tries. A finer threshold needs more tries per training item. That would be billed: about $1.30 for 3 more tries on the 6,075 items of E.104's mix, at E.102's rate of about $0.00007 per 20b call.
- **WildGuard has few Jev mistakes in the subsample** (39 and 30), so its per-pool results are noisy.
- **No extra signal was used.** Combining this with Jev's confidence is E.21 Part A's combined score as a router; as a flip rule it is untested here.

## Reproduce

```
uv run python -m sysone.analysis.e20a
```

- **Code:** `sysone.analysis.e20a`.
- **Tests:** `tests/test_e20_e21.py`.
- **Outputs:** `data/summary.json`, `data/report.md`.
