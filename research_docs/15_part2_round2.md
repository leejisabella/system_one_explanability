# 15. Part 2, round 2: targeting the items the reviewer can get right (E.108–E.110)

**Status (2026-11-05): E.108's data builder, the round-2 criteria and step 0 are built and tested offline; step 0 has run (below). E.109 and E.110 are being built. Nothing has been trained.** The strategies were chosen with the user on 2026-11-05, after SFT (E.106) and DPO (E.107) both failed all five criteria. Check in before every live step.

Round 1 is in [14_part2_posttraining.md](14_part2_posttraining.md), and its results are in [the E.1xx summary](../research_analyses/E1xx_summary/README.md).

## Why a round 2

Round 1 trained the reviewer on its own correct reasoning, and it didn't catch more of Jev's mistakes. Splitting Jev's 1,058 test mistakes by how the untrained reviewer does on them alone (5 tries each, from E.106's base run) shows why:

| Group | Share of Jev's mistakes | Civil / WG prompts / WG responses | What goes wrong | Reachable by training on the reviewer's own outputs? |
|---|---|---|---|---|
| **A.** Right alone on the first try | 34% (364) | 281 / 41 / 42 | It gives in to Jev's wrong label 25% of the time: **deference** | Yes: it already has the right reasoning |
| **B.** Wrong on the first try, right at least once in 5 | 21% (221) | 155 / 42 / 24 | It's **inconsistent** | Yes, with a weaker signal |
| **C.** Never right alone (0 of 5) | 45% (473) | 314 / 81 / 78 | **Shared blind spots** | No: there's no correct answer of its own to learn from |

The code fixes the rule (`sysone.analysis.e106.base_groups`): an item counts as C only if all 5 tries parsed and none was right, as in E.106's never-right measure; with that rule the split is 364 / 221 / 473.

**Group C is treated as a benchmark ceiling, not a target** (decided 2026-11-05).
- We take these items to be largely contested or wrong gold labels. The evidence is E.105's reading of the Civil Comments items that were never solved: about half had a doubtful gold label and none was a clear model mistake.
- Training toward those labels would teach label noise. So round 2 aims at groups A and B and reports C separately.
- This assumption is **not yet checked on WildGuard**, where C is about half of Jev's mistakes. Step 0 below checks it.

**What round 1 got wrong about groups A and B.**
- **SFT's mix was 85% items Jev got right,** so it mostly learned to accept.
- **DPO did target Jev's mistakes,** but its pairs were long, nearly identical texts, and its loss never left chance.
- **Neither method was aimed at the moment of deference:** the reviewer knows the answer, sees Jev's label, and gives in.

**The ceiling.** If the reviewer stopped giving in on group A, it would catch about 46% of Jev's mistakes instead of 38%, about +9 points pooled. Gains on group B would add to that.

## The plan

| Step | What | Where it runs | Cost (estimate) |
|---|---|---|---|
| **0** | Read a sample of WildGuard group C items, to check the ceiling assumption | Laptop, no API | Free |
| **E.108** | SFT from the base, two arms on the same items: **consistency training** and **balanced-proposal SFT** | Bedrock (top-up traces), then GPU | ≲ $1 Bedrock; 4–5 GPU hours (~$10–20 of credits) |
| **E.109** | **GRPO** in review mode, from E.108's better arm | GPU, likely 2 × H100 | 6–15 GPU hours (~$20–60 of credits) |
| **E.110** | **Activation steering**: an interpretability study run alongside, on the base and every trained model | GPU (shares E.108's and E.109's sessions) | 2–4 GPU hours |

**Order:** step 0, then E.108 with E.110's base measurements in the same GPU session, then E.109, then E.110's measurements on the trained models. GPU dollar figures use the Spot H100 rate seen in round 1, about $2.50–4 an hour. The VM is stopped as soon as each pipeline ends.

### Step 0: is group C really a ceiling on WildGuard?

- **The sample:** 30 WildGuard group C items (15 prompts, 15 responses), drawn at random by hash from E.106's base run.
- **The reading:** for each item, record the text, the gold label, Jev's label and the reviewer's 5 alone answers. Classify each as *doubtful gold label*, *genuinely ambiguous* or *clear model mistake*, as E.105 did for Civil Comments.
- **The rule (fixed now):** if 10 or more of the 30 are clear model mistakes, the ceiling assumption doesn't hold for WildGuard.
  - In that case, WildGuard group C becomes a target again, and the write-ups say so.
  - That's also the one case where a stronger model may write gold-reaching reasoning to train on (see [What we decided not to do](#what-we-decided-not-to-do-and-why)). It would supply training data only, never a new stage in the cascade.
- Step 0 doesn't block E.108. The criteria below score groups A and B either way.

**Step 0 result (2026-11-05): the ceiling assumption does not hold for WildGuard.** 10 of 30 WildGuard group C items were clear model mistakes, exactly at the pre-registered threshold. They were all benign items both models flagged as harmful. So WildGuard's group C is a target again; on Civil Comments the assumption stands. The round-2 criteria are unchanged. See [research_analyses/E108](../research_analyses/E108/README.md#step-0-is-wildguards-group-c-a-benchmark-ceiling). Whether to add gold training data from a stronger model for those items is a decision still open.

## E.108: consistency training and balanced-proposal SFT

**The question:** if the reviewer is trained so that seeing Jev's label doesn't change its answer, does it stop giving in on the items it can solve?

### The two arms

Both arms train on **the same items with the same proposals shown**. They differ only in the reasoning used as the target.

| | Arm 1: **consistency training** (`consist_review`) | Arm 2: **balanced-proposal SFT** (`balanced_review`) |
|---|---|---|
| Prompt | Review format, with a proposal shown | The same |
| Target reasoning | **The reviewer's own reasoning from deciding the item alone,** with no proposal in view | **The reviewer's own review reasoning** that reached the gold label with this proposal in view |
| Target answer | The gold label, with the verdict it implies (accept or override) | The same |
| The idea | The answer shouldn't depend on the proposal at all. Train the reviewer to reason as if it weren't there (bias-augmented consistency training, Chua et al., 2024) | The proposal is right only half the time, so it carries no information. Train the reviewer to check it (synthetic anti-sycophancy data, Wei et al., 2023) |

Comparing the arms answers a second question: is it better to ignore the proposal, or to engage with it and check it?

### The training items

All items come from E.104's training pools, which share no item with the test set. Every target is a trace by **gpt-oss-20b itself**, never the teacher, so both arms train on reasoning the reviewer can already produce.

| Group | What it is | Proposal shown | Items | Where the traces come from |
|---|---|---|---|---|
| `jev_wrong` | Jev's mistakes the reviewer solved alone at least once in E.105 | Jev's wrong label | 1,052 (Civil 689, prompts 208, responses 155) | Arm 1: E.105's alone traces. Arm 2: E.105's review traces for 897; **new review samples for the other 155**, which never got a correct review trace |
| `matched_wrong` | Items Jev got right, shown the *other* label | A flipped, wrong label | ~350 | Arm 1: E.105's alone traces. Arm 2: **new review samples** |
| `jev_right` | Items Jev got right | Jev's right label | ~1,400 | Both arms: E.105's traces (alone for arm 1, review for arm 2) |

- **Half the proposals are wrong:** 1,052 + ~350 against ~1,400 right. Round 1 had 15% wrong. This is the main change from E.106.
- **Most wrong proposals are Jev's real mistakes.** The flipped labels are easy: on items it gets right alone, the reviewer gives in to a random wrong label only about 7% of the time. They're in the mix so that "this proposal is wrong" isn't the same thing as "this is one of Jev's labels".
- **The 155 items with no correct review trace are the clearest deference cases.** On them, the reviewer was right alone but gave in every time Jev's label was shown. They're kept in both arms. Arm 2 gets up to 8 new review samples for each, and any item still unsolved is dropped from **both** arms, so the arms stay paired. The report counts how many were dropped.
- **No alone-format examples.** Both arms are evaluated alone too, so a loss of accuracy alone would show up.
- **Group C items in training** (never solved in E.105, by reviewer or teacher) are excluded by construction, consistent with the ceiling assumption.
- **The exact counts** are set by the data builder and recorded in the dataset manifest before any training. The `jev_right` and `matched_wrong` items are drawn at random by hash, stratified by pool.

**New Bedrock calls:** review samples for the 155 deference items (up to 8 each) and the ~350 `matched_wrong` items (up to 4 each), about 2,500 calls at most. At E.105's rates that's under $1. The preflight projects it before the run.

### Training

The same recipe as E.106, except for the data, so a difference from E.106 comes from the data:
- **Starting point: the untrained base,** not E.106's SFT model, since that model made deference worse.
- **LoRA settings:** r 8, alpha 16, linear layers only. Learning rate 2e-4, effective batch 16, max length 4,096, loss on the reviewer's turn only, reasoning effort medium.
- **2 epochs, not 1.** The dataset is about 2,800 examples, so one epoch would be about 175 updates, fewer than E.106's 412, and E.106 barely moved the model. This is fixed now, before any training.
- **Seed:** 108.

### Evaluation

As E.106, with nothing new to build:
- **The test set:** E.102's frozen test plan, on the same vLLM stack, with each arm added to run `e106-v1-2026-11-04`. E.107 was evaluated the same way.
- **The comparison point:** the base's existing evaluation in that run, which also sets the group boundaries.
- **Each arm** reviews every test item once and decides it alone once, and also reviews the held-out upstream models' mistakes.
- **New model aliases:** `consist_review` and `balanced_review`.

## Pre-registered criteria for round 2

These apply to **E.108's two arms and to E.109.** They're fixed before any round-2 model is trained, and will be encoded in `configs/experiments/e106.yaml` (`criteria_round2`) and `sysone.analysis.e106`. All changes are paired with the base on the same items, with 95% bootstrap intervals.

**Group definitions** come from the base's alone record in `e106-v1-2026-11-04` and are fixed for every model. **"Solvable"** means groups A and B together, the 585 of Jev's mistakes the base gets right at least once alone (or has fewer than 5 parsed tries on).

| # | Criterion | Rule | Base value |
|---|---|---|---|
| R1 | **Catches more of the solvable mistakes, without becoming contrarian** (primary) | Pooled: Jev's wrong labels overridden on solvable items rises (interval above 0), *and* Jev's right labels accepted falls by at most 2 points | Override 61.7% (361/585); accept 89.9% |
| R2 | **Gives in less where it knows the answer** (primary) | Pooled, group A: Jev's wrong labels accepted falls (interval below 0) | 25.0% (91/364) |
| R3 | **Not just a more accurate model** (primary) | Pooled, solvable items: overrides more of Jev's wrong labels than the E.106 control arm `sft_plain`, which trained on gold labels with no proposals (interval above 0) | `sft_plain`: measured in the same run |
| R4 | **Holds in every pool** | R1's override change is above 0 (point estimate) in each of the three pools | – |
| R5 | **Transfers to held-out upstream models** | R1's rule for each of Llama 4 Maverick, Qwen3-32B and Nova Lite, on their wrong decisions | Measured in the same run |
| R6 | **Makes the cascade pay off** | Pooled: cascade accuracy beats Jev alone (interval above 0) | Base −3.5 points; Jev 86.2% |

- **A model works if R1, R2 and R3 all pass.**
- **R4–R6 say how far it generalizes and whether it's useful.** R6 is the strictest: the cascade's accuracy still includes group C.
- **Reported, with no pass or fail:**
  - group C (Jev's wrong labels accepted);
  - round 1's five criteria, for continuity;
  - deference at equal difficulty;
  - switching on each model's *own* right-alone items, as in E.106;
  - accuracy alone;
  - harmful content let through;
  - override by annotator agreement;
  - median reasoning length;
  - the two E.108 arms against each other (paired).
- **Power:** WildGuard group A has only 41–42 items per pool, so R2 is pooled and R4 asks only for the right sign.

## E.109: GRPO in review mode

**The question:** does rewarding the right final answer *in review*, with Jev's label shown, teach the reviewer to catch Jev's mistakes, including the inconsistent group B?

### How it works

GRPO (group relative policy optimization; Shao et al., 2024):
1. For each training prompt, the model writes 8 answers.
2. Each answer gets a reward.
3. Each answer is pushed up or down by how its reward compares with the average of the 8.

An item where all 8 answers score the same gives no signal. That's why group C, where all 8 are wrong, can't be learned this way, and why it's left out.

### Design

- **Starting point:** the E.108 arm with the larger R1 override change, if it passes R1. If neither arm passes R1, start from the base. This is fixed now.
- **Prompts:** review format, from E.104's training pools.
  - Jev's wrong labels and Jev's right labels, roughly 50:50.
  - Plus E.108's `matched_wrong` items, at the same share as in E.108.
- **Keeping only items that teach.** Before training, the starting model writes 8 answers to each candidate prompt on the GPU; this is cheap with vLLM.
  - Only prompts with 1–7 correct answers are kept.
  - All-wrong prompts (group C-like) and all-right prompts carry no signal, so dropping them saves compute.
  - The pre-pass also measures where the model starts.
- **Reward:** 1 if the reply parses, its final label is the gold label, and its verdict agrees with that label. Otherwise 0.
  - There's no reward for agreeing or disagreeing with Jev, and no length term.
  - The 50:50 mix and the KL penalty keep it from learning to override everything.
- **Settings:**
  - Group size 8, temperature 1.0, up to 2,048 completion tokens (E.106's median was about 400).
  - KL penalty β 0.04 against the starting model.
  - LoRA r 8 on the linear layers, learning rate 1e-5.
  - Up to 300 updates of 16 prompts, seed 109.
  - A 20-update smoke run may change only the learning rate. Any change is recorded before the main run.
- **Stop rules, fixed now.** Stop early if either of these holds over 2 checks in a row (one check every 25 updates):
  - **Turning contrarian:** on a held-out set of 200 Jev-right training prompts, accept-when-right falls more than 3 points below the starting model.
  - **No learning signal:** mean reward has been flat and the share of all-same groups is above 90%.

  The run also stops at 15 GPU hours.
- **Logged:** reward, the share of all-same groups, override rate on Jev-right prompts, reply length and KL.

### Infrastructure: the main risk

TRL's GRPO trainer writes answers with vLLM while it trains.
- **Memory:** a bf16 copy of gpt-oss-20b for training plus a second copy for vLLM is about 84 GB, more than one H100's 80 GB.
- **The plan:** 2 × H100 (`a3-highgpu-2g`), one training and one serving answers through TRL's `vllm-serve`. **This needs Spot H100 quota of 2 in us-central1, so check the quota before building.**
- **Fallback on one GPU:** TRL's colocated vLLM with sleep mode and a low memory share. It's untested for this model, so it's tried in the smoke run only if the 2-GPU quota isn't available.

### Implementation choices (built 2026-11-05, before any run)

`sysone.posttrain.grpo`, `configs/posttrain/grpo_review.yaml` and `scripts/run_e109.sh` add these details, which the design above didn't fix:
- **Balancing after the filter.** The filter drops far more Jev-right prompts (mostly all right) than Jev-wrong ones. So per pool, by hash, Jev-wrong and Jev-right prompts are cut to the same count, and `matched_wrong` to a third of it (`prepass.balance`).
- **The held-out 200 Jev-right prompts** come from the val split, topped up from train by hash if val has too few. Those are then removed from training.
- **"Flat" reward** means the mean reward over a check's 25 updates is within 0.02 of the previous check's. The no-signal rule can fire at update 75 at the earliest.
- **Contrarian baseline:** the stop rule compares against the starting model's held-out accept-when-right from the pre-pass. The same rate is also measured once on the training path (TRL's server) before update 1, with a warning if the two differ by more than 3 points.
- **The 15-hour cap** counts pre-pass and training wall hours × GPUs, not merging or evaluation.
- **TRL settings:**
  - a constant learning rate;
  - `loss_type: dapo` (TRL 0.23's default);
  - prompts pre-rendered at reasoning effort medium, because TRL passes no template arguments;
  - `max_prompt_length` off, since TRL's default of 512 would cut the policy text;
  - 2 completions per forward pass, with 64-step accumulation, so each update is 16 prompts × 8 answers.
- **The reward** parses the final channel with the evaluation's own `parse_review`, so a reply cut off in its reasoning scores 0.
- **Not pre-registered: which checkpoint to evaluate after an early stop.** By default it's the adapter at the stop, and a checkpoint is saved at every check. If a contrarian stop happens, decide before evaluating and record the choice.
- **vLLM must be importable in the training environment** (TRL's trainer imports its NCCL communicator), as well as served by `trl vllm-serve`. It isn't in `pyproject.toml` yet, because adding it would re-resolve the lock against torch. Settle the version in the smoke test.

### Evaluation

As E.108: alias `grpo_review`, the same run, the same round-2 criteria.

## E.110: activation steering (interpretability)

**The question:** is "giving in to Jev" visible as a direction inside the reviewer's activations? If so, does removing it stop the reviewer giving in, and did training remove it or only hide it?

This is a study, not a training method. It runs on top of the other experiments, using the base and every trained model.

### Finding the direction

- **Data:** training-pool items only, so the direction is never fit on the test set.
  - Jev's mistakes that the reviewer solved alone, where E.105 also has review traces with Jev's label shown.
  - On these, some review traces **gave in** and some **held** (382 + 127 + 96 = 605 items with at least one give-in trace).
- **Activations:** run each existing review trace back through the base model, with no new generation, and record the residual stream at every layer.
  - The main position is the last token before the final answer.
  - Also recorded: the mean over reasoning tokens, and the last prompt token. The last prompt token comes before any reasoning, so it shows whether giving in is decided before the model thinks.
- **The direction,** per layer: the mean over give-in traces minus the mean over held traces. This is contrastive activation addition (Panickssery et al., 2023).
- **Split:** 80% of items to fit, 20% held out, by hash.

### Three measurements

| | What | How | Read as |
|---|---|---|---|
| **1. Is it there?** | Does the direction separate give-in from held traces? | On held-out items, projection onto the direction, with AUC per layer. A logistic probe per layer as a check | **Found** if the best layer's held-out AUC is at least 0.70 |
| **2. Is it causal?** | Does removing it reduce giving in? | During review generation, subtract α × direction at the best layer (and its neighbors). α is chosen on held-out *training* items from {0.5, 1, 2, 4} × the direction's norm, and frozen before the test set. Then review test group A (364) with Jev's wrong label, and the Jev-right subsample (1,034) | **Causal** if acceptance of Jev's wrong labels on group A falls (paired interval below 0) and accept-when-right falls by at most 2 points |
| **3. Did training remove it?** | Is the direction weaker in the trained models? | The same traces, run through each trained model. Compare the give-in vs. held gap along the base's direction, and also each model's own direction | A smaller gap means training changed the mechanism; the same gap with less giving in means it changed the output only |

- **Control directions:** a random direction of the same norm, and a "proposal is present" direction (review prompt minus alone prompt), steered the same way. A real effect should beat both.
- **Tooling:** Hugging Face transformers with forward hooks, not vLLM. This is slower, but steering needs it. One H100 is enough for inference.
- **Caveats:** gpt-oss is a mixture-of-experts model, so the behavior may not live in one clean direction; a null result is informative too. The traces came from Bedrock's serving setup and are replayed through our own, the same mismatch round 1 had.

## What we decided not to do, and why

Decided with the user on 2026-11-05.

| Not doing | What it would have been | Why not |
|---|---|---|
| **Deference DPO with redesigned pairs** | DPO on pairs that differ only in giving in vs. holding, with more epochs and a loss weighted to the verdict | E.107's DPO on the same traces never left chance. Re-cutting the same data into new pairs isn't expected to fix that |
| **Self-consistency review** | The reviewer reviews each item several times and the majority verdict wins | Voting over the reviewer's own samples can't fix the reviewer: on Jev's mistakes, its samples are mostly wrong already |
| **Uncertainty-gated review / escalation** | The reviewer overrides only when its samples agree, and escalates unsure items to a stronger model or a person | It adds a third model to the cascade. The cascade stays Jev → gpt-oss-20b. A new model may be used only to make gold training data, never as a stage in the cascade |
| **Training toward group C** | Training on the never-solved items: answer-only SFT on gold labels, a teacher shown the gold label (STaR rationalization; Zelikman et al., 2022), or human-written reasoning | Group C is treated as a benchmark ceiling, so training toward its labels would teach label noise. **Reopened only if step 0 finds WildGuard's group C is real model mistakes,** and then only as gold training data from a stronger model |
| **More prompt variants** | New review instructions | E.103 tested four, and none changed what the reviewer catches |
| **Tuning round 1** | E.106 or E.107 with more epochs, a higher rank, or the expert weights | They were aimed at the wrong items, and tuning doesn't change that. E.108 changes the data instead and keeps the recipe |

**Deferred, not rejected:** the held-out-domain study first planned as E.108 (train without WildGuard responses and test on them). It runs on whichever round-2 method works, using the existing `part2_no_responses_v1` split.

## To build

| Piece | Where (planned) | For |
|---|---|---|
| Group A/B/C labels from the base's alone record | `sysone.analysis.e106` | Criteria, E.110 |
| Round-2 criteria R1–R6 | `sysone.analysis.e106.evaluate_criteria`, `configs/experiments/e106.yaml` (`criteria_round2`) | E.108, E.109 |
| E.108 data: new review samples, the mix, export of both arms with a manifest | `sysone.experiments.e108` (reusing E.105's sampler and `sysone.posttrain.formats`) | E.108 |
| Training configs | `configs/posttrain/consist_review.yaml`, `balanced_review.yaml`; aliases in `configs/models.yaml` | E.108 |
| GRPO: pre-pass, reward, trainer, stop rules | `sysone.posttrain.grpo`, `configs/posttrain/grpo_review.yaml`, `scripts/run_e109.sh` | E.109 |
| Steering: activation capture, directions, probes, steered generation | `sysone.interp.steering`, `scripts/run_e110.sh` | E.110 |
| WildGuard group C sample for step 0 | `research_analyses/E108/step0_group_c.py` | Step 0 |
| Tests for each, offline | `tests/test_part2_round2.py` | All |

## Risks and open questions

- **The ceiling assumption is the biggest one.** If WildGuard's group C is real model mistakes, about half of Jev's WildGuard errors are out of scope by our own choice. Step 0 checks this first.
- **E.108 may simply reach the +9-point ceiling on group A and stop there.** That would pass R1 and R2 and still leave the cascade below Jev (R6). It would be a clean result about deference, not a fix for the cascade.
- **Arm 1's reasoning never mentions the proposal.** The trained reviewer may learn to ignore proposals entirely, which is fine for its correctness but means it stops engaging with Jev. Arm 2 is the check.
- **GRPO may turn contrarian or collapse,** and needs 2 GPUs. The stop rules and the quota check cover both.
- **One seed per model,** as in round 1. If a criterion passes narrowly, train a second seed before drawing conclusions.
