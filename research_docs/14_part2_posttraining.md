# 14. Part 2: post-training the reviewer to be robust to Jev's errors (E.104–E.108)

**Status (2026-11-04): built and tested offline; nothing has run live.**
- The training pools are frozen.
- Every runner, the evaluation and the analysis pass end-to-end dry runs on the real pools and on the real E.102 and E.103 runs.
- The GPU scripts (training, merging, serving) have **not** run on a GPU yet. The first GPU session starts with the smoke test in the [runbook](#runbook).
- **The reviewer being trained is Qwen3-14B** (open-weight, Apache 2.0), not Part 1's gpt-oss-20b. See [Why Qwen3-14B](#why-qwen3-14b) and the [deviations](#deviations) this caused.

Part 1 is in [13_cascade_deference.md](13_cascade_deference.md). Its results are in [research_analyses/E102](../research_analyses/E102/README.md) and [E103](../research_analyses/E103/README.md).

## Why Part 2 needs training

What Part 1 found:
- **The reviewer goes along with Jev's mistakes far more often than random ones** (E.102). On Civil Comments it accepts 60% of Jev's wrong labels, against 22% of random wrong labels.
- **Most of that gap is shared blind spots.** Jev's mistakes fall on items the reviewer gets wrong on its own. A smaller part, about +7 points at equal difficulty, is genuine deference: seeing Jev's answer pulls the reviewer toward it.
- **No prompt change helps** (E.103). Blind-first, two warnings and consider-the-opposite all left the reviewer catching the same share of Jev's mistakes, and the cascade no more accurate.
- **The shared blind spots don't move at all.** On items the reviewer never gets right alone, it accepts 91–100% of Jev's wrong labels under every protocol.

So the fix has to change what the reviewer *knows*, not how it is asked. Part 2 trains it.

## The idea, in plain terms

1. **Jev finds the hard items.** Jev and the reviewer stumble in the same places, so Jev's mistakes on a fresh pool point to the items the reviewer probably gets wrong too. Screening thousands of items with Jev costs about a dollar.
2. **The dataset's labels teach.** Every training example targets the human (or, for WildGuardTrain, GPT-4) gold label, **never Jev's**: Jev is wrong exactly on the items that matter.
3. **The reviewer keeps reasoning.** Qwen3 thinks before it answers, and training on bare labels would teach it to stop. So every example carries a reasoning trace that reaches the gold label. Where possible the trace is the reviewer's own (rejection sampling). The teacher (Qwen3-32B, thinking) writes one only where the reviewer never succeeds.
4. **A control arm separates "robust to Jev" from "just better at the task".** Training on gold labels will make the reviewer more accurate on its own, and a more accurate reviewer catches more of Jev's mistakes for free. So a second model is trained on **the same items, labels and kind of traces, but without Jev's proposal in the prompt**. Only the treatment arm's advantage over this control is the effect of training against Jev's errors.

The training data is in review format:
- an item with Jev's wrong label shown, where the target is to override it;
- an item with Jev's right label shown, where the target is to accept it;
- an item with no proposal, decided alone.

## The pipeline

| Step | What | Where it runs | Code | Cost |
|---|---|---|---|---|
| **E.104** | Jev screens 3 training pools; the training mix is saved | TypeSafe | `sysone.experiments.e104` | ≈ $1.4 TypeSafe |
| **E.105** | Reasoning traces for every example (reviewer, then teacher); paired datasets exported | GPU (vLLM) | `sysone.experiments.e105`, `scripts/serve_part2.sh` | GPU hours only |
| **E.106** | Save the bf16 base, SFT both arms, merge, serve, evaluate base and both arms | GPU (1 × 80 GB) | `sysone.posttrain.sft`, `scripts/serve_part2.sh`, `sysone.experiments.e106` | GPU hours only |
| **E.107** | DPO on the reviewer's own correct vs. wrong traces, from the SFT treatment checkpoint; evaluate | GPU | `sysone.posttrain.dpo`, `e106` | GPU hours |
| **E.108** | Generalization: held-out upstream models (in every evaluation) and a held-out domain (train without WildGuard responses) | GPU | `e106` with the `*_no_responses` configs | GPU hours |

Each step reads the previous step's saved run, so each one can be rerun or resumed on its own.

### E.104: training pools and the mix

The training pools are frozen like every other sample: text in gitignored `data/cascade/`, manifests in `benchmarks/cascade/`. They are **disjoint from every test sample**.

| Pool | Items | Source and labels | Kept apart from the tests by |
|---|---|---|---|
| `civil_toxicity_train_v1` | 5,545 (1,500 per identity × label cell; the no-identity toxic cell has only 1,045) | Unused Jigsaw *test-set* Civil Comments, human labels | Excluding every E.101 and E.102 item. As before, no comment from Jigsaw's train IDs or `google/civil_comments`' train split, which TypeSafe's "Jev boost" sets include. |
| `wildguard_prompt_train_v1` | 5,000 (1,250 per adversarial × label cell) | WildGuardTrain, **GPT-4 labels** (training only) | Dropping any prompt that appears in WildGuardTest (none did at the pinned revision) |
| `wildguard_response_train_v1` | 5,000 (1,250 per cell) | WildGuardTrain responses, shown with their prompts | Same |

WildGuardTrain has no annotator agreement, so its items carry `annotator_agreement: unknown`.

**The mix** (`sysone.posttrain.mix`) puts each training item into exactly one group:

| Group | Items | Format | Target |
|---|---|---|---|
| `jev_wrong` | every item Jev gets wrong | review, with Jev's label shown | override, with the gold label |
| `jev_right` | `jev_right_per_wrong` (2) × as many Jev-right items | review, with Jev's label shown | accept |
| `matched_wrong` | off by default: Jev-right items given the other label | review | override |
| `alone` | `alone_per_wrong` (1) × as many other items | alone, no proposal | the gold label |

5% of items are held out by hash for validation loss. The real test set is E.102's pools.

**The main knob is `jev_right_per_wrong`.** At test time Jev is right 75–92% of the time. If the training mix overrides too often, the model learns to override, and its acceptance of Jev's right labels falls. That's the contrarian trade-off E.103 warned about. 2:1 is a starting point; if criterion 1 fails on acceptance, raise it.

**Contamination check.** The E.104 report sets Jev's error rate on each training pool against its rate on the matching E.102 test pool. A much lower rate on a training pool would suggest Jev was trained on it. Then that pool's Jev mistakes would be unusually few and atypical, and the pool should be replaced.

### E.105: reasoning traces and the datasets

**Examples to trace:**
- every review-format mix row;
- every mix item once in alone format, which is what the control arm trains on.

At the default settings that's about 17,000 examples.

**Sampling (rejection sampling):**
- The reviewer (`base_local`: the untrained Qwen3-14B, served by vLLM) samples each example in rounds. The traces must be its own, and Bedrock serves Qwen3 only without thinking, so this runs on the GPU.
- It stops once it has a correct trace and either a wrong one (for a DPO pair) or 2 tries. It goes up to 4 tries on hard examples.
- Then the teacher (`teacher_local`: Qwen3-32B with thinking, served on its own, fp8 if memory is tight) samples (up to 2 tries) every example the reviewer never solved. Run the two as `--steps traces` and `--steps teacher`, each with its own server up.

A trace counts only if:
- it reaches the gold label, with a verdict that agrees in review;
- it parsed and wasn't cut off;
- its reasoning text was saved (logged since E.103).

**The report is itself a result.** It counts, per pool and group, the examples solved by the reviewer, by the teacher, or by neither. The unsolved `jev_wrong` examples are the blind spots this method can't teach.

**Export** (`data/posttrain/<dataset>/`, gitignored; manifest with hashes committed in `benchmarks/posttrain/<dataset>/`):

| File | What | Used by |
|---|---|---|
| `sft_review.{train,val}.jsonl` | Treatment arm: the mix as built (review and alone formats) | E.106 `sft_review` |
| `sft_plain.{train,val}.jsonl` | Control arm: the same items and gold labels, all in alone format | E.106 `sft_plain` |
| `dpo_review.{train,val}.jsonl` | The reviewer's own correct trace (chosen) against its own wrong one (rejected) for the same prompt; in review, a trace that deferred to the wrong proposal is preferred as the rejected one | E.107 |

**Both arms get exactly the same items.** An item is exported only if both arms have a usable trace for it, so the arms differ only in whether a proposal is shown.

Two datasets are exported:
- `part2_all_v1`: every training pool;
- `part2_no_responses_v1`: without WildGuard responses, for E.108's held-out domain.

**The records** (`sysone.posttrain.formats`) are TRL conversational records. The prompt is built by the same `Reviewer` code as the evaluation prompts, so a test prompt is byte-identical to the training prompts of its kind (a test checks this). The completion is one assistant turn:
- `thinking`: the trace. Qwen3's chat template reads it as `reasoning_content` and renders it as `<think>…</think>` before the answer; `hf.render` renames the key per family (`train_config.FAMILIES`);
- `content`: the canonical JSON answer.

### E.106: SFT, serving and evaluation

**Training** (`sysone.posttrain.sft`, `configs/posttrain/sft_{review,plain}.yaml`):
- LoRA on Qwen3-14B (dense): r 8, alpha 16, every linear layer.
- Learning rate 2e-4, 1 epoch, effective batch 16, max length 4,096.
- The loss falls on the reviewer's turn only.
- Prompts are rendered with Qwen3's own chat template with thinking on (`enable_thinking`), as vLLM renders them at evaluation.
- Training needs one 80 GB GPU (A100 or H100), roughly an hour or two per arm at these sizes (estimate). A 48 GB GPU may do with a smaller per-device batch.
- gpt-oss stays supported (`family: gpt-oss` in a training config), for a later comparison with Part 1's reviewer.
- `--check` validates a config and its dataset against the committed manifest without a GPU. Run it before every GPU session.

**The base model is re-measured on the same stack.** A fine-tuned model can't run on Bedrock, and a different server can shift behavior: numerical precision, the chat template, the reasoning setting. So E.102's Bedrock numbers are not the comparison point. Instead:
- the **base model is saved in bf16** (`--save-base`), like the trained checkpoints (merged adapters). Qwen3 is bf16 on the Hub already; saving it pins the exact weights and sampling settings that every checkpoint then shares;
- **every reviewer, the base included, is served by the same vLLM** and evaluated by the same runner.

**Evaluation** (`sysone.experiments.e106`, `configs/experiments/e106.yaml`), on **E.102's frozen test plan**:
- **The base reviewer** (`base_local`) decides alone 5 times per item. That record is the difficulty scale for every reviewer, so the scale doesn't drift as training changes the model's own accuracy.
- **Every reviewer** reviews:
  - Jev's wrong labels, the matched random wrong labels and the subsample's correct labels, exactly as in E.102;
  - 150 wrong decisions per pool from each **held-out upstream model** (E.103's Llama 4 Maverick, Qwen3-32B, Nova Lite). No model was trained on their mistakes.
- **Trained reviewers** also decide each item alone once, for their own accuracy.
- Serve one model at a time and pass `--reviewers <alias>`. One run ID holds all reviewers, and each resumes on its own.

**The analysis** (`sysone.analysis.e106`) reports, per reviewer and pool:
- override-when-wrong and accept-when-right, always together;
- cascade accuracy and alone accuracy;
- deference at equal difficulty;
- acceptance of Jev's wrong labels on the base's never-right items (the shared blind spots);
- override by annotator agreement;
- harmful content let through;
- each upstream model's wrong labels overridden.

It also gives paired changes against the base and against the control arm, on the same items, and evaluates the pre-registered criteria below as pass or fail.

### E.107: DPO

- `sysone.posttrain.dpo` with `configs/posttrain/dpo_review.yaml` trains a fresh LoRA adapter from the **merged SFT treatment checkpoint**. Its reference model is that checkpoint with the adapter switched off.
- Pairs are on-policy: both the chosen and the rejected trace are the reviewer's own.
- Settings: learning rate 5e-6, beta 0.1.
- Evaluate as E.106 with `--reviewers dpo_review`. It is compared against the base and against `sft_plain`.

**RL is not built.** The natural next step is GRPO with a reward of 1 for the gold final label, Jev's proposal in the prompt, and a penalty for an inconsistent verdict. Build it only if SFT and DPO leave a gap on criterion 1.

### E.108: generalization

1. **Held-out upstream models:** criterion 4, in every evaluation.
2. **Held-out domain:**
   - Train `configs/posttrain/sft_{review,plain}_no_responses.yaml` on `part2_no_responses_v1`, which has no WildGuard responses.
   - Evaluate them (`sft_review_no_responses`, `sft_plain_no_responses`) on all three test pools.
   - The WildGuard responses pool is then out of domain. Read its numbers on their own: does the effect carry over to judging responses when training only ever showed prompts and comments?
3. **Later:** Jev's wrong labels with fluent, persuasive rationales. This needs the "Rationales" experiment from research_docs/13.

## Pre-registered success criteria

These are fixed **before any trained model is evaluated**, and encoded in `configs/experiments/e106.yaml` (`criteria`) and `sysone.analysis.e106.evaluate_criteria`. Changing them after seeing results must be recorded as a deviation.

A trained reviewer succeeds if, on E.102's test pools, compared with the base on the same stack and the same items:

| # | Criterion | Rule |
|---|---|---|
| 1 | **Improves the trade-off, not just contrarian** | Pooled: Jev's wrong labels overridden rises (paired 95% interval above 0), and Jev's right labels accepted falls by at most 2 points |
| 2 | **Beats the cheap protocols** | In every pool, overriding rises by more than the best E.103 protocol's change: Civil Comments +2.4, WildGuard prompts +1.2, WildGuard responses +6.9 points for gpt-oss-20b; for Qwen, re-measured on the base Qwen ([Deviations](#deviations), 2) |
| 3 | **Makes the cascade pay off** | Pooled: cascade accuracy beats the better of Jev alone and the reviewer alone (paired interval above 0) |
| 4 | **Transfers to held-out upstream models** | Criterion 1's rule, for each of Llama 4 Maverick, Qwen3-32B and Nova Lite |
| 5 | **Beats the control arm** | Pooled: the treatment overrides more of Jev's wrong labels than the control arm trained on the same items without proposals (paired interval above 0) |

Criterion 5 is the one that makes the result about Jev rather than about accuracy.

**Also reported, with no pass/fail:**
- deference at equal difficulty;
- acceptance on the never-right items;
- harmful content let through;
- override on high- vs. low-agreement Civil Comments items.

If many blind spots are label noise (low-agreement items), no training fixes them. That should read as a ceiling, not as a failure.

## Why Qwen3-14B

Decided 2026-11-04, before E.104 ran. The design above was first written for gpt-oss-20b, Part 1's reviewer.
- **It's open-weight (Apache 2.0) and dense, with thinking that can be switched on.** LoRA on 14B fits one 80 GB GPU with room to spare, and the pinned 2025 training libraries support Qwen3.
- **Its traces can be its own.** Rejection sampling needs the model being trained to write the traces. Self-hosted on vLLM, Qwen3 thinks and returns its reasoning; Bedrock serves Qwen3 only without it.
- **It tests Part 1 on a second reviewer.** E.106's first step evaluates the untrained Qwen on E.102's test plan exactly as gpt-oss was, so Part 1's measures (deference at equal difficulty, error overlap, harmful let through) come out for a second, reasoning reviewer from another family, as E.101 and E.102 asked for. Part 1's gpt-oss results stand as they are.
- **Expect more shared blind spots.** On Civil Comments, Qwen3-32B's errors overlapped with Jev's more than gpt-oss's did (phi 0.69 vs. 0.47, E.102), perhaps because TypeSafe's "Jev boost" sets were embedded with a Qwen model. A Qwen reviewer may catch fewer of Jev's mistakes before training, which leaves more for training to fix, and is itself a result.

## Deviations

Recorded 2026-11-04, before any trained model was evaluated, because the reviewer changed (also in `configs/experiments/e106.yaml`):
1. **Criterion 4 counts only upstream models from another family than the reviewer:** Llama 4 Maverick and Nova Lite. Qwen3-32B is still reviewed and reported, as a same-family check.
2. **Criterion 2's bar is re-measured on Qwen.** The pre-registered bars were the best E.103 protocol's gain for gpt-oss-20b. For Qwen, E.103's four protocols are rerun with the untrained Qwen (`configs/experiments/e103_qwen.yaml`) after E.106 has evaluated `base_local`: its standard reviews, difficulty record and blind-first answers are `base_local`'s E.106 rows, and the plan is still E.102's. Setting `criteria.e103_bar_run` in `e106.yaml` to that run makes the analysis take the bars from it (the best protocol's paired change per pool, computed as for gpt-oss, whose bars a test reproduces exactly). This must happen before any trained reviewer is evaluated. Until it's set, the gpt-oss bars apply and the report marks criterion 2 provisional.
3. **E.105's traces come from the GPU, not Bedrock.** Its cost is GPU hours instead of about $3–4 on Bedrock.

## Runbook

Check in before every live step: E.104, E.105 and the first GPU session.

**On the laptop (the one paid step, E.104):**

```bash
uv run pytest                                                      # offline
uv run python scripts/preflight_part2.py                           # Jev only, cents; the self-hosted reviewer and teacher are checked on the GPU
uv run python -m sysone.experiments.e104 --run-id e104-v1-<date>   # Jev screen + mix; read report.md (contamination check, mix counts)
# set source_run in configs/experiments/e105.yaml to the E.104 run
```

**On the GPU machine** (clone the repo; copy `data/cascade/`, which is gitignored, or rebuild it with `sysone.cascade.items build`; copy `results/runs/e102-v1-2026-11-04`, `e103-v1-2026-11-04` and the E.104 run):

```bash
uv sync --group train
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
uv run --group train python -m sysone.posttrain.sft --save-base checkpoints/qwen3-14b-base
# 1. E.105 traces: the base reviewer's own, then the teacher's. One server at a time.
scripts/serve_part2.sh checkpoints/qwen3-14b-base qwen3-14b-base &
uv run python scripts/preflight_part2.py --local base_local        # the reply parses and its reasoning comes back
uv run python -m sysone.experiments.e105 --run-id e105-v1-<date> --steps traces
# stop the server
scripts/serve_part2.sh Qwen/Qwen3-32B qwen3-32b-teacher 8000 --quantization fp8 &
uv run python scripts/preflight_part2.py --local teacher_local
uv run python -m sysone.experiments.e105 --run-id e105-v1-<date> --steps teacher
# stop the server
uv run python -m sysone.experiments.e105 --run-id e105-v1-<date> --steps export   # read report.md (unsolved examples)
uv run python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml --check
uv run python -m sysone.posttrain.sft --config configs/posttrain/sft_plain.yaml --check
git add benchmarks/posttrain && git commit                         # the dataset manifests, before any training
# 2. Smoke test: set max_steps: 5 in a copy of sft_review.yaml, train, merge, serve, and send one request
# 3. Train both arms
uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml
uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_review.yaml --merge
uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_plain.yaml
uv run --group train python -m sysone.posttrain.sft --config configs/posttrain/sft_plain.yaml --merge
# 4. Evaluate, one model at a time (base first: its alone record is the difficulty scale)
scripts/serve_part2.sh checkpoints/qwen3-14b-base qwen3-14b-base &
uv run python -m sysone.experiments.e106 --run-id e106-v1-<date> --reviewers base_local
# 5. Criterion 2's bar, with the base still served: set source_run in configs/experiments/e103_qwen.yaml
#    to the E.106 run, run the four protocols, then set criteria.e103_bar_run in e106.yaml to this run
uv run python -m sysone.experiments.e103 --config configs/experiments/e103_qwen.yaml --run-id e103-qwen-v1-<date>
# stop the server; serve checkpoints/qwen3-14b-sft-review-merged as qwen3-14b-sft-review; --reviewers sft_review
# then sft_plain the same way. The report (results/runs/e106-v1-<date>/report.md) updates after each.
```

**Record GPU hours by hand** in the run's manifest or the write-up. The harness counts $0 per token for `local` models: `configs/budget.yaml` gives the `local` provider a $0 project cap, which lets $0 calls through and refuses anything priced. E.106 also caps Bedrock and TypeSafe at a cent, so an evaluation can't spend money by mistake.

## Code map

| Piece | File |
|---|---|
| Training pools (builders) | [src/sysone/cascade/items.py](../src/sysone/cascade/items.py): `civil_toxicity_train_v1`, `wildguard_prompt_train_v1`, `wildguard_response_train_v1` |
| Mix | [src/sysone/posttrain/mix.py](../src/sysone/posttrain/mix.py) |
| Training records | [src/sysone/posttrain/formats.py](../src/sysone/posttrain/formats.py) |
| E.104 / E.105 / evaluation runners | [src/sysone/experiments/e104.py](../src/sysone/experiments/e104.py), [e105.py](../src/sysone/experiments/e105.py), [e106.py](../src/sysone/experiments/e106.py) |
| Criterion 2's bar for Qwen | [src/sysone/experiments/e103.py](../src/sysone/experiments/e103.py) with [configs/experiments/e103_qwen.yaml](../configs/experiments/e103_qwen.yaml); `best_override_change` in [src/sysone/analysis/e103.py](../src/sysone/analysis/e103.py) |
| Evaluation analysis and criteria | [src/sysone/analysis/e106.py](../src/sysone/analysis/e106.py) |
| Training configs and checks | [src/sysone/posttrain/train_config.py](../src/sysone/posttrain/train_config.py), [configs/posttrain/](../configs/posttrain/) |
| SFT / DPO / merging (GPU) | [src/sysone/posttrain/sft.py](../src/sysone/posttrain/sft.py), [dpo.py](../src/sysone/posttrain/dpo.py), [hf.py](../src/sysone/posttrain/hf.py) |
| Self-hosted client (vLLM) | [src/sysone/clients/local.py](../src/sysone/clients/local.py); aliases `base_local`, `sft_review`, `sft_plain`, `dpo_review`, `*_no_responses` in [configs/models.yaml](../configs/models.yaml) |
| Serving | [scripts/serve_part2.sh](../scripts/serve_part2.sh) |
| Preflight | [scripts/preflight_part2.py](../scripts/preflight_part2.py) |
| Tests | [tests/test_part2.py](../tests/test_part2.py) |

## Risks and open questions

- **The GPU code is untested on a GPU.**
  - The training libraries are pinned to the 2025 series the scripts were written against: transformers 4.57, TRL 0.23, peft 0.17, torch 2.9 (the `train` group in `pyproject.toml`). Newer major releases changed APIs.
  - The smoke test should confirm three things: Qwen3's chat template renders the prompt as a prefix of the full conversation and keeps the `<think>` block (`hf.render` raises if not); the loss is masked to the completion; and a merged checkpoint serves in vLLM with its reasoning returned (`--reasoning-parser qwen3`).
- **Qwen3 may think for longer than gpt-oss.** The reasoning budget is still 4,096 tokens. Replies cut off at that limit aren't usable as traces and count as unparsed in evaluation. The E.105 report and the preflight show how often that happens; raise `reviewer_reasoning_max_tokens` (and `max_length` in training) if it's common.
- **The traces and training now share one context.** With the Qwen reviewer, traces, training and evaluation all go through Qwen3's own template, on vLLM and transformers. (With gpt-oss, the traces came from Bedrock, whose system context isn't visible.)
- **The reviewer and one held-out upstream model are both Qwen.** Criterion 4 now leaves Qwen3-32B out (see [Deviations](#deviations)); its numbers are still reported.
- **WildGuardTrain's labels come from GPT-4.** Some "Jev mistakes" there may be GPT-4 mistakes. The test labels stay human. If WildGuard numbers look odd, check how often the base reviewer, Jev and GPT-4 disagree on training items.
- **The teacher may solve few blind spots.** If Qwen3-32B also fails on most never-right items, the training data will be thin exactly where it matters. The E.105 report shows this before any GPU time is spent.
  - One option is STaR-style rationalization: show the teacher the gold label as a hint, then train on the trace without the hint. It's not built, because the trace can leak the hint.
- **Data volume.** The mix is sized by Jev's error count, about 1,400 Civil Comments, 500 WildGuard prompts and 400 WildGuard responses (estimate). If that's too little, the Civil Comments pool can't grow: toxic test-set comments are used up. WildGuardTrain has 80,000 more items.
- **One trained model per arm, one seed.** The paired intervals cover item sampling, not training randomness. If the result is close, train a second seed of each arm before drawing conclusions.
