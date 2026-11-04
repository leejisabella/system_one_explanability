# Project Status — start here

Last updated: 2026-11-04 (E.101–E.103 cascade experiments done on branch `e101-cascade-deference`; Part 2 waits for a GPU). Before that, 2026-10-28 (Pilot 0 passes on refund_v1; preflight passed; Pilot 1 ready, with Llama replacing the blocked Claude explainer). Read this first when picking the project back up.

## In one paragraph

We study whether explanations of opaque AI decisions give people verifiable grounds to contest those decisions (main question), and whether LLMs asked to explain Jev's decisions can be pushed into justifying bad or user-harmful decisions (stretch goal). The research plan is written up in [research_docs/](research_docs/README.md). The code so far is the infrastructure for the **first experiment, Pilot 0**: a frozen benchmark of 60 synthetic refund cases, a cost-capped harness for calling outside models, a wrapper that asks Jev for typed decisions through TypeSafe's native API, and a runner and analysis script that save and summarize results. The **Pilot 1** (Experiment 1) harness is also built: explainers decide the cases themselves and explain Jev's decisions, and a simulator model (gpt-oss-120b) scores each frozen explanation on hidden counterfactual variants. Every model is chosen and priced from official price lists, and the whole project must fit in **$50 on AWS, $50 on GCP and $5 on TypeSafe**, which the harness enforces across all runs. Jev is called as `jev-latest`; each run records the full probabilities and the exact version on every call, and halts if the version changes. **Pilot 0 ran live on 2026-10-28** (`pilot0-v0-2026-10-28`, $0.0137): Jev is stable and parses perfectly, but only 6 of 60 cases were customer-harming wrong decisions, short of the 20 needed, so a new benchmark, **refund_v1**, adds the defects Jev follows. **On refund_v1 Pilot 0 passes** (`pilot0-v1-2026-10-28`, $0.0160): 24 customer-harming wrong decisions, 100% parsed, 97% of variants unanimous (results in [research_docs/12](research_docs/12_first_experiments_roadmap.md#pilot-0-results-on-refund_v1-2026-10-28-go)). AWS has verified the account and every Bedrock model answers except **Claude**: AWS refuses the Marketplace subscription it needs for this account, so **Llama 4 Maverick replaces Claude Haiku 5.5** as Pilot 1's second explainer (Claude can join later as a third). The **preflight passed** on 2026-10-28 and projects Pilot 1 at about $0.31. The first live attempt (`pilot1-v1-2026-10-28`) halted: Llama reasoned past its 64-token decision cap (fixed with an answer-only system message) and the Gemini API key was on Google's free tier (Gemini now runs through **Vertex AI**, billed to the education credits). Both fixes passed the preflight, and the rerun **`pilot1-v1b-2026-10-28` completed** ($0.41; [research_analyses/E1_pilot1/](research_analyses/E1_pilot1/README.md)): the harness works, but without any explanation the simulator already predicts Jev on 90% of hidden variants, explanations restate the policy rather than Jev, and asked to explain Jev's wrong approvals the explainers justify them. Next: hidden variants where Jev departs from its policy (a purchase-window sweep) and the LLM judge.

## Refined direction: cascade deference (E.1xx, started 2026-11-03)

On branch `e101-cascade-deference`: Jev decides first and a reasoning model (gpt-oss-20b, open-weight) reviews its decision. Part 1 asks whether the reviewer accepts Jev's errors more often than random ones (the **flip rate**, measured against the reviewer alone). Design, datasets (WildGuardTest prompt harmfulness, Civil Comments toxicity) and go/no-go criteria: [research_docs/13](research_docs/13_cascade_deference.md).

**E.101 ran 2026-11-04** (`e101-v1-2026-11-04`, commit `8716b2d`, $0.42 of the $8 + $1 budget, 7,800 requests, 0 failures; write-up in [research_analyses/E101_pilot/](research_analyses/E101_pilot/README.md)).
- **GO on Civil Comments**, where Jev's error rate is 26% and the flip rate on its errors is 20% (5/25).
- **NO-GO on WildGuard**: Jev is too accurate (9% errors) and beats the reviewer (91% vs. 89%).
- **The main finding is shared errors, not deference.** Jev's errors fall on items the reviewer also gets wrong alone: 53 of 78 vs. 19 expected by chance on Civil Comments, 22 of 27 vs. 3 on WildGuard. This holds on items the annotators agreed on. So the reviewer overrides 36% of Jev's errors vs. 77% of matched random errors.
- **Deference beyond difficulty** (20% vs. 11% on items the reviewer gets right alone) points the expected way but isn't significant yet.
- **The cascade doesn't pay off.** Reviewing takes 25–45% more reasoning tokens than deciding alone, and no confidence threshold beats the better single model.
- **Safety:** on WildGuard, the reviewer let 7 of the 12 harmful prompts Jev passed through, and 0 of the 12 that random errors passed.

**E.102 ran 2026-11-04** (`e102-v1-2026-11-04`, commit `dcfc0d8`, $1.81, 40,179 requests, 0 failures; write-up in [research_analyses/E102/](research_analyses/E102/README.md)). Jev screened 3,000 Civil Comments items and all of WildGuardTest (prompts and responses). gpt-oss-20b reviewed Jev's mistakes, matched random mistakes and a random subsample. Qwen3-32B (no thinking) was a reference model.

In plain terms:
- **The reviewer accepts far more of Jev's mistakes than random ones.** On Civil Comments it accepts 60% of Jev's wrong labels and 22% of random ones. The pre-registered flip-rate contrast is +12 points [+7, +17].
- **The main reason is shared blind spots.** Jev's mistakes sit on the items the reviewer also gets wrong alone: 39% of them on items it never gets right, against 11% of random ones.
- **Compared at equal difficulty, real deference remains but is small:** about +7 points pooled [+0.8, +12.6], +4 on Civil Comments alone (not significant).
- **Jev is the hub of the error overlap.** It shares more mistakes with gpt-oss and with Qwen than those two share with each other. On Civil Comments, Jev–Qwen stands out.
- **Safety:** about two thirds of the harmful prompts and responses Jev misses also get past the reviewer (68% and 62%), against about a tenth when a random error missed them.
- **No cascade beats its best single model.**

**What changes:**
- Future E.1xx experiments use the difficulty-matched deference estimate as their primary measure.
- Part 2 has two problems to fix: deference, the smaller one, and shared blind spots, which resisting Jev can't fix.
- The natural next experiment is blind-first and the other cheap alternatives, measured at equal difficulty.

**E.103 ran 2026-11-04** (`e103-v1-2026-11-04`, commit `82c4b70`, $2.25 of a $6 cap, no Jev calls; write-up in [research_analyses/E103/](research_analyses/E103/README.md)). On E.102's exact items, gpt-oss-20b reviewed again under four cheap protocols: blind-first (it answers alone before seeing Jev's decision), a warning, a calibrated warning (Jev's real error rate) and consider-the-opposite. Llama 4 Maverick, Qwen3-32B and Nova Lite decided every pool item, as held-out upstream models for Part 2.

In plain terms:
- **No prompt change fixes the problem.** Under every protocol the reviewer overrides about the same share of Jev's mistakes (37–43% on Civil Comments against 40%), and cascade accuracy moves by at most ±3 points, never clearly.
- **Blind-first shrinks measured deference but not the harm.** Pooled deference drops from +7.0 to +2.7 points (interval includes zero). Jev's wrong labels still get through as often (63%); the reviewer now accepts more *random* ones, because it sticks with its first answer, right or wrong.
- **Warnings don't help; the honest one backfires.** "Wrong about 25% of the time" raised deference on Civil Comments (+12 points).
- **Shared blind spots are untouched:** 91–100% of Jev's wrong labels are accepted on items the reviewer never gets right alone, under every protocol.
- **About half or more of Jev's harmful misses still get through** (47–64%, against 62–68% with the standard prompt).
- **Best no-training pipeline:** a majority vote of Jev, gpt-oss and Qwen (77% / 92% / 93%), within noise of Jev alone.
- **The upstream models are fair held-out tests:** their error rates (24–28%, 12–13%, 9–10%) and overlap with Jev (phi 0.50–0.62) are close to gpt-oss's.

**What changes:**
- E.102's finding is not an artifact of prompt wording, so Part 2 needs training. Its baseline is the standard prompt (override 40% / 30% / 29% of Jev's wrong labels, accept 87% / 95% / 93% of its right ones, cascade 77% / 88% / 88%), with the vote as a second bar.
- Part 2 should train the reviewer to be *right* on the items Jev finds hard, not just to disagree with Jev, and should report deference next to override and accept rates, since deference can improve while nothing useful changes.
- Part 2 (E.104 on, post-training gpt-oss-20b) waits for a GPU.

## Part 2: post-training the reviewer (E.104–E.108, built 2026-11-04, not run)

On branch `e104-part2-posttraining`, which merges after `e101-cascade-deference`. The design, the pre-registered success criteria and the step-by-step runbook are in [research_docs/14](research_docs/14_part2_posttraining.md).

In plain terms:
- **Jev finds the reviewer's blind spots; the gold labels teach.** Jev screens fresh training pools, and its mistakes mark the items the reviewer probably gets wrong too.
- **Training examples show Jev's proposal** and target the gold label: override Jev when it's wrong, accept when it's right. Each example carries a reasoning trace that reaches the right answer, written by the reviewer itself where it can, otherwise by gpt-oss-120b.
- **A control arm** is trained on the same items and labels without Jev's proposal, so "robust to Jev" can be told apart from "just a better classifier".
- **The evaluation re-measures the base model on the same GPU stack** as the trained ones, on E.102's frozen test plan, with E.103's Llama, Qwen and Nova as held-out upstream models.

What exists:
- **Training pools, frozen and disjoint from every test sample:** `civil_toxicity_train_v1` (5,545), `wildguard_prompt_train_v1` (5,000) and `wildguard_response_train_v1` (5,000).
- **Runners:**
  - E.104: Jev screen and mix;
  - E.105: traces and paired dataset export;
  - E.106: evaluation of any set of reviewers, with the five pre-registered criteria computed as pass or fail.
- **GPU scripts:** SFT, DPO, saving the bf16 base, merging (the `train` dependency group), and vLLM serving (`scripts/serve_part2.sh`).
- **A self-hosted client** (`local` provider).
- **A preflight** (`scripts/preflight_part2.py`).
- **Tests:** offline, in `tests/test_part2.py`. Dry runs of E.104 → E.105 → `--check` and of the evaluation pass on the real pools and the real E.102 and E.103 runs.

Not done yet:
- **No live run.** E.104 would cost about $1.4 TypeSafe and E.105 about $3–4 Bedrock (estimates).
- **The GPU scripts have never run on a GPU.** The first session starts with the smoke test in the runbook.

## What exists

| Piece | Where | State |
|-------|-------|-------|
| Research plan (questions, framework, datasets, experiments, metrics, stretch goal, risks, references) | [research_docs/00–10](research_docs/README.md) | Written |
| Harness design, model choices with prices, cost controls, project budget | [research_docs/11_harness_and_infrastructure.md](research_docs/11_harness_and_infrastructure.md) | Written |
| Roadmap to the first experiments, Pilot 0 design, results layout | [research_docs/12_first_experiments_roadmap.md](research_docs/12_first_experiments_roadmap.md) | Written |
| Provider clients (TypeSafe, Bedrock incl. Converse and flex tier, Gemini, OpenRouter), cost guard with project caps, version lock, spending log, cache | [src/sysone/clients/](src/sysone/clients/), [src/sysone/budget/](src/sysone/budget/) | Built, tested offline |
| Benchmark generator, refund policy, renderer | [src/sysone/benchmark/](src/sysone/benchmark/) | Built, tested |
| Benchmark v0 (60 cases, 160 variants, template text) | [benchmarks/refund_v0/](benchmarks/refund_v0/) | Frozen; Pilot 0 ran on it |
| Benchmark v1 (74 cases, 188 variants; 24 incorrect-input cases with the defects Jev follows) | [benchmarks/refund_v1/](benchmarks/refund_v1/) | Frozen; the current benchmark |
| Jev decision wrapper (typed choice question, full probabilities) | [src/sysone/decision/jev.py](src/sysone/decision/jev.py) | Built, tested with a fake Jev |
| Pilot 0 runner and analysis | [src/sysone/experiments/pilot0.py](src/sysone/experiments/pilot0.py), [src/sysone/analysis/pilot0.py](src/sysone/analysis/pilot0.py) | Run live on refund_v0 |
| Pilot 1 runner, explainer and simulator wrappers, structured basis, analysis | [src/sysone/experiments/pilot1.py](src/sysone/experiments/pilot1.py), [src/sysone/explain/](src/sysone/explain/), [src/sysone/analysis/pilot1.py](src/sysone/analysis/pilot1.py) | Built, dry run passed |
| Results write-ups, one folder per experiment (E0 done, with figures) | [research_analyses/](research_analyses/README.md) | E0 written |
| Open decisions and setup tasks | [TODO.md](TODO.md) | Up to date |

Not built yet: the natural-language version of the cases (step 6b, to be `refund_v2`), the access layer that controls what an explainer may see, CAPE, and the stretch-goal harness.

## How to pick it up

```bash
uv sync                                                   # install
uv run pytest                                             # offline tests; no network, no cost
uv run python -m sysone.experiments.pilot0 --dry-run      # whole Pilot 0 pipeline with a fake Jev, no cost
uv run python -m sysone.experiments.pilot1 --dry-run --pilot0-run <pilot0 dry-run id>   # Pilot 1 with fake models
```

Each dry run writes a folder under `results/runs/`. It contains random decisions (the fake models ignore the case), so delete it afterwards.

## What's blocking, and what to do next

In order (details in [TODO.md](TODO.md)):

1. ✅ **AWS verified the account**; Marketplace permissions are on the IAM user and Anthropic's use-case form is approved.
   - ❌ **Claude is still blocked:** `GetFoundationModelAvailability` shows `agreementAvailability: NOT_AVAILABLE`, and calls fail with "not available for this account" (not billed), even from an admin in the console. Needs an AWS Support case (Account and billing). Llama 4 Maverick replaces it for now.
2. **Check credits and set budget alerts** in the AWS and GCP consoles (e.g. alerts at $40).
3. **Commit**, so each run's manifest names a clean commit.
4. ✅ **Preflight** (`uv run python scripts/preflight.py`, 2026-10-28, `preflight-20261028T035934Z`, $0.0018): every request kind for Jev, both explainers and the simulator parsed; Gemini accepted thinking level `minimal`; gpt-oss ran on flex. Projected Pilot 1 cost ≈ $0.31. Smoke tests of `case_writer`, `case_writer_2`, `explainer_open`, `decision_alt` passed.
5. ✅ **Pilot 0 on refund_v1** passed (`pilot0-v1-2026-10-28`); results written up in [research_analyses/E0_jev_characterization/](research_analyses/E0_jev_characterization/README.md).
6. ✅ **Pilot 1** (`pilot1-v1b-2026-10-28`, $0.41) written up in [research_analyses/E1_pilot1/](research_analyses/E1_pilot1/README.md). The experiments were then renumbered to put this phase's questions first ([research_docs/05](research_docs/05_experiments.md#order-and-dependencies)): E1 Jev vs. explainer, E2 distinguishing similar targets, E3 information ablation, then E4 CAPE, E5 identifiability, E6 robustness and abstention, E7 socially sensitive cues, E8 generalization.
7. ✅ **Benchmarks with policy-departure variants** built 2026-10-28 ([research_docs/04](research_docs/04_datasets_and_benchmark.md#domains-built-2026-10-28)): `refund_v1_1` (refund_v1's mix plus 30 purchase-window sweep cases, 25–65 days) and five new domains with their own threshold sweeps, `moderation_v1`, `fraud_v1`, `tools_v1`, `routing_v1` and `eligibility_v1` (70 cases, 210 variants each). `refund_v2` stays reserved for the natural-language text. **Not run yet:** Pilot 0 on them (Jev only, about 1,400 variants; scaled from Pilot 0 on refund_v1, $0.016 for 188 variants, about $0.12 for the same samples per variant, to re-estimate before running), The runners, analyses, preflight and explainer/simulator prompts now work on every domain (dry runs pass on all six; refund prompts and the analyses of the real runs are unchanged).
8. ✅ **LLM judge built** 2026-10-28 (gpt-oss-120b): `uv run python -m sysone.experiments.judge --pilot1-run <id>` scores contestability and how explanations justify wrong decisions, into its own run folder. On `pilot1-v1b` it would judge 288 explanations for about $0.09 (estimate; the preflight now measures a judge call). **Before its numbers count:** human labels on about 50 explanations (`uv run python -m sysone.analysis.judge results/runs/<judge-run> --export-labeling sheet.csv`), scored with `--labels`; Pilot 1's 12 hand-coded justifications are a first check. The **"explains itself" confound** now has a measure (each answer's pull against the control, where it stands apart) and a screening step (`steps: [decisions]`) that shows whether a benchmark can identify it; on Pilot 1 it can't (0 Gemini variants).
9. **E1 full run** on that benchmark, with the judge, the explainers' `own_decision` explanations, and the pipeline comparison (cost, latency and quality of Jev alone, Jev + explanation, and the explainer alone; latency is now logged on every call, [research_docs/12](research_docs/12_first_experiments_roadmap.md#cost-and-latency-of-each-pipeline)) (and Claude as a third explainer if AWS unblocks it). **E2** needs rule-based targets in the harness (the refund policy with one parameter changed, e.g. `window_days` 45); **E3** needs probe results passed to the explainer (4 fixed, 4 or 8 chosen).
10. **Re-estimate the project budget** from the measured costs per call before each experiment ([research_docs/11](research_docs/11_harness_and_infrastructure.md#project-budget); allocations revised 2026-10-28 for the new E2 and E3).
11. **Step 6b** (natural-language cases with the case writer, about $0.10, frozen as `refund_v2`) can happen any time; its generator is not built yet.

## Things that are easy to forget

- **Cost limits:** $2 per API request, $25 per run, and project caps per provider across all runs (AWS $50, GCP $50, TypeSafe $5), set in [configs/budget.yaml](configs/budget.yaml). Calls that could exceed any of them are refused before sending; if any call actually costs more than $2, the run stops. Smoke tests count toward the project caps.
- **The project caps come from `results/`.** They sum every run's spending log there, so deleting run folders makes them undercount. Live runs must use the default `results/` (no `--results-dir`), and only one live run at a time: a run doesn't see another run's calls in flight. The AWS and GCP consoles are the authoritative record.
- **Jev version lock:** `jev-latest` moves when TypeSafe ships a release. A run locks to the first version that answers (saved as `model_version` in its manifest) and halts on any answer from another version, including on resume. To keep using a version after the alias moves, set the fixed ID (e.g. `jev-1.14.0`) in [configs/models.yaml](configs/models.yaml), or set `expected_model_version` in the experiment config.
- **Unpriced models are refused:** any model missing from [configs/pricing.yaml](configs/pricing.yaml), any `TODO` model ID, and any service tier (e.g. flex) without its own price.
- **Model families stay apart by role:** case writers and the simulator/judge never share a family with an explainer ([research_docs/11](research_docs/11_harness_and_infrastructure.md#model-access)). Check this when changing a model.
- **Flex tier:** the simulator and judge request Bedrock's flex tier; it is charged at the flex price only when Bedrock's response confirms flex, otherwise at the standard price.
- **`results/` stays local.** It is gitignored and not backed up anywhere. Write down the `run_id` of each run that matters.
- **Benchmarks are frozen.** Never edit `benchmarks/refund_v*/` by hand: loading checks a hash and will refuse an edited file. To change cases, freeze a new version under a new name.
- **Runs that matter:** `pilot0-v0-2026-10-28` (Pilot 0 on refund_v0, no-go) and `pilot0-v1-2026-10-28` (Pilot 0 on refund_v1, go; Pilot 1 builds on it).
- **Provisional name list:** the name-swap cases use first names from Bertrand & Mullainathan (2004) as a placeholder until the audit name list is chosen.
- **Pilot 0 does not reuse cached answers** from earlier runs (`use_cache: false`), so every new run makes fresh, billed calls. Resuming the same run never re-pays for saved decisions.
- **Unverified assumptions behind cost estimates** are listed in [research_docs/11](research_docs/11_harness_and_infrastructure.md#assumptions-still-to-verify).
- **Prices change.** Gemini prices rise on Dec 1, 2026 and Jan 1, 2027; the price table already uses the higher prices. Bedrock prices come from AWS's Price List files published 2026-10-12; re-check before large runs.

## Glossary

| Term | Meaning |
|------|---------|
| Case | One synthetic refund request, with its true record, the record shown to the model, the true outcome, defects and evidence |
| Case type | correct, incorrect_input, ignored_evidence, irrelevant_cue, ambiguous, hard_negative |
| Variant | One version of a case shown to Jev: the original, or an edited version (correction, name swap, and so on) |
| Sample | One repeat of the same variant, used to measure stability |
| Call | One API request to an outside provider; each retry is its own call |
| Run | One execution of an experiment, identified by `run_id`, with its own folder and $25 cap |
| Policy outcome | What the refund policy says for the record actually shown |
| True outcome | What the refund policy says for the true underlying record |
| Contestable case | A case where Jev's decision differs from the true outcome |

## Bug review, 2026-10-21

A full review found and fixed 11 issues, each with a regression test in [tests/test_regressions.py](tests/test_regressions.py):

1. A cost-cap trip didn't survive resuming the same run; now it does.
2. Reasoning tokens could escape the worst-case estimate for OpenRouter and Gemini; now a 32,768-token allowance is assumed unless a budget is set.
3. A response with no usage data was logged as $0; now it's charged at its worst case.
4. A crash mid-run was recorded as "completed"; now it's "failed".
5. A misconfiguration could log hundreds of failed requests; now the run stops after 10 in a row.
6. New Pilot 0 runs silently reused cached Jev answers; now the cache is off by default.
7. Bedrock's SDK could make one hidden retry, and its 60 s read timeout was short; now retries are off and the timeout is 300 s.
8. Gemini timeouts crashed the run and its SDK retries weren't pinned; now timeouts are handled as uncertain billing and SDK retries are off.
9. A confidence of `true` was accepted as 1.0; now it's rejected.
10. Hard-negative price edits could make a price negative (never happened in v0); now prices stay positive.
11. A refused launch left an empty run folder and a misleading credentials error; now nothing is created and the message names the placeholder model.

Benchmark v0 was unaffected: a test confirms the generator still reproduces it byte for byte.
