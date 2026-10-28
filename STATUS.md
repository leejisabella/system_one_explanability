# Project Status — start here

Last updated: 2026-10-28 (Pilot 0 passes on refund_v1; preflight passed; Pilot 1 ready, with Llama replacing the blocked Claude explainer). Read this first when picking the project back up.

## In one paragraph

We study whether explanations of opaque AI decisions give people verifiable grounds to contest those decisions (main question), and whether LLMs asked to explain Jev's decisions can be pushed into justifying bad or user-harmful decisions (stretch goal). The research plan is written up in [research_docs/](research_docs/README.md). The code so far is the infrastructure for the **first experiment, Pilot 0**: a frozen benchmark of 60 synthetic refund cases, a cost-capped harness for calling outside models, a wrapper that asks Jev for typed decisions through TypeSafe's native API, and a runner and analysis script that save and summarize results. The **Pilot 1** (Experiment 1) harness is also built: explainers decide the cases themselves and explain Jev's decisions, and a simulator model (gpt-oss-120b) scores each frozen explanation on hidden counterfactual variants. Every model is chosen and priced from official price lists, and the whole project must fit in **$50 on AWS, $50 on GCP and $5 on TypeSafe**, which the harness enforces across all runs. Jev is called as `jev-latest`; each run records the full probabilities and the exact version on every call, and halts if the version changes. **Pilot 0 ran live on 2026-10-28** (`pilot0-v0-2026-10-28`, $0.0137): Jev is stable and parses perfectly, but only 6 of 60 cases were customer-harming wrong decisions, short of the 20 needed, so a new benchmark, **refund_v1**, adds the defects Jev follows. **On refund_v1 Pilot 0 passes** (`pilot0-v1-2026-10-28`, $0.0160): 24 customer-harming wrong decisions, 100% parsed, 97% of variants unanimous (results in [research_docs/12](research_docs/12_first_experiments_roadmap.md#pilot-0-results-on-refund_v1-2026-10-28-go)). AWS has verified the account and every Bedrock model answers except **Claude**: AWS refuses the Marketplace subscription it needs for this account, so **Llama 4 Maverick replaces Claude Haiku 5.5** as Pilot 1's second explainer (Claude can join later as a third). The **preflight passed** on 2026-10-28 and projects Pilot 1 at about $0.31. The first live attempt (`pilot1-v1-2026-10-28`) halted: Llama reasoned past its 64-token decision cap (fixed with an answer-only system message) and the Gemini API key was on Google's free tier (Gemini now runs through **Vertex AI**, billed to the education credits). Both fixes passed the preflight; the rerun is `pilot1-v1b-2026-10-28`.

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

Not built yet: the natural-language version of the cases (step 6b, to be `refund_v2`), the access layer that controls what an explainer may see, Pilot 1's contestability scoring (needs the LLM judge), CAPE, and the stretch-goal harness.

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
6. **Run Pilot 1 live** on the refund_v1 Pilot 0 run (about $0.31 across AWS and GCP, from the preflight). The first attempt, `pilot1-v1-2026-10-28`, halted and is kept as a record only; rerun as `pilot1-v1b-2026-10-28`: `uv run python -m sysone.experiments.pilot1 --pilot0-run pilot0-v1-2026-10-28 --run-id pilot1-v1b-2026-10-28`.
7. **Re-estimate the project budget** from the measured costs per call, before later experiments ([research_docs/11](research_docs/11_harness_and_infrastructure.md#project-budget)).
8. **Step 6b** (natural-language cases with the case writer, about $0.10, frozen as `refund_v2`) can happen any time; its generator is not built yet.

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
