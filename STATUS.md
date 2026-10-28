# Project Status — start here

Last updated: 2026-10-26 (Jev moved to TypeSafe's native API). Read this first when picking the project back up.

## In one paragraph

We study whether explanations of opaque AI decisions give people verifiable grounds to contest those decisions (main question), and whether LLMs asked to explain Jev's decisions can be pushed into justifying bad or user-harmful decisions (stretch goal). The research plan is written up in [research_docs/](research_docs/README.md). The code so far is the infrastructure for the **first experiment, Pilot 0**: a frozen benchmark of 60 synthetic refund cases, a cost-capped harness for calling outside models, a wrapper that asks Jev for typed decisions through TypeSafe's native API, and a runner and analysis script that save and summarize results. Jev is called as `jev-latest`; each run records the full probabilities and the exact version on every call, and halts if the version changes. Everything has been tested offline with a fake Jev. **No live model has been called yet.**

## What exists

| Piece | Where | State |
|-------|-------|-------|
| Research plan (questions, framework, datasets, experiments, metrics, stretch goal, risks, references) | [research_docs/00–10](research_docs/README.md) | Written |
| Harness design and cost controls | [research_docs/11_harness_and_infrastructure.md](research_docs/11_harness_and_infrastructure.md) | Written |
| Roadmap to the first experiments, Pilot 0 design, results layout | [research_docs/12_first_experiments_roadmap.md](research_docs/12_first_experiments_roadmap.md) | Written |
| Provider clients (TypeSafe, OpenRouter, Bedrock, Gemini), cost guard, version lock, spending log, cache | [src/sysone/clients/](src/sysone/clients/), [src/sysone/budget/](src/sysone/budget/) | Built, tested offline |
| Benchmark generator, refund policy, renderer | [src/sysone/benchmark/](src/sysone/benchmark/) | Built, tested |
| Benchmark v0 (60 cases, 160 variants, template text) | [benchmarks/refund_v0/](benchmarks/refund_v0/) | Frozen |
| Jev decision wrapper (typed choice question, full probabilities) | [src/sysone/decision/jev.py](src/sysone/decision/jev.py) | Built, tested with a fake Jev |
| Pilot 0 runner and analysis | [src/sysone/experiments/pilot0.py](src/sysone/experiments/pilot0.py), [src/sysone/analysis/pilot0.py](src/sysone/analysis/pilot0.py) | Built, dry run passed |
| Open decisions and setup tasks | [TODO.md](TODO.md) | Up to date |

Not built yet: the natural-language version of the cases (step 6b), the access layer that controls what an explainer may see, explainers and their baselines, CAPE, and the stretch-goal harness.

## How to pick it up

```bash
uv sync                                                   # install
uv run pytest                                             # 108 offline tests; no network, no cost
uv run python -m sysone.experiments.pilot0 --dry-run      # whole Pilot 0 pipeline with a fake Jev, no cost
```

The dry run writes a folder under `results/runs/`. It contains random decisions (the fake Jev ignores the case), so delete it afterwards.

## What's blocking, and what to do next

In order:

1. **Set the TypeSafe key**: `TYPESAFE_API_KEY`. Run `uv run python scripts/smoke_test.py jev` to check it and see which version `jev-latest` resolves to (costs a tiny fraction of a cent).
2. **Run Pilot 0 live**: `uv run python -m sysone.experiments.pilot0 --run-id pilot0-v0-<date>` (480 Jev calls, capped at $25, expected to cost well under $0.01). The report is printed and saved in the run folder. Its probability-noise table gives a first noise floor.
3. **Set AWS credentials** for us-east-2 with Claude Opus 4.5 access enabled, only when step 5 needs the case writer. `uv run python scripts/smoke_test.py case_writer` checks them.
4. **Decide go / no-go** for Experiment 1 from the report (criteria in [research_docs/12](research_docs/12_first_experiments_roadmap.md)).
5. **Step 6b**: have the case writer rewrite the cases as natural text, freeze as `refund_v1` (≈ $1.50). This can happen any time, independently of 1–4.
6. **Choose the explainer models and the judge**, then build Pilot 1 (Experiment 1) and Pilot 2 (stretch goal S1).

## Things that are easy to forget

- **Cost limits:** $2 per API request and $25 per run, set in [configs/budget.yaml](configs/budget.yaml). Calls that could exceed either are refused before sending; if any call actually costs more than $2, the run stops.
- **Jev version lock:** `jev-latest` moves when TypeSafe ships a release. A run locks to the first version that answers (saved as `model_version` in its manifest) and halts on any answer from another version, including on resume. To keep using a version after the alias moves, set the fixed ID (e.g. `jev-1.14.0`) in [configs/models.yaml](configs/models.yaml), or set `expected_model_version` in the experiment config.
- **Placeholders:** any model ID starting with `TODO` in [configs/models.yaml](configs/models.yaml) is refused, as is any model missing from [configs/pricing.yaml](configs/pricing.yaml).
- **`results/` stays local.** It is gitignored and not backed up anywhere. Write down the `run_id` of each run that matters.
- **Benchmarks are frozen.** Never edit `benchmarks/refund_v0/` by hand: loading checks a hash and will refuse an edited file. To change cases, freeze a new version under a new name.
- **Provisional name list:** the name-swap cases use first names from Bertrand & Mullainathan (2004) as a placeholder until the audit name list is chosen.
- **Pilot 0 does not reuse cached answers** from earlier runs (`use_cache: false`), so every new run makes fresh, billed calls. Resuming the same run never re-pays for saved decisions.
- **Unverified assumptions behind cost estimates** are listed in [research_docs/11](research_docs/11_harness_and_infrastructure.md#assumptions-still-to-verify).
- **Prices change.** Gemini prices rise on Dec 1, 2026 and Jan 1, 2027; the price table already uses the higher prices.

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
