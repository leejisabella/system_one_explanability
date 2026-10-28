# 12 — Roadmap to the First Experiments

How we get from the current harness (provider clients and cost controls) to the first experiment results, and how results are saved so they can be analyzed later without paying for the calls again.

## The first experiment: Pilot 0 — Jev decision characterization

Before we can measure anything about *explanations*, we need to know how Jev itself behaves on our benchmark. Every later experiment depends on this:

- Experiment 1 (faithfulness vs. contestability) needs cases where Jev's decision is actually wrong under ground truth. If Jev never errs on the benchmark, there is nothing to contest.
- Experiments 1–3 compare explanations against Jev's *observed* behavior under counterfactual edits. That only means something if Jev's decisions are stable across repeated samples.
- Jev must stay one decision function $f$ for the whole run. `jev-latest` can move to a new release, so every call records the version that answered and the run halts if it changes (see [11_harness_and_infrastructure.md](11_harness_and_infrastructure.md)).

Pilot 0 is also the cheapest experiment: it calls only Jev (plus the case writer once to produce case text), and it needs no explainer models, which are still undecided.

### Questions Pilot 0 answers

| # | Question | Measure |
|---|----------|---------|
| P0.1 | Does Jev return usable structured decisions? | Parse/validation success rate |
| P0.2 | Does Jev apply the stated policy to the record it is shown? | Agreement with the policy outcome computed on the *presented* record, by case type and by the rule that decides it, with date-window rules reported apart |
| P0.3 | How often is Jev's decision wrong under ground truth? | Agreement with the *true* outcome, by case type and rule. Each contestable case is classified as following a planted defect or as Jev's own misapplication of the policy (date-window rules vs. other rules) |
| P0.4 | Are decisions stable, and does option order matter? | Agreement and P(approve) range across repeated samples; how often the first-listed option is chosen |
| P0.5 | Does Jev respond to the decisive field? | Flip rate when a policy-relevant field is changed across the decision threshold |
| P0.6 | Is Jev sensitive to irrelevant cues? | Flip rate under name/dialect edits that leave the policy outcome unchanged |
| P0.7 | Did one model version answer every call? | Distribution of `model_used` (the version); a locked run should show one |
| P0.8 | What does it cost? | Cost per decision, to budget Experiments 1–2 |

Note the difference between P0.2 and P0.3. For an incorrect-input case, Jev can apply the policy perfectly to the record it sees and still be wrong under ground truth. That gap is what a contestability explanation should surface.

**Question design.** Jev chooses among approve, deny and undetermined ("the decision cannot be determined"), so ambiguous cases have a correct answer. The option order rotates with the sample index, so across three repeats each option is listed first once. Both follow independent audits: without a "can't tell" option Jev answers unanswerable questions confidently, and it sometimes leans toward the first-listed option. Because repeats differ in order, their P(approve) range is an upper bound on pure noise; set `rotate_option_order: false` for a pure noise-floor run. Date arithmetic is listed by TypeSafe as unreliable for Jev, and 32 of the 60 cases are decided by the purchase window, so the report separates date-window errors.

### Design

- **Policy family:** refund eligibility (customer support). Simple, explicit rules (purchase window, item condition, receipt, exceptions) that are easy to compute programmatically.
- **Cases:** 60, 10 of each case type from [04_datasets_and_benchmark.md](04_datasets_and_benchmark.md): correct, incorrect-input, ignored-evidence, irrelevant-cue, ambiguous, hard negative.
- **Variants per case:** the original plus 1–2 counterfactual variants, 160 variants in total. Which edits each case type gets is listed under *Benchmark v0 as built* below.
- **Samples:** 3 repeats per variant at a fixed temperature, for stability.
- **Volume:** 160 variants × 3 samples = 480 Jev calls. Cost depends on the Jev model's price (still TODO). The $2 per call and $25 per run caps apply regardless.
- **Case text:** first from deterministic templates (free), then realized in natural language by the case writer (Claude Opus 4.5 on Bedrock). The case writer costs an estimated $0.02–0.03 per case, about $1.50 for 60 cases.

### Go / no-go criteria for moving on to Experiment 1

- Parse success ≥ 98%.
- Enough cases where Jev's decision is wrong under ground truth, ideally at least 20 of the 60, so Experiment 1 has defects to find. If there are too few, make the defects harder before continuing.
- Sample-to-sample agreement high enough that counterfactual flips aren't just noise (threshold to set after seeing the data).
- One model version throughout the run.

## What comes after Pilot 0

| Pilot | Builds on | Needs | What it measures |
|-------|-----------|-------|------------------|
| **Pilot 1 — Experiment 1 pilot** | Pilot 0 decisions | Explainer models (TODO) | Free-form explanations and the decision-plus-probabilities baseline, scored for faithfulness and contestability |
| **Pilot 2 — Stretch S1 pilot** | Pilot 1 explanations | LLM judge (TODO) | Harmful Justification Rate, Defect Disclosure Rate and Over-accusation Rate on the same explanations (mostly scoring, few new calls) |

Pilot 1 and 2 reuse Pilot 0's saved decisions, so Jev isn't called again for them.

## Steps to get there

Steps 1–6 are built (✅), along with the analysis script from step 8. Step 7 waits on `TYPESAFE_API_KEY`; step 6b can run any time.

| Step | Work | Cost |
|------|------|------|
| 1 ✅ | **Benchmark schema and refund policy family:** typed records, the policy as a Python function, ground-truth outcome, defect injection for all six case types, counterfactual edits | Free |
| 2 ✅ | **Template renderer** to turn records into case text deterministically | Free |
| 3 ✅ | **Freeze benchmark v0:** write the cases to `benchmarks/refund_v0/cases.jsonl` with a content hash | Free |
| 4 ✅ | **Jev decision wrapper:** prompt (policy + case → structured decision), JSON schema, parsing and validation, repeat sampling, query budget $B$ | Free to build |
| 5 ✅ | **Run harness and results store** (see below): run manifest, JSONL writers, resume | Free |
| 6 ✅ | **End-to-end dry run** with a fake Jev, to check the whole pipeline and the analysis script | Free |
| 6b | **Natural-language realization** of the 60 cases with the case writer; freeze as `refund_v1` | ≈ $1.50 |
| 7 | **Pilot 0 live run** (template text first, then natural-language text) | Depends on Jev price; capped at $25 per run |
| 8 | **Analysis script** for P0.1–P0.8 (✅ built and run on the dry run); go / no-go decision after the live run | Free |

## Benchmark v0 as built

`benchmarks/refund_v0/` (seed 0, generator `refund-gen-0.1`, template text). It is committed to git and its hash is checked on every load.

| Case type | Count | True outcome | What is in the presented case | Variants beyond the original |
|-----------|-------|--------------|-------------------------------|------------------------------|
| correct | 10 | 5 approve, 5 deny | Accurate record | Decisive edit (outcome flips) |
| incorrect_input | 10 | approve | One decisive field corrupted (purchase date a month early, item logged as opened, or receipt not linked), so the policy on the presented record says deny | Correction (true value restored) |
| ignored_evidence | 10 | approve | Accurate record, but the overriding exception (carrier delay or manager pre-approval) appears only in an attachment | Exception shown as a field (same outcome); exception removed (deny) |
| irrelevant_cue | 10 | 5 approve, 5 deny | Accurate record | Name swapped to the other name group; message register swapped (same outcome) |
| ambiguous | 10 | undetermined | Receipt status, item condition or purchase date missing | Missing fact resolved favorably (approve) and unfavorably (deny) |
| hard_negative | 10 | deny | Correct denial, plus a real but irrelevant error (price or order ID) | Error fixed (same outcome); decisive edit (approve) |

Each case also records the evidence that would confirm its facts (receipt, order system, warehouse inspection, carrier record, manager log, each with an access cost), the decision that would harm the customer under ground truth, and a harm severity by price. Case IDs are assigned after shuffling, so they carry no information about case type.

**Provisional:** the name swaps use first names from the title of Bertrand & Mullainathan (2004) as a placeholder until the audit name list is chosen (see [TODO.md](../TODO.md)). Dialect cues are left to Dataset C.

## How to run

```bash
uv run python -m sysone.benchmark.build --name refund_v0 --seed 0 --per-type 10   # already done; refuses to overwrite
uv run python -m sysone.experiments.pilot0 --dry-run          # fake Jev: no network, no cost
uv run python -m sysone.experiments.pilot0 --run-id <id>      # live, needs TYPESAFE_API_KEY
uv run python analysis/pilot0.py results/runs/<id>            # regenerate the report from saved results
```

Running again with the same `--run-id` resumes: decisions already saved are skipped, failed requests are retried, and the run's earlier spend still counts toward its $25 cap.

Other run behavior, set in [configs/experiments/pilot0.yaml](../configs/experiments/pilot0.yaml):

- **No cache reuse by default** (`use_cache: false`). A new run queries Jev afresh rather than reusing answers from earlier runs, which would hide changes in Jev over time.
- **Stops after repeated failures** (`max_consecutive_errors: 10`). For example, if Jev rejects a request parameter, the run halts instead of logging hundreds of errors.
- **The manifest's `status`** is `completed`, `halted: <reason>` (cost cap or repeated failures), `interrupted` (Ctrl-C) or `failed: <error>` (an unexpected crash).

## Saving results for later analysis

Principles:

1. **Raw outputs and derived scores are kept separate.** Scoring and metrics can be recomputed from saved outputs at any time without new API calls.
2. **Write as we go.** Each result is appended to a JSONL file as soon as it arrives, so a crash loses nothing and a resumed run skips work already done.
3. **Everything is traceable.** Every row carries `run_id`, `case_id`, `variant_id`, `sample_index` and the request's cache key, so it can be joined to the spending log and to the raw response in the cache.
4. **Benchmarks are frozen and versioned.** A run records exactly which benchmark version and content hash it used.
5. **Runs record their configuration.** The manifest stores the git commit, config files, model specs and prices, so results can be interpreted later even if configs change.

Layout:

```
benchmarks/
  refund_v0/
    cases.jsonl          # frozen cases: true record, presented record, policy, ground truth,
                         #   case type, defects, evidence, counterfactual variants
    meta.json            # generator version, seed, content hash, case counts
results/                 # gitignored
  cache/responses.sqlite # raw responses keyed by request hash
  runs/<run_id>/
    manifest.json        # experiment, benchmark version + hash, git commit, configs,
                         #   model specs, prices, seeds, start/end time, total cost
    ledger.jsonl         # every API attempt and its cost
    decisions.jsonl      # one row per Jev query: case/variant/sample IDs, parsed decision,
                         #   P(approve) and all probabilities, confidence, parse status,
                         #   model_used (exact version), cost, request key
    explanations.jsonl   # explainer outputs (Pilot 1 onward)
    scores.jsonl         # derived per-case metrics; can be regenerated (Pilot 1 onward)
    summary.json         # aggregate metrics for the run; can be regenerated
    report.md            # readable version of summary.json; can be regenerated
analysis/
  pilot0.py              # regenerates summary.json and report.md for a saved run
```

`results/` is gitignored and, for now, kept locally only. Note the `run_id` of each run that matters, since that is how results are found later.

## Open decisions that block later steps

- TypeSafe API key (blocks step 7).
- Explainer models (blocks Pilot 1).
- LLM judge (blocks the free-text part of Pilot 2).
