# 12 — Roadmap to the First Experiments

How we get from the current harness (provider clients and cost controls) to the first experiment results, and how results are saved so they can be analyzed later without paying for the calls again.

## The first experiment: Pilot 0 — Jev decision characterization

Before we can measure anything about *explanations*, we need to know how Jev itself behaves on our benchmark. Every later experiment depends on this:

- Experiment 1 (faithfulness vs. contestability) needs cases where Jev's decision is actually wrong under ground truth. If Jev never errs on the benchmark, there is nothing to contest.
- Experiments 1–3 compare explanations against Jev's *observed* behavior under counterfactual edits. That only means something if Jev's decisions are stable across repeated samples.
- Jev must stay one decision function $f$ for the whole run. `jev-latest` can move to a new release, so every call records the version that answered and the run halts if it changes (see [11_harness_and_infrastructure.md](11_harness_and_infrastructure.md)).

Pilot 0 is also the cheapest experiment: it calls only Jev (plus the case writer once to produce case text), and it needs no explainer models.

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
- **Volume:** 160 variants × 3 samples = 480 Jev calls, about $0.01 at $0.042 per 1M input tokens. The $2 per call, $25 per run and $5 TypeSafe project caps apply.
- **Case text:** first from deterministic templates (free), then realized in natural language by the case writer (Qwen3 235B on Bedrock, $0.22 / $0.88 per 1M tokens): about $0.10 for all 160 variants.

### Go / no-go criteria for moving on to Experiment 1

- Parse success ≥ 98%.
- At least 20 cases where Jev's decision is wrong under ground truth **and harms the customer** (a denial of a refund the policy owes on the true record), so Experiment 1 has contestable decisions to find. Wrong approvals harm the store rather than the person contesting, so they are reported separately and don't count. If there are too few, change the benchmark before continuing. (Revised 2026-10-28 after the first run, from "at least 20 wrong decisions".)
- At least 90% of variants unanimous across repeated samples, so counterfactual flips aren't just noise (set 2026-10-28 after the first two runs, both about 97%).
- One model version throughout the run.

## Pilot 0 results on refund_v0 (2026-10-28)

Run `pilot0-v0-2026-10-28` (results/ is local only): 480 calls, all answered by `jev-1.14.0`, 100% parsed, **$0.0137** in total ($0.000029 per decision).

| Criterion | Result |
|---|---|
| Parse rate ≥ 98% | Yes, 100% |
| One model version | Yes |
| ≥ 90% of variants unanimous | Yes: 97% across 3 repeats; 95th-percentile range of P(approve) 0.09, largest 0.17 (includes any order effect) |
| ≥ 20 customer-harming wrong decisions | **No: 6** (plus 1 wrong approval) |

What the run showed:

- **Jev misjudges the 30-day window in the customer's favour.** It denied only 58% of samples for requests outside the window; 14 variants 37–52 days past purchase were approved. That masks the "purchase date recorded a month early" defect: Jev approved all 4 such cases anyway, which happens to match the ground truth.
- **Jev reads attachments.** All 10 ignored-evidence cases were decided correctly, so that design produced no wrong decisions.
- **Jev follows corrupted receipt and condition fields.** All 6 such incorrect-input cases were wrong denials: 6 of the 7 contestable cases.
- **No sensitivity to names or message register** (0 of 20 cue edits flipped a decision) and **no option-order effect** (first-listed option chosen 32.7% of the time, 33.3% expected).
- Corrections flip the decision only 60% of the time, again because of the date-window errors.

Response: [refund_v1](#benchmark-v1-as-built), which adds the defects Jev follows. The full write-up of both runs, with figures, is in [research_analyses/E0_jev_characterization/](../research_analyses/E0_jev_characterization/README.md).

## Pilot 0 results on refund_v1 (2026-10-28): go

Run `pilot0-v1-2026-10-28`: 564 calls, all answered by `jev-1.14.0`, 100% parsed, **$0.0160** ($0.000028 per decision). The working tree was uncommitted when it ran (`git.dirty` in the manifest); the benchmark hash and all configs are recorded.

| Criterion | Result |
|---|---|
| Parse rate ≥ 98% | Yes, 100% |
| One model version | Yes |
| ≥ 90% of variants unanimous | Yes: 97%; 95th-percentile range of P(approve) 0.07, largest 0.13 |
| ≥ 20 customer-harming wrong decisions | **Yes: 24** (plus 3 wrong approvals from date-window errors) |

- **All 24 incorrect-input cases are wrong denials** that follow the corrupted record, and **all 24 corrections flip the decision** (mean P(approve) shift 0.99). These are the contestable cases Experiment 1 needs.
- The date-window weakness repeats: Jev denies only 51% of samples outside the window, which accounts for the 3 wrong approvals.
- **Missing receipts:** Jev answered "undetermined" in only 7 of 12 samples where the receipt status is missing (58%); in the other 5 it decided anyway.
- Again no sensitivity to names or message register (0 of 20 cue edits flipped) and no option-order effect (first-listed option chosen 33.5%, 33.3% expected).

## What comes after Pilot 0

| Pilot | Builds on | Needs | What it measures |
|-------|-----------|-------|------------------|
| **Pilot 1 — Experiment 1 pilot** | Pilot 0 decisions | Explainers (Llama 4 Maverick, Gemini 3.1 Flash-Lite; Claude Haiku 5.5 was planned but is blocked on this AWS account); simulator (gpt-oss-120b) | Free-form and structured explanations and the decision-plus-probabilities baseline, scored for faithfulness and contestability. Each explanation is frozen once the explainer has seen Jev's decision, then scored on the benchmark's hidden counterfactual variants: how often its implied decisions agree with Jev's, and how often Jev's and the explainer's independent decisions disagree |
| **Pilot 2 — Stretch S1 pilot** | Pilot 1 explanations | LLM judge (gpt-oss-120b); harness not built yet | Harmful Justification Rate, Defect Disclosure Rate and Over-accusation Rate on the same explanations (mostly scoring, few new calls) |

Pilot 1 and 2 reuse Pilot 0's saved decisions, so Jev isn't called again for them.

### Pilot 1 protocol

Follows the Experiment 1 protocol in [05_experiments.md](05_experiments.md#experiment-1-faithfulness-versus-contestability):

1. **Independent decisions.** Jev's decisions on every original case and variant already come from Pilot 0. The explainer is asked separately to decide the same originals and variants itself, without seeing Jev's answers.
2. **Frozen explanation.** For each original case, the explainer is given Jev's decision and writes an explanation. The explanation is saved and never revised.
3. **Hidden variants.** The case's counterfactual variants from the benchmark are fixed in advance and never shown to the explainer while it explains. Each frozen explanation is scored on them by the decision it implies for each variant.
4. **Measures.** (a) Agreement between explanation-implied decisions and Jev's actual decisions on the hidden variants. (b) Disagreement between Jev's and the explainer's independent decisions. These sit alongside the contestability scoring against the verified defects.

### How an explanation's implied decision is obtained

**Simulator (main measure).** A separate simulator model predicts Jev's decision on each hidden variant from the frozen explanation. It sees the policy, the original case, Jev's decision on the original, the frozen explanation and the variant's presented case text. It never sees Jev's decision on the variant, the true record or the ground truth.

- **Model choice.** The simulator is never the explainer being scored, never from that explainer's model family, and never Jev. It may be the same model as the LLM judge if that rule holds. Chosen 2026-10-27: OpenAI's gpt-oss-120b on Bedrock (flex tier, reasoning effort `low`), from a third family, so one simulator scores every explainer (Llama and Gemini in Pilot 1; Claude too if it is unblocked). GPT-6.1 Terra was considered first, but AWS publishes no price for it and OpenRouter's could not be confirmed. The runner refuses a simulator from an explainer's family.
- **Fixed settings.** The same settings for every call: provider-default temperature, reasoning effort `low` with a 2,048-token reasoning cap, and the same three options Jev gets (approve, deny, undetermined). Bedrock's Converse API has no seed parameter, so none is set.
- **No-explanation control.** The simulator also predicts each variant with the explanation removed and everything else unchanged. Faithfulness is the gain the explanation adds over this control (Normalized Simulatability Gain).

**Structured simulator response.** Every simulator answer has three fields:

1. **Predicted decision:** approve, deny or undetermined, with a probability for each.
2. **Basis:** the rule the simulator took from the explanation, in a fixed structure: an ordered list of rules and a default outcome. Each rule is a set of conditions on the case's policy-relevant fields (manager pre-approval, final sale, receipt, item condition, days since purchase with or without carrier delay, price) and an outcome; the first rule whose conditions all hold decides. This covers a decisive field with its threshold or required value, and any exception applied before it. The format is in [src/sysone/explain/basis.py](../src/sysone/explain/basis.py); the refund policy itself can be written in it exactly, which the tests check.
3. **Covered:** whether the explanation determines this variant at all. "The explanation does not say" is different from predicting that Jev would answer undetermined.

The basis can be run, so each variant yields two predictions from one call:

- **A-direct:** the simulator's predicted decision.
- **A-executed:** the basis run against the variant's structured record. It gives no decision when a rule needs a fact the record lacks, or no rule holds and there is no default.

**Structured explainer condition (Option B).** Explainers are not required to write structured explanations, since free-form explanations are one of the conditions under test. A separate condition asks the explainer for an explanation in the same structure as the basis field. Those explanations are also run directly against each variant's structured record, which gives a deterministic faithfulness score that needs no simulator. Comparing this condition with free-form explanations shows whether structure makes explanations more faithful.

**Scores are kept separate, not blended,** so that a low score can be traced to the explanation or to the simulator:

| Score | Applies to | Role |
|-------|------------|------|
| A-direct agreement with Jev, as gain over the no-explanation control | All explanations | Main faithfulness score |
| A-direct vs. A-executed agreement | All explanations | Simulator check: a gap means the simulator reasoned on its own instead of from the explanation |
| Covered rate | All explanations | Explanation completeness: fraction of hidden variants the explanation decides |
| Executed agreement with Jev (Option B) | Structured explainer condition only | Deterministic anchor, no model involved |

**Calls.** One control simulation per hidden variant (shared by every condition), plus one per explanation per hidden variant. Jev is not called again.

### Pilot 1 as built

[src/sysone/experiments/pilot1.py](../src/sysone/experiments/pilot1.py), configured in [configs/experiments/pilot1.yaml](../configs/experiments/pilot1.yaml), with the explainer and simulator wrappers in [src/sysone/explain/](../src/sysone/explain/) and the analysis in [src/sysone/analysis/pilot1.py](../src/sysone/analysis/pilot1.py).

- **Input:** a completed Pilot 0 run. Jev's decision on each variant is the majority of its 3 Pilot 0 samples, with probabilities averaged. Variants where the samples tie are left out of scoring, and cases whose original ties are not explained (listed in the manifest as `skipped_cases_jev_tie`).
- **Conditions:** `free_form` and `structured` (one explanation per explainer per case, given Jev's decision only), and `decision_probabilities` (no explainer call: the simulator is shown Jev's decision and averaged probabilities).
- **Explainers:** `explainer_open` (Llama 4 Maverick) and `explainer_gemini` (Gemini 3.1 Flash-Lite, thinking level `minimal`). The plan was Claude Haiku 5.5 (`explainer_bedrock`) in Llama's place, but on 2026-10-28 AWS refused the Marketplace subscription Claude needs for this account, even with the use-case form approved. If AWS unblocks it, Claude is added as a third explainer on the same Pilot 0 decisions. Each makes 3 independent decisions per variant, with the same question and rotating option order Jev gets. Bedrock models get the JSON format in the prompt and their reply is parsed; Gemini gets a JSON schema.
- **Steps run in order and resume:** explainer decisions, then explanations, then simulations. A saved explanation is never regenerated, even if the explainer changes.
- **Version lock:** each model is locked to the first version that answers (`model_versions` in the manifest), as for Jev in Pilot 0.
- **Volume with two explainers on benchmark v1:** 1,128 explainer decisions, 296 explanations, and 684 simulator calls (114 hidden variants × the control, 4 explained groups and the baseline). Projected from the preflight's measured costs per call (2026-10-28): about $0.10 for Llama 4 Maverick and $0.09 for the simulator on AWS, and $0.12 for Gemini 3.1 Flash-Lite on GCP, about $0.31 in all. The live run's ledger replaces these estimates.
- **Not built yet:** contestability scoring against the verified defects, which needs the LLM judge.

## Steps to get there

Steps 1–9 are done (✅) except 6b. Step 10 (Pilot 1) is ready to run (preflight passed 2026-10-28); step 6b can run any time.

| Step | Work | Cost |
|------|------|------|
| 1 ✅ | **Benchmark schema and refund policy family:** typed records, the policy as a Python function, ground-truth outcome, defect injection for all six case types, counterfactual edits | Free |
| 2 ✅ | **Template renderer** to turn records into case text deterministically | Free |
| 3 ✅ | **Freeze benchmark v0:** write the cases to `benchmarks/refund_v0/cases.jsonl` with a content hash | Free |
| 4 ✅ | **Jev decision wrapper:** prompt (policy + case → structured decision), JSON schema, parsing and validation, repeat sampling, query budget $B$ | Free to build |
| 5 ✅ | **Run harness and results store** (see below): run manifest, JSONL writers, resume | Free |
| 6 ✅ | **End-to-end dry run** with a fake Jev, to check the whole pipeline and the analysis script | Free |
| 6b | **Natural-language realization** of the cases with the case writer; freeze as `refund_v2` (generator not built yet) | ≈ $0.10 |
| 7 ✅ | **Pilot 0 live run on refund_v0** (`pilot0-v0-2026-10-28`): no-go, 6 customer-harming wrong decisions | $0.0137 |
| 7b ✅ | **Pilot 0 live run on refund_v1** (`pilot0-v1-2026-10-28`): go, 24 customer-harming wrong decisions | $0.0160 |
| 8 ✅ | **Analysis script** for P0.1–P0.8, with wrong decisions split by who they harm | Free |
| 9 ✅ | **Pilot 1 harness:** explainer and simulator wrappers, structured basis and its executor, runner, analysis; dry run with fake models | Free |
| 10 | **Pilot 1 live run** | ≈ $1–2 across AWS and GCP |

## Benchmark v1 as built

`benchmarks/refund_v1/` (seed 1, generator `refund-gen-0.2`, template text, frozen 2026-10-28): the same policy and case types as v0, with **24 incorrect-input cases instead of 10**, using only the corruptions Jev follows (12 "item logged as opened", 12 "receipt not linked"; no purchase-date corruption, which Jev's date errors mask). Every one is presented as a denial of an owed refund. The other five types have 10 cases each: **74 cases, 188 variants**. Case IDs start with `refund_v1-`. The generator still reproduces v0 byte for byte, and a test checks both.

```bash
uv run python -m sysone.benchmark.build --name refund_v1 --seed 1 --per-type 10 \
    --incorrect-input 24 --incorrect-input-kinds condition has_receipt --id-prefix refund_v1
```

Volumes on v1: Pilot 0 is 564 Jev calls (≈ $0.016). Pilot 1 is 564 decisions per explainer, up to 148 explanations per explainer and up to 684 simulator calls (114 hidden variants × 6).

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

uv run python -m sysone.experiments.pilot1 --pilot0-run <pilot0 id> --dry-run         # fake explainers and simulator
uv run python -m sysone.experiments.pilot1 --pilot0-run <pilot0 id> --run-id <id>    # live
uv run python analysis/pilot1.py results/runs/<id>            # regenerate the Pilot 1 report
```

A Pilot 1 dry run needs a completed Pilot 0 run; a Pilot 0 dry run works.

Running again with the same `--run-id` resumes: decisions already saved are skipped, failed requests are retried, and the run's earlier spend still counts toward its $25 cap.

Other run behavior, set in [configs/experiments/pilot0.yaml](../configs/experiments/pilot0.yaml) (Pilot 1's config has the same settings):

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
    explanations.jsonl   # frozen explainer outputs, written given Jev's decision (Pilot 1 onward)
    explainer_decisions.jsonl  # the explainer's own decisions on originals and variants (Pilot 1 onward)
    simulations.jsonl    # one row per simulator call: explanation and variant IDs, with or without
                         #   the explanation, predicted decision and probabilities, basis,
                         #   covered, Jev's decision on the variant (for scoring only) (Pilot 1 onward).
                         #   Executed decisions are derived in the analysis.
    scores.jsonl         # derived per-case metrics; can be regenerated (Pilot 1 onward)
    summary.json         # aggregate metrics for the run; can be regenerated
    report.md            # readable version of summary.json; can be regenerated
analysis/
  pilot0.py              # regenerates summary.json and report.md for a saved run
  pilot1.py              # the same for a Pilot 1 run
```

`results/` is gitignored and, for now, kept locally only. Note the `run_id` of each run that matters, since that is how results are found later.

## Open items before the live runs

- ✅ AWS account verification (done by 2026-10-28).
- ❌ Claude on Bedrock: the use-case form is approved, but AWS refuses the Marketplace subscription for this account; an AWS Support case is needed. Llama 4 Maverick replaces Claude in Pilot 1.
- ✅ The preflight (`scripts/preflight.py`, 2026-10-28): every request kind parsed, Gemini accepted thinking level `minimal`, gpt-oss ran on flex.
- Whether the AWS and GCP credits cover these models (see [11_harness_and_infrastructure.md](11_harness_and_infrastructure.md#assumptions-still-to-verify)).
- Budget alerts in the AWS and GCP consoles.

The project budget and its allocation per experiment are in [11_harness_and_infrastructure.md](11_harness_and_infrastructure.md#project-budget).
