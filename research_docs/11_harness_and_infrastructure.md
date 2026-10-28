# 11 — Harness and Infrastructure

How experiments call outside models, and the cost controls around every call. Open decisions are tracked in [TODO.md](../TODO.md).

## Model access

Chosen 2026-10-27 under the project budget (see *Project budget* below). Prices are per 1M tokens (input / output), standard tier unless noted.

| Role | Alias | Provider | Model | Price |
|------|-------|----------|-------|-------|
| Decision model (Jev) | `jev` | TypeSafe native API | `jev-latest` (was `jev-1.14.0` on 2026-10-26) | $0.042 / free |
| Case writer (natural-language realization of Dataset A) | `case_writer` | Amazon Bedrock, us-east-2 (Ohio) | Qwen3 235B (`qwen.qwen3-235b-a22b-2507-v1:0`) | $0.22 / $0.88 |
| Second case writer (independent generation procedure, Risk 2) | `case_writer_2` | Bedrock | Mistral Large 3 (`mistral.mistral-large-3-675b-instruct`) | $0.50 / $1.50 |
| Explainer (first choice; blocked on this AWS account since 2026-10-28) | `explainer_bedrock` | Bedrock | Claude Haiku 5.5 (`global.anthropic.claude-haiku-5-5`) | $0.10 / $0.50 |
| Explainer | `explainer_gemini` | Vertex AI, global endpoint | Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`), thinking level `minimal` | $0.25 / $1.50 |
| Explainer in Pilot 1 (replacing Claude) and open-weight explainer (stretch goal) | `explainer_open` | Bedrock | Llama 4 Maverick (`us.meta.llama4-maverick-17b-instruct-v1:0`) | $0.24 / $0.97 |
| Simulator (Pilot 1 faithfulness scoring) | `simulator` | Bedrock | gpt-oss-120b (`openai.gpt-oss-120b-1:0`), flex tier, reasoning effort `low` | $0.15 / $0.60; flex $0.075 / $0.30 |
| LLM judge (Pilot 2 onward) | `judge` | Bedrock | same as the simulator | same |
| Second decision model (Experiment 8) | `decision_alt` | Bedrock | Amazon Nova Lite (`amazon.nova-lite-v1:0`) | $0.06 / $0.24 |

**Where the prices come from.** Bedrock: AWS's official Price List files for us-east-2 (offers `AmazonBedrock` and `AmazonBedrockFoundationModels`, published 2026-10-12, read 2026-10-27), standard on-demand rows. Cross-Region profiles (`us.`, `global.`) are billed at these prices; Claude's `global.` profile is 10% cheaper than its `us.` profile. The Bedrock pricing web page did not load its price tables, so it was not used. Gemini: ai.google.dev/gemini-api/docs/pricing, checked 2026-10-21 and again 2026-10-27. TypeSafe: docs.typesafe.ai/models, checked 2026-10-26.

Design choices:

- **Model families stay apart by role.** The case writers (Qwen, Mistral) differ from every explainer (Anthropic, Google, Meta) and from Jev. The simulator and judge (OpenAI) differ from every explainer they score. The Pilot 1 runner refuses a simulator from an explainer's family, and a test checks the whole assignment. Gemma, although on Bedrock, is a Google model and would clash with the Gemini explainer.
- **Jev through TypeSafe's native API.** Jev answers typed questions about a state. A choice question returns the chosen option, a probability for every option (two decimals) and TypeSafe's `confidence`, which is derived from the probabilities. The probabilities are what experiments measure. OpenRouter also lists Jev, but as a chat model, which would only give a confidence Jev types into its reply.
- **Version lock.** `jev-latest` moves when a release ships, and every response names the exact version that answered. Each run locks to the first version that answers, saves it in the manifest as `model_version`, and halts on any answer from another version, including cached answers and answers on resume. The mismatched call is still logged and billed. Pilot 1 locks every model the same way (`model_versions`).
- **Bedrock: Claude through InvokeModel, everything else through Converse** ([src/sysone/clients/bedrock.py](../src/sysone/clients/bedrock.py)). The InvokeModel body pins `anthropic_version`; Anthropic's docs note that recent Anthropic SDK releases send an API version that ARN-versioned Bedrock models reject. On Converse, gpt-oss's reasoning effort goes in `additionalModelRequestFields`, and its reasoning budget is added to `maxTokens`, which maps to OpenAI's `max_completion_tokens` and so includes reasoning. Bedrock refuses, before sending, any setting it would otherwise drop (a reasoning effort for a non-OpenAI model, a seed, a JSON schema).
- **Flex tier for the simulator and judge.** Requested as `serviceTier: {"type": "flex"}`. Bedrock's response names the tier that served the call; the flex price is charged only when it says flex, and the standard price otherwise. Worst-case estimates always use the standard price.
- **Per-model request settings live in [configs/models.yaml](../configs/models.yaml)** (`service_tier`, `reasoning_effort`) and are applied by `GuardedClient` before the cache key is computed, so answers from different settings never mix.
- **Gemini thinking.** Google documents no way to turn thinking fully off for 3.1 Flash-Lite (only 2.5 Flash-Lite, which new accounts may not be able to use), so the explainer runs at `thinking_level: minimal`. Thinking tokens are billed as output and recorded per call (`reasoning_tokens`); the first live call shows how many it actually uses.
- **Gemini through Vertex AI, not the API key** (since 2026-10-28). The AI Studio key stayed on the free tier (500 requests per day), because AI Studio rejects the course's education billing account as "inactive or unsupported". Vertex bills project `cs-329x-510703` and its credits directly. Same model, same price on the global endpoint ($0.25 / $1.50); regional endpoints cost 10% more. Vertex reports more input tokens for schema-constrained requests (about 1,215 vs 706 for a structured explanation), probably because it counts the schema.
- **Structured output.** Gemini gets a JSON schema. Bedrock models get the JSON format in the prompt and their reply is parsed.

### Assumptions still to verify

| Assumption | Where it matters | If wrong |
|------------|------------------|----------|
| The project's AWS and GCP credits cover these models (Claude on Bedrock is billed through AWS Marketplace, which some credits exclude) | Whether spend comes out of credits | Spend is billed to the card; the amounts are the same |
| gpt-oss's reasoning tokens are capped by `maxTokens` and billed in `outputTokens` | Worst-case estimate for the simulator and judge | A call could cost more than its estimate; the breaker trips if any call costs over $2, and at $0.60 per 1M output tokens that would take over 3M tokens |
| Gemini 3.1 Flash-Lite accepts `thinking_level: minimal` | Explainer requests | The first call fails (not billed) and the level needs changing |
| TypeSafe doesn't bill failed (non-200) requests | Logged cost of failed Jev calls | Failed calls are logged as free; Jev is cheap enough that the error is negligible |
| Jev's output tokens stay free | Logged cost for `jev` | Logged cost is too low; the worst-case estimate already allows 1,024 output tokens at the listed output price |
| Gemini counts thinking tokens against `max_output_tokens` | Bounding Gemini output | Output could exceed the estimate; the breaker trips if a call's actual cost exceeds $2 |

## Cost controls

Limits are set in [configs/budget.yaml](../configs/budget.yaml):

- **$2.00 per call.** A call is one API request to TypeSafe, Bedrock, Google or OpenRouter, and every retry counts as its own call.
- **$25.00 per run.** A run is one `run_id`.
- **Project caps per provider:** Bedrock (AWS) $50, Gemini (GCP) $50, TypeSafe $5, OpenRouter $0. These cover the whole project: every run's spending log under `results/runs/` is summed, including smoke tests.

Every call goes through `GuardedClient` → `CostGuard` ([src/sysone/budget/guard.py](../src/sysone/budget/guard.py)):

1. **Before the call:**
   - Worst case = (estimated input tokens × input price) + (`max_output_tokens` + `reasoning_max_tokens`) × output price.
   - If a request sets no `reasoning_max_tokens`, OpenRouter and Gemini requests assume up to 32,768 reasoning tokens, because their models may reason without being asked. If that pushes a call over $2, set an explicit reasoning budget. On Bedrock, reasoning is inside the output cap, so no extra allowance is needed.
   - The worst case always uses the standard price, even when a cheaper tier is requested.
   - Input tokens are estimated at 3 characters per token, which overestimates normal English.
   - The call is refused if the worst case is over $2, if the run's spend plus the worst cost of calls still in flight plus this call would exceed $25, or if the provider's spend across all runs plus its calls in flight plus this call would exceed its project cap. A provider with no project cap is refused.
2. **Calls that are never sent:**
   - Models whose ID starts with `TODO`.
   - Models with no entry in [configs/pricing.yaml](../configs/pricing.yaml).
3. **OpenRouter price ceiling.** OpenRouter requests also send the price-table rates as `provider.max_price`, so OpenRouter won't route to a more expensive endpoint.
4. **After the call:**
   - The actual cost is recorded. OpenRouter reports cost directly; TypeSafe, Bedrock and Gemini cost is computed as tokens × price table, at the flex price only when Bedrock confirms the flex tier served the call.
   - If any call actually cost more than $2, the estimate was wrong, so the guard trips and refuses all further calls in the run. The trip is read back from the spending log, so resuming the same `run_id` stays halted; fix the estimate and start a new run.
   - If a response comes back with no usage data, it is charged at its worst case.
5. **Failed calls:**
   - An explicit error response from the provider is logged at $0.
   - A timeout or dropped connection might have been billed, so it is charged at its worst case.
   - There are at most 2 retries, all made by the harness. The Bedrock and Gemini SDKs' own retries are switched off, so no attempt bypasses the cost guard.
6. **Spending log.** Every attempt is appended to `results/runs/<run_id>/ledger.jsonl`, including its tokens, estimated and actual cost, cost source, the service tier requested and served, `model_used` and tags. A resumed run reads its earlier spend from this file, and the project caps are summed from all of these files. Each run prints the project spend so far and saves it in its manifest (`project_spend_at_start`).
7. **Cache.** Identical requests are cached in `results/cache/responses.sqlite` and cost $0 on reuse. Experiments can turn it off; Pilot 0 does by default, so each new run queries Jev afresh. A repeated sample of the same request gets its own `sample_index`, so stability sampling still makes real calls.

The project caps are only as complete as `results/`: deleting a run folder removes its spend from the totals. The AWS and GCP consoles remain the authoritative record. Backstops outside the harness (recommended): AWS Budgets and GCP budget alerts, for example at $40 each. They only send alerts; they don't stop spending.

## Project budget

Every experiment, including the stretch goal, has to fit in **$50 on AWS, $50 on GCP and $5 on TypeSafe**. The model choices above were made for that. Planning allocation per experiment, also in [configs/budget.yaml](../configs/budget.yaml) (`allocations_usd`, not enforced):

| | AWS (Bedrock) | GCP (Gemini) | TypeSafe |
|---|---|---|---|
| Pilots + Experiment 1 (Jev vs. explainer) | $6 | $4 | $0.20 |
| Experiment 2 (similar targets) | $2.50 | $2.50 | $0.20 |
| Experiment 3 (information ablation) | $2.50 | $2.50 | $0.50 |
| Experiment 4 (CAPE, budget sweep) | $9 | $6 | $1.80 |
| Experiments 5–7 (identifiability, robustness, socially sensitive) | $7 | $6 | $0.80 |
| Experiment 8 (generalization) | $5 | $3 | $0.20 |
| Stretch S1–S3, S5 | $12 | $9 | $0.30 |
| Reserve | $6 | $17 | $1.00 |

Revised 2026-10-28 when the experiments were renumbered: the new Experiments 2 and 3 are funded by trimming CAPE (formerly Experiment 2: $12 / $8 / $2.50) and from the reserve (formerly $8 / $20). Pilot 1 cost $0.41 in all, so Experiments 2 and 3 at a few Pilot-1-sized runs each fit their allocations.

The whole-project estimate (2026-10-27, from dry-run prompt sizes and assumed reply lengths) is about $35–40 on AWS, $20–25 on GCP and $2–3 on TypeSafe, and could be off by 2–3×. AWS is the tight one. Rules that keep it within budget:

- **Re-estimate before each experiment** from measured costs per call (Pilot 0 and Pilot 1 give the first ones), with a go / no-go against what is left.
- **Experiment 4 sweeps with one explainer** (Claude Haiku 5.5, or Llama 4 Maverick while Claude is blocked; Llama costs about twice as much per token, so re-estimate first); Gemini runs at a single query budget.
- **Reuse identical Jev queries within a run** (many baselines and budgets ask the same counterfactual). Answers from earlier runs are still not reused.
- **Fewer repeats after the pilots** once their noise is measured, e.g. 1 explainer decision per variant instead of 3.
- **Off the cloud budgets:** S4 fine-tuning runs locally or in a free notebook with a small open-weight model, or only its in-context variant (S4a) runs; on Bedrock, a fine-tuned model typically needs hourly-billed reserved capacity. Experiment 8's classifiers for BANKING77 run locally.

## Code layout

```
configs/            models.yaml, pricing.yaml, budget.yaml, experiments/
src/sysone/
  types.py          ChatRequest / SystemOneRequest / ChatResponse / Usage
  budget/           pricing (worst-case estimate), guard (caps), ledger (JSONL)
  clients/          base (GuardedClient, VersionLock), typesafe, bedrock, gemini, openrouter, fake, cache
  registry.py       builds guarded clients from configs, one guard per run; project spend totals
  env.py            loads secrets from .env (gitignored)
  benchmark/        records, refund policy, generator, renderer, frozen-benchmark store
    domains/        moderation, fraud, tool selection, routing, eligibility on one shared generator
    adapter.py      one interface to any benchmark's domain, for the runners and analyses
  decision/         Jev decision wrapper (typed choice question per domain, probabilities, query budget B)
  explain/          explainer and simulator wrappers (prompts set per domain), structured basis and its executor
  runs.py           per-run manifest and append-only JSONL outputs
  experiments/      experiment runners (pilot0, pilot1)
  analysis/         metrics computed from saved runs (pilot0, pilot1)
benchmarks/         frozen benchmark versions (committed)
scripts/smoke_test.py   one tiny live call per alias
tests/              offline tests (fake provider, mocked HTTP and SDK clients)
```

The benchmark, Jev decision wrapper, run store, and the Pilot 0 and Pilot 1 runners are described in [12_first_experiments_roadmap.md](12_first_experiments_roadmap.md). Planned next: the access layer, contestability scoring with the judge, and the stretch-goal S1 harness.
