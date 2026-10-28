# 11 — Harness and Infrastructure

How experiments call outside models, and the cost controls around every call. Open decisions are tracked in [TODO.md](../TODO.md).

## Model access

| Role | Alias | Provider | Model | Status |
|------|-------|----------|-------|--------|
| Decision model (Jev) | `jev` | TypeSafe native API | `jev-latest` (was `jev-1.14.0` on 2026-10-26) | Chosen |
| Case writer (natural-language realization of Dataset A) | `case_writer` | Amazon Bedrock, us-east-2 (Ohio) | Claude Opus 4.5 (`us.anthropic.claude-opus-4-5-20251101-v1:0`) | Chosen |
| Explainer | `explainer_bedrock` | Amazon Bedrock | TBD | **TODO** |
| Explainer | `explainer_gemini` | Google Gemini | TBD | **TODO** |
| LLM judge (stretch goal) | `judge` | Bedrock or Gemini | TBD | **TODO** |

Design choices:

- **Jev through TypeSafe's native API.** Jev answers typed questions about a state. A choice question returns the chosen option, a probability for every option (two decimals) and TypeSafe's `confidence`, which is derived from the probabilities. The probabilities are what experiments measure. OpenRouter also lists Jev, but as a chat model, which would only give a confidence Jev types into its reply.
- **Version lock.** `jev-latest` moves when a release ships, and every response names the exact version that answered. Each run locks to the first version that answers, saves it in the manifest as `model_version`, and halts on any answer from another version, including cached answers and answers on resume. The mismatched call is still logged and billed.
- **Case writer is not Jev.** This keeps the model that writes benchmark text separate from the model being judged. The writer should also differ from the explainer in each run.
- **Bedrock via boto3 `invoke_model`.** The request body pins `anthropic_version`. Anthropic's docs note that recent Anthropic SDK releases send an API version that ARN-versioned Bedrock models such as Opus 4.5 reject.
- **The `us.` inference profile** is required for Opus 4.5 in us-east-2. It keeps traffic in US regions and costs 10% more than the `global.` profile.
- **The Bedrock adapter supports Claude models only.** Non-Claude Bedrock explainers would need a Converse-API path added.
- **The Gemini client supports either a Gemini API key or Vertex AI.** Which to use depends on where the Gemini credits are; Vertex prices may differ from the table.

### Gemini model options

Paid-tier Gemini API prices per 1M tokens (input / output), from ai.google.dev/gemini-api/docs/pricing, checked 2026-10-21. Thinking tokens are billed as output.

| Tier | Model ID | Price |
|------|----------|-------|
| Frontier | `gemini-4-argon` | $2 / $10 until Nov 30, 2026; $4 / $20 from Dec 1 |
| Pro | `gemini-3.1-pro-preview` | $2 / $12 up to 200k input tokens; $4 / $18 above (preview) |
| Pro, older | `gemini-2.5-pro` | $1.25 / $10 up to 200k; $2.50 / $15 above |
| Flash | `gemini-3.8-flash` | $0.75 / $3.75 until Dec 31, 2026; $1.50 / $7.50 from Jan 1 |
| Flash | `gemini-3.6-flash` | same as 3.8 Flash |
| Flash, older | `gemini-2.5-flash` | $0.30 / $2.50 |
| Cheap | `gemini-3.5-flash-lite` | $0.30 / $2.50 |
| Cheap | `gemini-3.1-flash-lite` | $0.25 / $1.50 |
| Cheap, older | `gemini-2.5-flash-lite` | $0.10 / $0.40 |

[configs/pricing.yaml](../configs/pricing.yaml) uses the higher of each pair (the later price, or the >200k tier) so worst-case estimates stay conservative.

**Recommendation (not yet decided):** one stronger explainer (`gemini-4-argon` or `gemini-3.1-pro-preview`) and one cheaper one (`gemini-3.8-flash`). That lets the stretch goal test whether rationalization of bad decisions varies with model capability.

### Assumptions still to verify

| Assumption | Where it matters | If wrong |
|------------|------------------|----------|
| Bedrock price for Opus 4.5 on the `us.` profile is $5.50 / $27.50 | Cost estimate and logged cost for `case_writer` | Logged cost is off; the breaker still trips if a call's computed cost exceeds $2 |
| TypeSafe doesn't bill failed (non-200) requests | Logged cost of failed Jev calls | Failed calls are logged as free; Jev is cheap enough that the error is negligible |
| Jev's output tokens stay free | Logged cost for `jev` | Logged cost is too low; the worst-case estimate already allows 1,024 output tokens at the listed output price |
| Gemini counts thinking tokens against `max_output_tokens` | Bounding Gemini output | Output could exceed the estimate; the breaker trips if a call's actual cost exceeds $2 |

## Cost controls

Limits are set in [configs/budget.yaml](../configs/budget.yaml):

- **$2.00 per call.** A call is one API request to TypeSafe, OpenRouter, Bedrock or Google, and every retry counts as its own call.
- **$25.00 per run.** A run is one `run_id`.

Every call goes through `GuardedClient` → `CostGuard` ([src/sysone/budget/guard.py](../src/sysone/budget/guard.py)):

1. **Before the call:**
   - Worst case = (estimated input tokens × input price) + (`max_output_tokens` + `reasoning_max_tokens`) × output price.
   - If a request sets no `reasoning_max_tokens`, OpenRouter and Gemini requests assume up to 32,768 reasoning tokens, because their models may reason without being asked. If that pushes a call over $2, set an explicit reasoning budget.
   - Input tokens are estimated at 3 characters per token, which overestimates normal English.
   - The call is refused if the worst case is over $2, or if the run's spend plus the worst cost of calls still in flight plus this call would exceed $25.
2. **Calls that are never sent:**
   - Models whose ID starts with `TODO`.
   - Models with no entry in [configs/pricing.yaml](../configs/pricing.yaml).
3. **OpenRouter price ceiling.** OpenRouter requests also send the price-table rates as `provider.max_price`, so OpenRouter won't route to a more expensive endpoint.
4. **After the call:**
   - The actual cost is recorded. OpenRouter reports cost directly; TypeSafe, Bedrock and Gemini cost is computed as tokens × price table.
   - If any call actually cost more than $2, the estimate was wrong, so the guard trips and refuses all further calls in the run. The trip is read back from the spending log, so resuming the same `run_id` stays halted; fix the estimate and start a new run.
   - If a response comes back with no usage data, it is charged at its worst case.
5. **Failed calls:**
   - An explicit error response from the provider is logged at $0.
   - A timeout or dropped connection might have been billed, so it is charged at its worst case.
   - There are at most 2 retries, all made by the harness. The Bedrock and Gemini SDKs' own retries are switched off, so no attempt bypasses the cost guard.
6. **Spending log.** Every attempt is appended to `results/runs/<run_id>/ledger.jsonl`, including its tokens, estimated and actual cost, cost source, `model_used` and tags. A resumed run reads its earlier spend from this file.
7. **Cache.** Identical requests are cached in `results/cache/responses.sqlite` and cost $0 on reuse. Experiments can turn it off; Pilot 0 does by default, so each new run queries Jev afresh. A repeated sample of the same request gets its own `sample_index`, so stability sampling still makes real calls.

Backstops outside the harness (recommended): an OpenRouter key with a credit limit, AWS Budgets, and GCP budget alerts. The AWS and GCP budgets only send alerts; they don't stop spending.

## Code layout

```
configs/            models.yaml, pricing.yaml, budget.yaml, experiments/
src/sysone/
  types.py          ChatRequest / SystemOneRequest / ChatResponse / Usage
  budget/           pricing (worst-case estimate), guard (caps), ledger (JSONL)
  clients/          base (GuardedClient, VersionLock), typesafe, openrouter, bedrock, gemini, fake, cache
  registry.py       builds guarded clients from configs, one guard per run
  benchmark/        records, refund policy, generator, renderer, frozen-benchmark store
  decision/         Jev decision wrapper (typed choice question, probabilities, query budget B)
  runs.py           per-run manifest and append-only JSONL outputs
  experiments/      experiment runners (pilot0)
  analysis/         metrics computed from saved runs (pilot0)
benchmarks/         frozen benchmark versions (committed)
scripts/smoke_test.py   one tiny live call per alias
tests/              offline tests (fake provider, mocked HTTP and SDK clients)
```

The benchmark, Jev decision wrapper, run store and Pilot 0 runner are described in [12_first_experiments_roadmap.md](12_first_experiments_roadmap.md). Planned next: the access layer, and the free-form explainer with Experiment 1 and stretch-goal S1 harnesses.
