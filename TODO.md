# TODO

Open decisions and setup tasks for the harness. Placeholder model IDs start with `TODO` in [configs/models.yaml](configs/models.yaml); the harness refuses to call them until they are replaced.

## Model choices (placeholders in configs/models.yaml)

- [ ] **Jev model ID** (`jev` alias). The only public OpenRouter listing as of 2026-10-21 is `typesafe/jev-router`, which picks a model and reasoning effort per request and has dynamic pricing. Prefer a fixed model ID if one is available to our key; otherwise use the router and analyze results by the logged `model_used`.
  - [ ] Add its price to [configs/pricing.yaml](configs/pricing.yaml) under `openrouter:`. That price is also sent as OpenRouter's `provider.max_price` ceiling.
  - [ ] Check whether `max_price` is honored for `jev-router`.
- [ ] **Bedrock explainer model(s)** (`explainer_bedrock`). If non-Claude, add a Converse-API path to [src/sysone/clients/bedrock.py](src/sysone/clients/bedrock.py).
- [ ] **Gemini explainer model(s)** (`explainer_gemini`). Options with prices are listed in [configs/pricing.yaml](configs/pricing.yaml).
- [ ] **LLM judge** for stretch-goal free-text scoring (`judge`), from a different model family than the explainer being scored.

## Benchmark and experiments

- [ ] **Audit name list** for name-swap cues. Benchmark v0 uses first names from the title of Bertrand & Mullainathan (2004) as a placeholder ([src/sysone/benchmark/generator.py](src/sysone/benchmark/generator.py)). Choosing a different list means freezing a new benchmark version.
- [ ] **Jev reasoning budget** (`reasoning_max_tokens` in [configs/experiments/pilot0.yaml](configs/experiments/pilot0.yaml)), once we know whether the Jev model reasons.
- [ ] **Step 6b:** natural-language realization of the cases with the case writer, frozen as `refund_v1` (≈ $1.50).
- [ ] **Step 7:** live Pilot 0 run, once the Jev model ID is set.

## Setup and verification

- [ ] Verify the Bedrock price for Claude Opus 4.5 (`us.` profile, us-east-2) on https://aws.amazon.com/bedrock/pricing/.
- [ ] Enable Claude Opus 4.5 model access in the Bedrock console for us-east-2.
- [ ] Decide Gemini API key vs. Vertex AI (depends on where the Gemini credits live). Vertex pricing may differ from the table.
- [ ] Create a dedicated OpenRouter key with a credit limit, and set AWS Budgets / GCP budget alerts as backstops.
- [ ] Run `uv run python scripts/smoke_test.py case_writer` once credentials are set.
