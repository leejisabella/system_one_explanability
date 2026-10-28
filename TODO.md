# TODO

Open decisions and setup tasks for the harness. Placeholder model IDs start with `TODO` in [configs/models.yaml](configs/models.yaml); the harness refuses to call them until they are replaced.

## Model choices (placeholders in configs/models.yaml)

- [x] **Jev model**: `jev-latest` on TypeSafe's native API (decided 2026-10-26). It pointed to `jev-1.14.0` that day. Runs lock to the version that answers first and halt if it changes.
  - [x] Price added to [configs/pricing.yaml](configs/pricing.yaml): $0.042 per 1M input tokens, output free.
  - [ ] If TypeSafe ships a new release mid-project, decide whether to stay on `jev-1.14.0` by its fixed ID or start fresh runs on the new version.
- [ ] **Bedrock explainer model(s)** (`explainer_bedrock`). If non-Claude, add a Converse-API path to [src/sysone/clients/bedrock.py](src/sysone/clients/bedrock.py).
- [ ] **Gemini explainer model(s)** (`explainer_gemini`). Options with prices are listed in [configs/pricing.yaml](configs/pricing.yaml).
- [ ] **LLM judge** for stretch-goal free-text scoring (`judge`), from a different model family than the explainer being scored.

## Benchmark and experiments

- [ ] **Audit name list** for name-swap cues. Benchmark v0 uses first names from the title of Bertrand & Mullainathan (2004) as a placeholder ([src/sysone/benchmark/generator.py](src/sysone/benchmark/generator.py)). Choosing a different list means freezing a new benchmark version.
- [ ] **Step 6b:** natural-language realization of the cases with the case writer, frozen as `refund_v1` (≈ $1.50).
- [ ] **Step 7:** live Pilot 0 run, once `TYPESAFE_API_KEY` is set.

## Setup and verification

- [ ] Verify the Bedrock price for Claude Opus 4.5 (`us.` profile, us-east-2) on https://aws.amazon.com/bedrock/pricing/.
- [ ] Enable Claude Opus 4.5 model access in the Bedrock console for us-east-2.
- [ ] Decide Gemini API key vs. Vertex AI (depends on where the Gemini credits live). Vertex pricing may differ from the table.
- [ ] Create a dedicated TypeSafe key, and set AWS Budgets / GCP budget alerts as backstops.
- [ ] Run `uv run python scripts/smoke_test.py jev` once `TYPESAFE_API_KEY` is set, and `... case_writer` once AWS credentials are set.
