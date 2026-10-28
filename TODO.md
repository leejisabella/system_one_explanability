# TODO

Open decisions and setup tasks. Every model in [configs/models.yaml](configs/models.yaml) is chosen and priced; model IDs starting with `TODO` would be refused, but none are left.

## Before the first live runs

- [x] **Keys in `.env`** (gitignored): `TYPESAFE_API_KEY`, `GEMINI_API_KEY`, AWS access key pair (2026-10-27).
- [ ] **AWS account verification.** On 2026-10-27 Bedrock refused calls with "Your account is currently being verified" (not billed). AWS says it normally takes under 2 hours; otherwise write to aws-verification@amazon.com.
- [ ] **IAM: add AWS Marketplace permissions** (`aws-marketplace:Subscribe`, `Unsubscribe`, `ViewSubscriptions`) and `bedrock:GetFoundationModelAvailability` to the `system-one` user; policy in [README.md](README.md#credentials). Needed once, for the first call to Claude (the only Marketplace-sold model here).
- [ ] **Anthropic use-case form:** submit it once in the Bedrock console (choose a Claude model in the model catalog). It can sit in a pending state; AWS emails when it's done.
- [ ] **Run the preflight** (`uv run python scripts/preflight.py`, well under $0.01), then smoke-test `case_writer`, `case_writer_2`, `explainer_open`, `decision_alt` with `uv run python scripts/smoke_test.py <alias>`. Check:
  - [ ] Claude Haiku 5.5 access (Anthropic may ask for a one-time use-case form in the Bedrock console).
  - [ ] Gemini 3.1 Flash-Lite accepts `thinking_level: minimal`, and how many thinking tokens it uses.
  - [ ] gpt-oss-120b's response reports `serviceTier: flex` (otherwise it is charged at the standard price).
- [ ] **Credits:** in AWS *Billing → Credits*, check whether credits cover Claude (billed through AWS Marketplace) as well as the Bedrock-billed models; in GCP *Billing → Credits*, check that they cover the Gemini API project.
- [ ] **Budget alerts:** AWS Budgets and a GCP budget alert, e.g. at $40 each.
- [ ] **Commit** before live runs, so each run's manifest names a clean git commit.

## Model choices (decided 2026-10-27, research_docs/11)

- [x] **Jev:** `jev-latest` on TypeSafe's native API ($0.042 per 1M input tokens, output free). It pointed to `jev-1.14.0` on 2026-10-26.
  - [ ] If TypeSafe ships a new release mid-project, decide whether to stay on `jev-1.14.0` by its fixed ID or start fresh runs on the new version.
- [x] **Case writers:** Qwen3 235B (`case_writer`) and Mistral Large 3 (`case_writer_2`) on Bedrock.
- [x] **Explainers:** Claude Haiku 5.5 (Bedrock), Gemini 3.1 Flash-Lite (Gemini API), Llama 4 Maverick (Bedrock, stretch goal).
- [x] **Simulator and judge:** gpt-oss-120b on Bedrock, flex tier, reasoning effort `low`.
- [x] **Second decision model for Experiment 6:** Amazon Nova Lite on Bedrock.
- [x] **Prices** from AWS's official Price List files (us-east-2, published 2026-10-12) and Google's pricing page (checked 2026-10-27). Re-check before large runs.

## Benchmark and experiments

- [ ] **Audit name list** for name-swap cues. Benchmark v0 uses first names from the title of Bertrand & Mullainathan (2004) as a placeholder ([src/sysone/benchmark/generator.py](src/sysone/benchmark/generator.py)). Choosing a different list means freezing a new benchmark version.
- [ ] **Step 7:** live Pilot 0 run.
- [ ] **Step 6b:** build the natural-language realization with the case writer, frozen as `refund_v1` (≈ $0.10).
- [x] **Pilot 1 harness** built 2026-10-27, dry run passed. The live run waits on a live Pilot 0 run and AWS verification.
- [x] **Pilot 1:** how a frozen explanation's implied decision on each hidden variant is obtained (decided 2026-10-27): a simulator model with a structured response, scored against a no-explanation control, plus a structured-explanation condition run directly against the records. See [research_docs/12](research_docs/12_first_experiments_roadmap.md#how-an-explanations-implied-decision-is-obtained).
- [ ] **Pilot 1 contestability scoring** against the verified defects, with the judge.
- [ ] **Re-estimate costs** from Pilot 0 and Pilot 1's measured costs before each later experiment, against the allocations in [configs/budget.yaml](configs/budget.yaml).
- [ ] **Small-run option** (`--max-cases`) for trying a few cases before a full run (offered, not built).
- [ ] **Within-run Jev query reuse** for Experiment 2 (planned in research_docs/11 *Project budget*).

## Setup and verification

- [x] Project caps per provider: AWS $50, GCP $50, TypeSafe $5, enforced across all runs ([configs/budget.yaml](configs/budget.yaml)).
- [ ] **Bedrock structured output:** Bedrock models get the JSON format in the prompt and their reply is parsed. Wire up Converse's structured output if parse rates are low.
- [ ] Create a dedicated TypeSafe key if the current one is shared.
