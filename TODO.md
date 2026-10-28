# TODO

Open decisions and setup tasks. Every model in [configs/models.yaml](configs/models.yaml) is chosen and priced; model IDs starting with `TODO` would be refused, but none are left.

## Before the first live runs

- [x] **Keys in `.env`** (gitignored): `TYPESAFE_API_KEY`, `GEMINI_API_KEY`, AWS access key pair (2026-10-27).
- [x] **AWS account verification:** done by 2026-10-28; every Bedrock model except Claude answers.
- [x] **IAM: AWS Marketplace permissions** (`aws-marketplace:Subscribe`, `Unsubscribe`, `ViewSubscriptions`) and `bedrock:GetFoundationModelAvailability` on the `system-one` user (2026-10-28); policy in [README.md](README.md#credentials).
- [x] **Anthropic use-case form** submitted and approved (`authorizationStatus: AUTHORIZED`, 2026-10-28).
- [ ] **Claude on Bedrock is blocked for this account.** Calls, including from an admin in the console playground, fail with "anthropic.claude-haiku-5-5 is not available for this account"; `GetFoundationModelAvailability` shows `agreementAvailability: NOT_AVAILABLE` (the Marketplace subscription is never created). Open an AWS Support case (Account and billing) asking to enable Marketplace subscriptions for Bedrock models, and ask whether credits cover Marketplace charges. Until then Llama 4 Maverick replaces Claude as Pilot 1's second explainer; if AWS unblocks Claude, add it as a third explainer on the same Pilot 0 decisions.
- [x] **Preflight** passed 2026-10-28 (`preflight-20261028T035934Z`, $0.0018) with Llama and Gemini as explainers; projected Pilot 1 cost ≈ $0.31. Smoke tests of `case_writer`, `case_writer_2`, `explainer_open`, `decision_alt` passed.
  - [x] Gemini 3.1 Flash-Lite accepts `thinking_level: minimal`; it reported 0 thinking tokens.
  - [x] gpt-oss-120b's response reports `serviceTier: flex`.
  - [ ] Claude Haiku 5.5: blocked (above).
- [ ] **Credits:** in AWS *Billing → Credits*, check that credits cover the Bedrock-billed models (and Claude, billed through AWS Marketplace, if it is unblocked); in GCP *Billing → Credits*, check that they cover the Gemini API project.
- [ ] **Budget alerts:** AWS Budgets and a GCP budget alert, e.g. at $40 each.
- [ ] **Commit** before live runs, so each run's manifest names a clean git commit.

## Model choices (decided 2026-10-27, research_docs/11)

- [x] **Jev:** `jev-latest` on TypeSafe's native API ($0.042 per 1M input tokens, output free). It pointed to `jev-1.14.0` on 2026-10-26.
  - [ ] If TypeSafe ships a new release mid-project, decide whether to stay on `jev-1.14.0` by its fixed ID or start fresh runs on the new version.
- [x] **Case writers:** Qwen3 235B (`case_writer`) and Mistral Large 3 (`case_writer_2`) on Bedrock.
- [x] **Explainers:** Gemini 3.1 Flash-Lite (Gemini API) and Llama 4 Maverick (Bedrock; also the stretch goal's open-weight model). Claude Haiku 5.5 (Bedrock) was the first choice and stays configured as `explainer_bedrock`, but is blocked on this AWS account; Llama replaced it in Pilot 1 on 2026-10-28.
- [x] **Simulator and judge:** gpt-oss-120b on Bedrock, flex tier, reasoning effort `low`.
- [x] **Second decision model for Experiment 6:** Amazon Nova Lite on Bedrock.
- [x] **Prices** from AWS's official Price List files (us-east-2, published 2026-10-12) and Google's pricing page (checked 2026-10-27). Re-check before large runs.

## Benchmark and experiments

- [ ] **Audit name list** for name-swap cues. Benchmark v0 uses first names from the title of Bertrand & Mullainathan (2004) as a placeholder ([src/sysone/benchmark/generator.py](src/sysone/benchmark/generator.py)). Choosing a different list means freezing a new benchmark version.
- [x] **Step 7:** live Pilot 0 run on refund_v0 (`pilot0-v0-2026-10-28`, $0.0137): no-go, 6 customer-harming wrong decisions of 60 (20 needed).
- [x] **refund_v1** frozen 2026-10-28: 24 incorrect-input cases with the receipt and condition corruptions Jev follows.
- [x] **Step 7b:** Pilot 0 on refund_v1 (`pilot0-v1-2026-10-28`, $0.0160): go, 24 customer-harming wrong decisions.
- [ ] **Missing-receipt cases:** Jev answered "undetermined" in only 7 of 12 samples where the receipt status is missing. Note for Experiment 5 (abstention).
- [x] **Stability threshold** for go / no-go: at least 90% of variants unanimous across repeats (set 2026-10-28; both runs ≈ 97%).
- [x] **E0 write-up** with figures: [research_analyses/E0_jev_characterization/](research_analyses/E0_jev_characterization/README.md).
- [ ] **Jev's date-window errors** (approves requests up to 52 days old against a 30-day window): decide whether to study them as their own category, e.g. in Experiment 5.
- [ ] **Step 6b:** build the natural-language realization with the case writer, frozen as `refund_v2` (≈ $0.10).
- [x] **Pilot 1 harness** built 2026-10-27, dry run passed; preflight passed 2026-10-28. Ready for the live run.
- [x] **Pilot 1:** how a frozen explanation's implied decision on each hidden variant is obtained (decided 2026-10-27): a simulator model with a structured response, scored against a no-explanation control, plus a structured-explanation condition run directly against the records. See [research_docs/12](research_docs/12_first_experiments_roadmap.md#how-an-explanations-implied-decision-is-obtained).
- [ ] **Pilot 1 contestability scoring** against the verified defects, with the judge.
- [ ] **Re-estimate costs** from Pilot 0 and Pilot 1's measured costs before each later experiment, against the allocations in [configs/budget.yaml](configs/budget.yaml).
- [ ] **Small-run option** (`--max-cases`) for trying a few cases before a full run (offered, not built).
- [ ] **Within-run Jev query reuse** for Experiment 2 (planned in research_docs/11 *Project budget*).

## Setup and verification

- [x] Project caps per provider: AWS $50, GCP $50, TypeSafe $5, enforced across all runs ([configs/budget.yaml](configs/budget.yaml)).
- [ ] **Bedrock structured output:** Bedrock models get the JSON format in the prompt and their reply is parsed. Wire up Converse's structured output if parse rates are low.
- [ ] Create a dedicated TypeSafe key if the current one is shared.
