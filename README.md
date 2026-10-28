# system_one_explanability
Exploring model misalignment, System One model information availability, and decision explainability frameworks

- **Picking this up? Start with [STATUS.md](STATUS.md)**: what exists, what's blocking, what to do next
- Research questions, framework, experiments and references: [research_docs/](research_docs/README.md)
- Harness design, model access and cost controls: [research_docs/11_harness_and_infrastructure.md](research_docs/11_harness_and_infrastructure.md)
- Results, one folder per experiment: [research_analyses/](research_analyses/README.md)
- Open decisions and setup tasks: [TODO.md](TODO.md)

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # install dependencies
uv run pytest         # offline tests; no network calls, no cost
```

### Credentials

Put keys in `.env` (gitignored; copy [.env.example](.env.example)). The experiment runners and the smoke test load it; variables already set in your shell take precedence.

| Provider | Used for | How to configure |
|----------|----------|------------------|
| TypeSafe | Jev (`jev-latest`, native API) | `TYPESAFE_API_KEY` |
| Amazon Bedrock (us-east-2) | Case writers (Qwen3 235B, Mistral Large 3), explainers (Llama 4 Maverick; Claude Haiku 5.5, blocked on this account for now), simulator and judge (gpt-oss-120b), Experiment 6 decision model (Nova Lite) | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or `AWS_PROFILE`. IAM policy below. Claude also needs Anthropic's one-time use-case form in the Bedrock console and an AWS Marketplace subscription, which AWS currently refuses for this account (see [TODO.md](TODO.md)). |
| Google Cloud: Vertex AI (now Gemini Enterprise Agent Platform), global endpoint | Gemini explainer (Gemini 3.1 Flash-Lite) | `GEMINI_VERTEX_PROJECT` (the GCP project ID, billed to its billing account), with the Vertex AI API (`aiplatform.googleapis.com`) enabled and a one-time `gcloud auth application-default login` plus `gcloud auth application-default set-quota-project <project-id>`; tick every scope on the consent page. Without `GEMINI_VERTEX_PROJECT`, `GEMINI_API_KEY` is used, but an education billing account can't move an AI Studio key off the free tier (500 requests per day). |

Never commit keys.

IAM policy for the Bedrock user. The `aws-marketplace` actions are needed only for the first call to a Marketplace-sold model (here, Claude), when Bedrock subscribes the account automatically:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:ListFoundationModels",
      "bedrock:ListInferenceProfiles",
      "bedrock:GetFoundationModelAvailability",
      "aws-marketplace:Subscribe",
      "aws-marketplace:Unsubscribe",
      "aws-marketplace:ViewSubscriptions"
    ],
    "Resource": "*"
  }]
}
```

### Pilot 0 and Pilot 1

```bash
uv run python -m sysone.experiments.pilot0 --dry-run                          # fake Jev, no network, no cost
uv run python -m sysone.experiments.pilot1 --dry-run --pilot0-run <pilot0 id> # fake explainers and simulator
```

Results go to `results/runs/<run_id>/` (kept locally, not in git). See [research_docs/12_first_experiments_roadmap.md](research_docs/12_first_experiments_roadmap.md).

### Live checks

```bash
uv run python scripts/preflight.py                # one real request of every kind the pilots make; projects full-run cost
uv run python scripts/smoke_test.py jev           # TypeSafe: one yes/no question
uv run python scripts/smoke_test.py <alias>       # any chat model in configs/models.yaml: a one-word reply
```

Each makes one short real request (well under $0.01) and prints the cost, the service tier served, the project spend so far, and the path of the spending log. For Jev it also prints the exact version that answered.

## Cost limits

Every outside API request is capped at **$2.00**, every experiment run at **$25.00**, and the whole project at **$50 on AWS, $50 on GCP and $5 on TypeSafe**, summed across all runs ([configs/budget.yaml](configs/budget.yaml)). Requests that could exceed any limit are refused before they are sent. Models with placeholder IDs (`TODO...`) or no price entry are never called. Spending logs go to `results/runs/<run_id>/ledger.jsonl`; see the harness doc for details.
