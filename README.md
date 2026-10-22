# system_one_explanability
Exploring model misalignment, System One model information availability, and decision explainability frameworks

- **Picking this up? Start with [STATUS.md](STATUS.md)**: what exists, what's blocking, what to do next
- Research questions, framework, experiments and references: [research_docs/](research_docs/README.md)
- Harness design, model access and cost controls: [research_docs/11_harness_and_infrastructure.md](research_docs/11_harness_and_infrastructure.md)
- Open decisions and setup tasks: [TODO.md](TODO.md)

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # install dependencies
uv run pytest         # offline tests; no network calls, no cost
```

### Credentials

| Provider | Used for | How to configure |
|----------|----------|------------------|
| OpenRouter | Jev | `OPENROUTER_API_KEY` |
| Amazon Bedrock (us-east-2) | Case writer (Claude Opus 4.5), Bedrock explainers | Standard AWS credentials (`AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`). Claude Opus 4.5 model access must be enabled in the Bedrock console for us-east-2. |
| Google Gemini | Gemini explainers, judge | `GEMINI_API_KEY` (or a Vertex AI project; see TODO.md) |

Keep keys in your shell or a `.env` file (gitignored). Never commit them.

### Pilot 0 (first experiment)

```bash
uv run python -m sysone.experiments.pilot0 --dry-run   # fake Jev, no network, no cost
```

Results go to `results/runs/<run_id>/` (kept locally, not in git). See [research_docs/12_first_experiments_roadmap.md](research_docs/12_first_experiments_roadmap.md).

### Live smoke test

```bash
uv run python scripts/smoke_test.py case_writer
```

This makes one short real request (well under $0.01) and prints the cost and the path of the spending log.

## Cost limits

Every outside API request is capped at **$2.00** and every experiment run at **$25.00** ([configs/budget.yaml](configs/budget.yaml)). Requests that could exceed either limit are refused before they are sent. Models with placeholder IDs (`TODO...`) or no price entry are never called. Spending logs go to `results/runs/<run_id>/ledger.jsonl`; see the harness doc for details.
