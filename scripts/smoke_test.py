"""One tiny live request to a configured model alias, to check credentials and cost logging.

Usage:
  uv run python scripts/smoke_test.py jev           # TypeSafe: one yes/no question
  uv run python scripts/smoke_test.py <alias>       # any chat model in configs/models.yaml: a one-word reply,
                                                    # e.g. case_writer, explainer_open, explainer_gemini, simulator

Spends real money (well under $0.01 per call) and counts toward the project caps. Bypasses the cache. For Jev it prints the exact
version that answered, which is what an experiment run will lock to.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from sysone.env import load_env
from sysone.registry import Registry
from sysone.types import ChatRequest, SystemOneRequest


def main(alias: str) -> None:
    load_env()
    run_id = "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry = Registry(run_id)
    client = registry.client(alias)
    client.cache = None
    tags = {"experiment": "smoke"}
    if client.spec.provider == "typesafe":
        request = SystemOneRequest(
            state="The customer writes: I was charged twice for my order this month.",
            questions={"billing": {"type": "noul", "instructions": "Is this message about billing?"}},
            tags=tags,
        )
    else:
        # Models with a reasoning level get a small reasoning allowance on top of the reply.
        reasoning = 256 if client.spec.reasoning_effort else None
        request = ChatRequest.from_prompt("Reply with the single word: ready", max_output_tokens=16,
                                          reasoning_max_tokens=reasoning, tags=tags)
    response = client.complete(request)
    print(f"model_requested={client.spec.model} model_used={response.model_used}")
    print(f"answer={json.dumps(response.raw.get('answers')) if response.raw.get('answers') else repr(response.text)}")
    print(f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens} "
          f"(reasoning {response.usage.reasoning_tokens}) cost=${response.cost_usd:.6f}")
    if client.spec.service_tier:
        print(f"service tier requested={client.spec.service_tier} served={response.service_tier}")
    print(f"project spend so far: {registry.project_status()}")
    print(f"ledger: {registry.guard.ledger.path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "jev")
