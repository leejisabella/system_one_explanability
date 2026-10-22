"""One tiny live request to a configured model alias, to check credentials and cost logging.

Usage:  uv run python scripts/smoke_test.py case_writer
Spends real money (well under $0.01 for a short reply). Bypasses the cache.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from sysone.registry import Registry
from sysone.types import ChatRequest


def main(alias: str) -> None:
    run_id = "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry = Registry(run_id)
    client = registry.client(alias)
    client.cache = None
    request = ChatRequest.from_prompt(
        "Reply with the single word: ready", max_output_tokens=16, tags={"experiment": "smoke"}
    )
    response = client.complete(request)
    print(f"model_used={response.model_used} text={response.text!r}")
    print(f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens} cost=${response.cost_usd:.6f}")
    print(f"ledger: {registry.guard.ledger.path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case_writer")
