"""Preflight before E.103: every protocol and every upstream model, on real items, a few requests each.

Usage:
  uv run python scripts/preflight_e103.py              # live: about 30 requests, a few cents at most
  uv run python scripts/preflight_e103.py --dry-run    # fake models: no network, no cost

Per sample, on 2 items (the first and the longest): each protocol reviews a wrong label (blind-first
with the other label as its committed answer), and each upstream model decides alone. Reports
parsing, tokens, cost and tier, then projects the run from the source E.102 run's plan.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone

from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import load_samples
from sysone.experiments.e102 import reviewed
from sysone.experiments.e103 import E103Config, protocol_kwargs
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    config = E103Config.load(CONFIG_DIR / "experiments" / "e103.yaml")
    samples, _ = load_samples(config)
    plan = JsonlWriter(RESULTS_DIR / "runs" / config.source_run / "plan.jsonl").rows()

    run_id = f"preflight-e103-{'dryrun-' if args.dry_run else ''}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    registry = Registry(run_id, run_provider_caps_usd={"bedrock": 0.25})
    print(f"run {run_id}; project spend so far: {registry.project_status()}")

    def client(alias: str) -> GuardedClient:
        if args.dry_run:
            return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0)),
                                 FakeChatProvider(malformed_every=0), registry.guard, cache=None)
        c = registry.client(alias)
        c.cache = None
        return c

    costs: dict[str, list[float]] = defaultdict(list)
    failures = 0

    def report(key: str, ok: bool, response, error: str | None = None) -> None:
        nonlocal failures
        truncated = str(response.finish_reason or "").lower() in {"max_tokens", "length"}
        failures += (not ok) or truncated
        u = response.usage
        print(f"  {key:62} {'OK  ' if ok and not truncated else 'FAIL'} in/out {u.input_tokens}/{u.output_tokens} "
              f"${response.cost_usd:.6f}" + (f" tier={response.service_tier}" if response.service_tier else "")
              + ("  CUT OFF" if truncated else "") + (f"  {error}" if error else ""))
        costs[key.rsplit("/", 1)[0]].append(response.cost_usd)

    reviewer = client(config.reviewer)
    upstreams = {a: client(a) for a in config.upstreams}
    try:
        for name, items in samples.items():
            task = TASKS[items[0].task]
            r = Reviewer(reviewer, task, temperature=config.reviewer_temperature,
                         reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                         max_output_tokens=config.reviewer_max_output_tokens)
            for item in (items[0], max(items, key=lambda i: len(i.text))):
                wrong = next(lab for lab in task.labels if lab != item.label)
                for protocol in config.protocols:
                    kw = protocol_kwargs(protocol, jev_error_rate=0.2, own_answer=item.label)
                    res = r.decide(item.text, wrong, **kw, tags={"experiment": "preflight", "step": "review"})
                    report(f"{protocol}/{name}/{item.item_id}", res.parse_ok, res.response, res.parse_error)
                for alias, c in upstreams.items():
                    res = Reviewer(c, task, max_output_tokens=config.upstream_max_output_tokens).decide(
                        item.text, tags={"experiment": "preflight", "step": "review"})
                    report(f"{alias}/{name}/{item.item_id}", res.parse_ok, res.response, res.parse_error)
    except (ProviderError, BudgetError) as e:
        failures += 1
        print(f"  STOPPED: {e}")

    per = lambda key: sum(costs[key]) / len(costs[key]) if costs.get(key) else 0.0
    print("\nProjected cost of the full run (no retries counted):")
    total = 0.0
    for name in samples:
        rows = [p for p in plan if p["sample"] == name]
        n_reviews = sum(not p["jev_correct"] for p in rows) + sum(not p["matched_correct"] for p in rows) \
            + sum(p["in_subsample"] for p in rows)
        line = [f"{name:28}"]
        for protocol in config.protocols:
            c = n_reviews * per(f"{protocol}/{name}")
            total += c
            line.append(f"{protocol} {n_reviews}×=${c:.3f}")
        for alias in config.upstreams:
            c = len(rows) * per(f"{alias}/{name}")
            total += c
            line.append(f"{alias} {len(rows)}×=${c:.3f}")
        print("  " + "  ".join(line))
    print(f"  total ≈ ${total:.4f}  (cap {config.run_provider_caps_usd})")
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILED'}; spent ${registry.guard.spent_usd:.6f}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
