"""Preflight before E.101: a few real requests of every kind it makes, on real items.

Usage:
  uv run python scripts/preflight_e101.py              # live: about 20 requests, well under $0.05
  uv run python scripts/preflight_e101.py --dry-run    # fake models: no network, no cost

Per sample: Jev on 2 items; the reviewer alone, with a label, and with a label and confidence,
on the same 2 items (one of them the sample's longest), so the real prompts, reasoning effort and
flex tier are exercised. Reports whether each answer parsed, its tokens, cost, latency and the
tier served, then projects the full run's cost from the measured cost per call (by kind and
sample, since WildGuard's prompts run longer). Recorded in results/runs/preflight-e101-<time>/
and counted toward the project caps.
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
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.decision.jev import JevDecisionModel
from sysone.env import load_env
from sysone.experiments.e101 import E101Config, load_samples
from sysone.registry import CONFIG_DIR, Registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    config = E101Config.load(CONFIG_DIR / "experiments" / "e101.yaml")
    samples, _ = load_samples(config)

    run_id = f"preflight-e101-{'dryrun-' if args.dry_run else ''}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    registry = Registry(run_id, run_provider_caps_usd={"bedrock": 0.25, "typesafe": 0.05})
    print(f"run {run_id}; project spend so far: {registry.project_status()}")

    def client(alias: str) -> GuardedClient:
        if args.dry_run:
            fake = FakeDecisionProvider(malformed_every=0) if alias == config.jev else FakeChatProvider(malformed_every=0)
            return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0)), fake, registry.guard,
                                 cache=None)
        c = registry.client(alias)
        c.cache = None
        return c

    costs: dict[str, list[float]] = defaultdict(list)
    out_tokens: dict[str, list[int]] = defaultdict(list)
    failures = 0

    def report(key: str, ok: bool, response, error: str | None = None) -> None:
        nonlocal failures
        truncated = str(response.finish_reason or "").lower() in {"max_tokens", "length"}
        failures += (not ok) or truncated
        u = response.usage
        print(f"  {key:52} {'OK  ' if ok and not truncated else 'FAIL'} in/out {u.input_tokens}/{u.output_tokens} "
              f"${response.cost_usd:.6f}" + (f" {response.latency_s:.1f}s" if response.latency_s is not None else "")
              + (f" tier={response.service_tier}" if response.service_tier else "")
              + ("  CUT OFF at the output cap" if truncated else "") + (f"  {error}" if error else ""))
        costs[key.rsplit("/", 1)[0]].append(response.cost_usd)
        out_tokens[key.rsplit("/", 1)[0]].append(u.output_tokens)

    jev = client(config.jev)
    reviewers = {a: client(a) for a in config.reviewers}
    try:
        for name, items in samples.items():
            task = TASKS[items[0].task]
            picks = [items[0], max(items, key=lambda i: len(i.text))]
            jev_model = JevDecisionModel(jev, task=task.jev)
            for item in picks:
                d = jev_model.decide(task.policy, item.text, tags={"experiment": "preflight", "item_id": item.item_id})
                report(f"jev/{name}/{item.item_id}", d.decision.parse_ok, d.response, d.decision.parse_error)
            for alias, c in reviewers.items():
                reviewer = Reviewer(c, task, temperature=config.reviewer_temperature,
                                    reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                    max_output_tokens=config.reviewer_max_output_tokens)
                for item in picks:
                    wrong = next(lab for lab in task.labels if lab != item.label)
                    for kind, proposal, conf in (("alone", None, None), ("label", wrong, None), ("conf", wrong, 0.8)):
                        r = reviewer.decide(item.text, proposal, conf,
                                            tags={"experiment": "preflight", "step": "review", "item_id": item.item_id})
                        report(f"{alias}:{kind}/{name}/{item.item_id}", r.parse_ok, r.response, r.parse_error)
    except (ProviderError, BudgetError) as e:
        failures += 1
        print(f"  STOPPED: {e}")

    n_review = len(config.review_conditions)
    print("\nProjected cost of the full E.101 run (no retries or failures counted):")
    total = 0.0
    for name, items in samples.items():
        volume = {f"jev/{name}": len(items) * config.jev_samples}
        for alias in config.reviewers:
            volume[f"{alias}:alone/{name}"] = len(items) * config.alone_samples
            # Label-only conditions (two matched/jev, the flips) and the two with confidence.
            volume[f"{alias}:label/{name}"] = len(items) * (n_review - 2)
            volume[f"{alias}:conf/{name}"] = len(items) * 2
        for key, n in volume.items():
            if costs.get(key):
                per_call = sum(costs[key]) / len(costs[key])
                total += per_call * n
                print(f"  {key:44} {n:5} calls × ${per_call:.6f} = ${per_call * n:.4f} "
                      f"(mean output tokens {sum(out_tokens[key]) / len(out_tokens[key]):.0f})")
    print(f"  total ≈ ${total:.4f}  (caps: {config.run_provider_caps_usd})")
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILED'}; spent ${registry.guard.spent_usd:.6f}; "
          f"project spend now: {registry.project_status()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
