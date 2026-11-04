"""Preflight before E.102: a few real requests of every kind it makes, on real items of each sample.

Usage:
  uv run python scripts/preflight_e102.py              # live: about 25 requests, a few cents at most
  uv run python scripts/preflight_e102.py --dry-run    # fake models: no network, no cost

Per sample: Jev on 2 items (the first and the longest); the reviewer alone and reviewing a wrong
label on both; each reference model alone on both. Reports parsing, tokens, cost, latency and tier,
then projects the run's cost: Stage 1 (Jev's screen) exactly, Stage 2 from assumed Jev error rates
(E.101's where known), which the real screen replaces.
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
from sysone.experiments.e101 import load_samples
from sysone.experiments.e102 import E102Config
from sysone.registry import CONFIG_DIR, Registry

# Jev's error rate assumed for Stage 2's projection: E.101's measurements, and a guess for the new view.
ASSUMED_JEV_ERROR = {"civil_toxicity_pool_v1": 0.26, "wildguard_prompt_full_v1": 0.09, "wildguard_response_full_v1": 0.15}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    config = E102Config.load(CONFIG_DIR / "experiments" / "e102.yaml")
    samples, _ = load_samples(config)

    run_id = f"preflight-e102-{'dryrun-' if args.dry_run else ''}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
    failures = 0

    def report(key: str, ok: bool, response, error: str | None = None) -> None:
        nonlocal failures
        truncated = str(response.finish_reason or "").lower() in {"max_tokens", "length"}
        failures += (not ok) or truncated
        u = response.usage
        print(f"  {key:58} {'OK  ' if ok and not truncated else 'FAIL'} in/out {u.input_tokens}/{u.output_tokens} "
              f"${response.cost_usd:.6f}" + (f" {response.latency_s:.1f}s" if response.latency_s is not None else "")
              + (f" tier={response.service_tier}" if response.service_tier else "")
              + ("  CUT OFF" if truncated else "") + (f"  {error}" if error else ""))
        costs[key.rsplit("/", 1)[0]].append(response.cost_usd)

    jev = client(config.jev)
    reviewer = Reviewer  # built per task below
    try:
        for name, items in samples.items():
            task = TASKS[items[0].task]
            picks = [items[0], max(items, key=lambda i: len(i.text))]
            for item in picks:
                d = JevDecisionModel(jev, task=task.jev).decide(task.policy, item.text,
                                                                tags={"experiment": "preflight", "step": "review", "item_id": item.item_id})
                report(f"jev/{name}/{item.item_id}", d.decision.parse_ok, d.response, d.decision.parse_error)
            r = reviewer(client(config.reviewer), task, temperature=config.reviewer_temperature,
                         reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                         max_output_tokens=config.reviewer_max_output_tokens)
            refs = {a: reviewer(client(a), task, max_output_tokens=config.reference_max_output_tokens) for a in config.references}
            for item in picks:
                wrong = next(lab for lab in task.labels if lab != item.label)
                for kind, proposal in (("alone", None), ("review", wrong)):
                    res = r.decide(item.text, proposal, tags={"experiment": "preflight", "step": "review", "item_id": item.item_id})
                    report(f"{config.reviewer}:{kind}/{name}/{item.item_id}", res.parse_ok, res.response, res.parse_error)
                for alias, ref in refs.items():
                    res = ref.decide(item.text, tags={"experiment": "preflight", "step": "review", "item_id": item.item_id})
                    report(f"{alias}:alone/{name}/{item.item_id}", res.parse_ok, res.response, res.parse_error)
    except (ProviderError, BudgetError) as e:
        failures += 1
        print(f"  STOPPED: {e}")

    def per_call(key: str) -> float:
        return sum(costs[key]) / len(costs[key]) if costs.get(key) else 0.0

    print("\nProjected cost (no retries counted). Stage 2 assumes Jev's error rate per sample:", ASSUMED_JEV_ERROR)
    stage1 = stage2 = 0.0
    for name, items in samples.items():
        n = len(items)
        s1 = n * config.jev_samples * per_call(f"jev/{name}")
        err = ASSUMED_JEV_ERROR.get(name, 0.2)
        n_wrong = err * n
        # Items reviewed: Jev-wrong, matched-wrong (same count, overlapping by chance) and the subsample.
        union = n * (1 - (1 - err) ** 2) + config.subsample_size * (1 - err) ** 2
        reviewer_calls = union * config.alone_samples
        review_calls = 2 * n_wrong + config.subsample_size
        s2 = (reviewer_calls * per_call(f"{config.reviewer}:alone/{name}") + review_calls * per_call(f"{config.reviewer}:review/{name}")
              + sum(config.subsample_size * config.reference_samples * per_call(f"{a}:alone/{name}") for a in config.references))
        stage1, stage2 = stage1 + s1, stage2 + s2
        print(f"  {name:28} Stage 1: {n * config.jev_samples:6.0f} Jev calls ${s1:.4f}   Stage 2: about "
              f"{reviewer_calls + review_calls:6.0f} reviewer calls + {config.subsample_size * config.reference_samples * len(config.references)} "
              f"reference calls ${s2:.4f}")
    print(f"  total ≈ ${stage1:.4f} (TypeSafe, Stage 1) + ${stage2:.4f} (Bedrock, Stage 2); caps {config.run_provider_caps_usd}")
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILED'}; spent ${registry.guard.spent_usd:.6f}; "
          f"project spend now: {registry.project_status()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
