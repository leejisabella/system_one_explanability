"""Preflight before Part 2's paid steps (E.104, E.105) and before evaluating on the GPU (E.106).

Usage:
  uv run python scripts/preflight_part2.py              # live: about 20 requests, a few cents at most
  uv run python scripts/preflight_part2.py --dry-run    # fake models: no network, no cost
  uv run python scripts/preflight_part2.py --local base_local   # only: is the vLLM server up, and does it reason?

Per training pool, on 2 items (the first and the longest): Jev decides once (E.104), and the reviewer
and the teacher review a wrong label (E.105). Checks that every reply parses and that the reviewer's
and teacher's reasoning comes back (E.105 can't export traces without it), then projects E.104's and
E.105's cost from the measured cost per call. With --local, sends 2 alone decisions to that alias.
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
from sysone.experiments.e101 import _usage, load_samples
from sysone.experiments.e104 import E104Config
from sysone.experiments.e105 import E105Config
from sysone.registry import CONFIG_DIR, Registry

# Assumptions for E.105's projection, to re-check against E.104's real mix counts: most examples
# stop after 2 reviewer tries and hard ones go to 4; the teacher sees about a tenth of examples;
# Jev is wrong on about 15% of the training pools (E.102's test pools: 25%, 10% and 8%).
REVIEWER_CALLS_PER_EXAMPLE = 2.4
TEACHER_SHARE = 0.10
JEV_ERROR_RATE = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", metavar="ALIAS", help="check a self-hosted reviewer instead")
    args = parser.parse_args()
    load_env()
    c4 = E104Config.load(CONFIG_DIR / "experiments" / "e104.yaml")
    c5 = E105Config.load(CONFIG_DIR / "experiments" / "e105.yaml")
    samples, _ = load_samples(c4)
    run_id = f"preflight-part2-{'dryrun-' if args.dry_run else ''}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    registry = Registry(run_id, run_provider_caps_usd={"bedrock": 0.25, "typesafe": 0.05})

    def client(alias: str, fake_provider=None) -> GuardedClient:
        if args.dry_run:
            return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0)),
                                 fake_provider or FakeChatProvider(malformed_every=0), registry.guard, cache=None)
        c = registry.client(alias)
        c.cache = None
        return c

    failures = 0
    costs: dict[str, list[float]] = defaultdict(list)

    def report(key: str, ok: bool, response, *, needs_reasoning: bool) -> None:
        nonlocal failures
        reasoning = _usage(response)["reasoning"]
        bad = not ok or (needs_reasoning and not reasoning)
        failures += bad
        u = response.usage
        print(f"  {key:58} {'FAIL' if bad else 'OK  '} in/out {u.input_tokens}/{u.output_tokens} ${response.cost_usd:.6f}"
              + (f" reasoning {len(reasoning)} chars" if reasoning else " no reasoning"))
        costs[key.split("/")[0]].append(response.cost_usd)

    try:
        if args.local:
            c = client(args.local)
            for name, items in samples.items():
                res = Reviewer(c, TASKS[items[0].task], reasoning_max_tokens=4096).decide(items[0].text, tags={"step": "review"})
                report(f"{args.local}/{name}", res.parse_ok, res.response, needs_reasoning=True)
                break
        else:
            jev = client(c4.jev, FakeDecisionProvider(malformed_every=0))
            reviewer, teacher = client(c5.reviewer), client(c5.teacher)
            for name, items in samples.items():
                task = TASKS[items[0].task]
                for item in (items[0], max(items, key=lambda i: len(i.text))):
                    d = JevDecisionModel(jev, task=task.jev).decide(task.policy, item.text, tags={"experiment": "preflight"})
                    report(f"jev/{name}/{item.item_id}", d.decision.parse_ok, d.response, needs_reasoning=False)
                    wrong = next(lab for lab in task.labels if lab != item.label)
                    for alias, c in (("reviewer", reviewer), ("teacher", teacher)):
                        r = Reviewer(c, task, reasoning_max_tokens=c5.reviewer_reasoning_max_tokens,
                                     max_output_tokens=c5.reviewer_max_output_tokens)
                        res = r.decide(item.text, wrong, tags={"experiment": "preflight", "step": "review"})
                        report(f"{alias}/{name}/{item.item_id}", res.parse_ok, res.response, needs_reasoning=True)
    except (ProviderError, BudgetError) as e:
        failures += 1
        print(f"  STOPPED: {e}")

    if not args.local:
        per = lambda k: sum(costs[k]) / len(costs[k]) if costs.get(k) else 0.0
        n_items = sum(len(v) for v in samples.values())
        jev_cost = n_items * c4.jev_samples * per("jev")
        # Per Jev error: one jev_wrong row, jev_right_per_wrong and alone_per_wrong rows. Every row is traced
        # alone (the control arm) and every review-format row (all but the alone group) in review too.
        m = c4.mix_settings
        wrong = JEV_ERROR_RATE * n_items
        rows = wrong * (1 + m.jev_right_per_wrong + m.matched_wrong_per_wrong + m.alone_per_wrong)
        examples = rows + wrong * (1 + m.jev_right_per_wrong + m.matched_wrong_per_wrong)
        e105 = examples * REVIEWER_CALLS_PER_EXAMPLE * per("reviewer") + examples * TEACHER_SHARE * c5.teacher_samples * per("teacher")
        print(f"\nProjected: E.104 Jev ≈ ${jev_cost:.2f} ({n_items} items x {c4.jev_samples}; cap {c4.run_provider_caps_usd}); "
              f"E.105 ≈ ${e105:.2f} (~{examples:.0f} examples; cap {c5.run_provider_caps_usd}). Rough: re-estimate "
              "E.105 from E.104's actual mix counts.")
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILED'}; spent ${registry.guard.spent_usd:.6f}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
