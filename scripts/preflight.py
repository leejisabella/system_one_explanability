"""Preflight before live runs: one real request of every kind Pilot 0 and Pilot 1 make, on a real case.

Usage:
  uv run python scripts/preflight.py                  # every role: jev, Pilot 1's explainers, simulator
  uv run python scripts/preflight.py --only jev       # just Jev (e.g. before AWS is ready)
  uv run python scripts/preflight.py --dry-run        # fake models: no network, no cost

Sends: one Jev decision; per explainer, one decision, one free-form and one structured
explanation; and two simulations (the no-explanation control and one with the structured
explanation). That exercises the real prompts, JSON schemas, thinking level and flex tier,
which a one-word smoke test does not. It reports whether each answer parsed, its tokens,
cost and the service tier served, then projects the cost of full Pilot 0 and Pilot 1 runs
from the measured cost per call. Spends real money (well under $0.01 in total), recorded
in results/runs/preflight-<time>/ and counted toward the project caps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone

import yaml

from sysone.benchmark.refund_policy import PolicyParams, policy_text
from sysone.benchmark.render import render_case_text
from sysone.benchmark.store import load_benchmark
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.clients.base import GuardedClient, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.decision.jev import JevDecisionModel
from sysone.env import load_env
from sysone.explain.explainer import FREE_FORM, STRUCTURED, ExplainerModel
from sysone.explain.simulator import Simulator
from sysone.registry import CONFIG_DIR, REPO_ROOT, Registry

# Explainer roles are the aliases in configs/experiments/pilot1.yaml.
EXPLAINERS = tuple(yaml.safe_load((CONFIG_DIR / "experiments" / "pilot1.yaml").read_text())["explainers"])
ROLES = ("jev", *EXPLAINERS, "simulator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()

    p1 = yaml.safe_load((CONFIG_DIR / "experiments" / "pilot1.yaml").read_text())
    p0 = yaml.safe_load((CONFIG_DIR / "experiments" / "pilot0.yaml").read_text())
    cases, meta = load_benchmark(REPO_ROOT / p0["benchmark"])
    policy = policy_text(PolicyParams(**meta["policy"]))
    # A case with hidden variants, so the simulator has a real variant to predict.
    case = next(c for c in cases if len(c.variants) > 1)
    original, variant = render_case_text(case.original), render_case_text(case.variants[1])

    run_id = f"preflight-{'dryrun-' if args.dry_run else ''}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    registry = Registry(run_id)
    print(f"run {run_id}; project spend so far: {registry.project_status()}")

    def client(alias: str) -> GuardedClient:
        if args.dry_run:
            fake = FakeDecisionProvider(malformed_every=0) if alias == "jev" else FakeChatProvider(malformed_every=0)
            return GuardedClient(ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0)), fake, registry.guard,
                                 cache=None)
        c = registry.client(alias)
        c.cache = None
        return c

    costs: dict[str, list[float]] = defaultdict(list)
    failures = 0

    def report(alias: str, kind: str, ok: bool, response, error: str | None = None) -> None:
        nonlocal failures
        failures += not ok
        u = response.usage
        truncated = str(response.finish_reason or "").upper() in {"MAX_TOKENS", "LENGTH", "FINISHREASON.MAX_TOKENS"}
        failures += truncated and ok
        print(f"  {alias:18} {kind:22} {'OK  ' if ok and not truncated else 'FAIL'} in/out {u.input_tokens}/{u.output_tokens} "
              f"(reasoning {u.reasoning_tokens}) ${response.cost_usd:.6f}"
              + (f" tier={response.service_tier}" if response.service_tier else "")
              + ("  CUT OFF at the output cap" if truncated else "")
              + (f"  {error}" if error else ""))
        costs[f"{alias}/{kind}"].append(response.cost_usd)

    jev_label, structured_text = "deny", None
    try:
        if "jev" in args.only:
            d = JevDecisionModel(client("jev")).decide(policy, original, tags={"experiment": "preflight"})
            report("jev", "decision", d.decision.parse_ok, d.response, d.decision.parse_error)
            print(f"  {'':18} version {d.response.model_used}, probabilities {d.decision.probabilities}")
            jev_label = d.decision.label or jev_label
        for alias in EXPLAINERS:
            if alias not in args.only:
                continue
            explainer = ExplainerModel(client(alias), temperature=p1["explainer_temperature"],
                                       reasoning_max_tokens=p1["explainer_reasoning_max_tokens"])
            tags = {"experiment": "preflight", "explainer": alias}
            d = explainer.decide(policy, original, tags={**tags, "step": "explainer_decision"})
            report(alias, "decision", d.parse_ok, d.response, d.parse_error)
            for condition in (FREE_FORM, STRUCTURED):
                e = explainer.explain(condition, policy, original, jev_label,
                                      tags={**tags, "step": "explanation", "condition": condition})
                report(alias, f"explanation:{condition}", e.parse_ok, e.response, e.parse_error)
                if condition == STRUCTURED and e.parse_ok:
                    structured_text = e.text
        if "simulator" in args.only:
            sim = Simulator(client("simulator"), temperature=p1["simulator_temperature"],
                            reasoning_max_tokens=p1["simulator_reasoning_max_tokens"], seed=p1["simulator_seed"])
            for kind, text in (("simulation:control", None), ("simulation:explained", structured_text or "No explanation.")):
                s = sim.simulate(policy, original, jev_label, text, variant,
                                 tags={"experiment": "preflight", "step": "simulation"})
                report("simulator", kind, s.parse_ok, s.response, s.parse_error or s.basis_error)
    except (ProviderError, BudgetError) as e:
        failures += 1
        print(f"  STOPPED: {e}")

    # Full-run volumes on this benchmark (research_docs/12, "Pilot 1 as built").
    n_cases, n_variants = len(cases), sum(len(c.variants) for c in cases)
    n_hidden = n_variants - n_cases
    n_explainers = len(p1["explainers"])
    volume = {
        "jev/decision": n_variants * p0["samples_per_variant"],
        **{f"{a}/decision": n_variants * p1["explainer_decision_samples"] for a in EXPLAINERS},
        **{f"{a}/explanation:{c}": n_cases for a in EXPLAINERS for c in (FREE_FORM, STRUCTURED)},
        "simulator/simulation:control": n_hidden,
        "simulator/simulation:explained": n_hidden * (2 * n_explainers + 1),
    }
    print("\nProjected cost of full runs from these measurements (no retries or failures counted):")
    total = 0.0
    for key, n in volume.items():
        if costs.get(key):
            per_call = sum(costs[key]) / len(costs[key])
            total += per_call * n
            print(f"  {key:42} {n:5} calls × ${per_call:.6f} = ${per_call * n:.4f}")
    print(f"  total ≈ ${total:.4f}  (Pilot 0 = the jev line; Pilot 1 = the rest)")
    print(f"\n{'ALL OK' if not failures else f'{failures} FAILED'}; spent ${registry.guard.spent_usd:.6f}; "
          f"project spend now: {registry.project_status()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
