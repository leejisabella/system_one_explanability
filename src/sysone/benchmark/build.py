"""Generate and freeze a benchmark.

Usage:
  uv run python -m sysone.benchmark.build --name refund_v0 --seed 0 --per-type 10
  uv run python -m sysone.benchmark.build --name refund_v1 --seed 1 --per-type 10 \\
      --incorrect-input 24 --incorrect-input-kinds condition has_receipt --id-prefix refund_v1
  uv run python -m sysone.benchmark.build --name refund_v1_1 --seed 2 --per-type 10 \\
      --incorrect-input 24 --incorrect-input-kinds condition has_receipt --threshold-sweep 30 \\
      --id-prefix refund_v1_1
  uv run python -m sysone.benchmark.build --domain moderation --name moderation_v1 --seed 0

refund_v0 was built with the first command (its case IDs use the policy ID as prefix).
refund_v1 adds incorrect-input cases with the corruptions Jev follows (research_docs/12).
refund_v1_1 adds threshold-sweep cases across 25-65 days, where Jev departs from its
policy (Pilot 1). The other domains (sysone.benchmark.domains) build 10 cases of each
type, threshold sweeps included, by default.
"""

from __future__ import annotations

import argparse
import dataclasses

from sysone.benchmark.cases import CaseType
from sysone.benchmark.domains import DOMAINS, DomainCaseGenerator
from sysone.benchmark.generator import GENERATOR_VERSION, INCORRECT_INPUT_KINDS, RefundCaseGenerator
from sysone.benchmark.refund_policy import PolicyParams
from sysone.benchmark.store import freeze_benchmark
from sysone.registry import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="refund", choices=["refund", *DOMAINS])
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--incorrect-input", type=int, help="number of incorrect-input cases (default: --per-type)")
    parser.add_argument("--incorrect-input-kinds", nargs="+", choices=INCORRECT_INPUT_KINDS,
                        default=list(INCORRECT_INPUT_KINDS))
    parser.add_argument("--threshold-sweep", type=int,
                        help="number of threshold-sweep cases (refund default: 0; other domains: --per-type)")
    parser.add_argument("--id-prefix", help="case-ID prefix (default: the policy ID)")
    args = parser.parse_args()
    if args.domain != "refund":
        build_domain(args)
        return

    params = PolicyParams()
    counts = {CaseType.INCORRECT_INPUT: args.incorrect_input} if args.incorrect_input is not None else {}
    if args.threshold_sweep:
        counts[CaseType.THRESHOLD_SWEEP] = args.threshold_sweep
    generator = RefundCaseGenerator(args.seed, params, incorrect_input_kinds=tuple(args.incorrect_input_kinds))
    cases = generator.generate(args.per_type, counts=counts, id_prefix=args.id_prefix)
    out = freeze_benchmark(
        cases,
        REPO_ROOT / "benchmarks" / args.name,
        {
            "generator_version": GENERATOR_VERSION,
            "seed": args.seed,
            "per_type": args.per_type,
            "counts": {str(t): n for t, n in counts.items()},
            "incorrect_input_kinds": list(args.incorrect_input_kinds),
            "id_prefix": args.id_prefix or params.policy_id,
            "policy": dataclasses.asdict(params),
            "text": "template",
            **({"domain": "refund"} if args.threshold_sweep else {}),
        },
    )
    print(f"wrote {len(cases)} cases to {out}")


def build_domain(args: argparse.Namespace) -> None:
    domain = DOMAINS[args.domain]
    counts = {}
    if args.incorrect_input is not None:
        counts[CaseType.INCORRECT_INPUT] = args.incorrect_input
    if args.threshold_sweep is not None:
        counts[CaseType.THRESHOLD_SWEEP] = args.threshold_sweep
    generator = DomainCaseGenerator(domain, args.seed)
    cases = generator.generate(args.per_type, counts=counts, id_prefix=args.id_prefix)
    out = freeze_benchmark(
        cases,
        REPO_ROOT / "benchmarks" / args.name,
        {
            "domain": domain.name,
            "generator_version": domain.generator_version,
            "seed": args.seed,
            "per_type": args.per_type,
            "counts": {str(t): n for t, n in counts.items()},
            "id_prefix": args.id_prefix or generator.params.policy_id,
            "policy": dataclasses.asdict(generator.params),
            "task": dataclasses.asdict(domain.task),
            "text": "template",
        },
    )
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
