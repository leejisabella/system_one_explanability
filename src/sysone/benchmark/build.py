"""Generate and freeze a refund benchmark.

Usage:  uv run python -m sysone.benchmark.build --name refund_v0 --seed 0 --per-type 10
"""

from __future__ import annotations

import argparse
import dataclasses

from sysone.benchmark.generator import GENERATOR_VERSION, RefundCaseGenerator
from sysone.benchmark.refund_policy import PolicyParams
from sysone.benchmark.store import freeze_benchmark
from sysone.registry import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-type", type=int, default=10)
    args = parser.parse_args()

    params = PolicyParams()
    cases = RefundCaseGenerator(args.seed, params).generate(args.per_type)
    out = freeze_benchmark(
        cases,
        REPO_ROOT / "benchmarks" / args.name,
        {
            "generator_version": GENERATOR_VERSION,
            "seed": args.seed,
            "per_type": args.per_type,
            "policy": dataclasses.asdict(params),
            "text": "template",
        },
    )
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
