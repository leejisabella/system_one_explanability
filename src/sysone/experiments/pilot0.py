"""Pilot 0: query Jev on every variant of every benchmark case, several samples each.

Usage:
  uv run python -m sysone.experiments.pilot0 --dry-run          # fake Jev, no network, no cost
  uv run python -m sysone.experiments.pilot0 --run-id <id>      # live run (needs the Jev model ID)

Re-running with the same --run-id resumes: rows already in decisions.jsonl are skipped,
and the run's earlier spend still counts toward its $25 cap.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.pilot0 import analyze_run
from sysone.benchmark.refund_policy import PolicyParams, policy_text
from sysone.benchmark.render import render_case_text
from sysone.benchmark.store import load_benchmark
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeDecisionProvider
from sysone.decision.jev import JevDecisionModel
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, Registry
from sysone.runs import Run, git_info, now_iso

EXPERIMENT = "pilot0"


@dataclass(frozen=True)
class Pilot0Config:
    benchmark: str
    model_alias: str = "jev"
    samples_per_variant: int = 3
    temperature: float | None = None
    max_output_tokens: int = 256
    reasoning_max_tokens: int | None = None
    structured_output: bool = True
    # Off by default: a new run should query Jev afresh, not reuse answers from earlier
    # runs (which would hide changes in Jev). Resuming a run never needs the cache.
    use_cache: bool = False
    # Stop the run after this many failed requests in a row (e.g. a rejected parameter).
    max_consecutive_errors: int = 10

    @classmethod
    def load(cls, path: str | Path) -> Pilot0Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


class _TooManyErrors(RuntimeError):
    pass


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["case_id"], row["variant_id"], row["sample_index"]


def run_pilot0(config: Pilot0Config, run: Run, client: GuardedClient, *, benchmark_dir: Path) -> str:
    """Run (or resume) Pilot 0 and return the final run status."""
    cases, meta = load_benchmark(benchmark_dir)
    policy = policy_text(PolicyParams(**meta["policy"]))
    if not config.use_cache:
        client.cache = None
    jev = JevDecisionModel(
        client,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        reasoning_max_tokens=config.reasoning_max_tokens,
        structured_output=config.structured_output,
    )
    writer = run.writer("decisions")
    # Only successful rows count as done, so failed requests are retried on resume.
    done = {row_key(r) for r in writer.rows() if r["status"] == "ok"}

    spec = client.spec
    run.update_manifest(
        experiment=EXPERIMENT,
        status="running",
        started_at=run.manifest().get("started_at", now_iso()),
        resumed_at=now_iso() if done else None,
        benchmark={"path": str(benchmark_dir), "name": meta["name"], "sha256": meta["sha256"]},
        config=dataclasses.asdict(config),
        model={
            "alias": spec.alias,
            "provider": spec.provider,
            "model": spec.model,
            "price_per_m": dataclasses.asdict(spec.price) if spec.price else None,
        },
        budget={"per_call_cap_usd": client.guard.per_call_cap_usd, "run_cap_usd": client.guard.run_cap_usd},
        git=git_info(),
        config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )

    status = "completed"
    consecutive_errors = 0
    try:
        for case in cases:
            for variant in case.variants:
                case_text = render_case_text(variant)
                for sample in range(config.samples_per_variant):
                    if (case.case_id, variant.variant_id, sample) in done:
                        continue
                    base = {
                        "run_id": run.run_id,
                        "case_id": case.case_id,
                        "case_type": str(case.case_type),
                        "variant_id": variant.variant_id,
                        "variant_kind": str(variant.kind),
                        "sample_index": sample,
                        "policy_outcome": str(variant.policy_outcome),
                        "true_outcome": str(case.true_outcome),
                        "presented_outcome": str(case.presented_outcome),
                        "model_requested": spec.model,
                    }
                    tags = {"experiment": EXPERIMENT, "case_id": case.case_id,
                            "variant_id": variant.variant_id, "sample_index": str(sample)}
                    try:
                        result = jev.decide(policy, case_text, sample_index=sample, tags=tags)
                    except ProviderError as e:
                        writer.append({**base, "status": "error", "error": str(e), "ts": now_iso()})
                        consecutive_errors += 1
                        if consecutive_errors >= config.max_consecutive_errors:
                            raise _TooManyErrors(f"{consecutive_errors} failed requests in a row; last: {e}") from e
                        continue
                    consecutive_errors = 0
                    d, resp = result.decision, result.response
                    writer.append({
                        **base,
                        "status": "ok",
                        "decision": d.label,
                        "confidence": d.confidence,
                        "parse_ok": d.parse_ok,
                        "parse_error": d.parse_error,
                        "raw_text": d.raw_text,
                        "model_used": resp.model_used,
                        "finish_reason": resp.finish_reason,
                        "cost_usd": resp.cost_usd,
                        "cached": resp.cached,
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "reasoning_tokens": resp.usage.reasoning_tokens,
                        "request_key": result.request_key,
                        "ts": now_iso(),
                    })
    except (BudgetError, _TooManyErrors) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(client.guard.spent_usd, 6))
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "pilot0.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="use a fake Jev: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = Pilot0Config.load(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"

    registry = Registry(run_id, results_dir=Path(args.results_dir))
    if args.dry_run:
        spec = ModelSpec("jev", "fake", "fake/jev", Price(0.0, 0.0))
        client = GuardedClient(spec, FakeDecisionProvider(), registry.guard, cache=None)
    else:
        try:
            client = registry.client(config.model_alias)
        except ModelNotConfigured as e:
            raise SystemExit(f"{e}\nUse --dry-run until the Jev model ID is set.") from None

    run = Run(run_id, args.results_dir)
    status = run_pilot0(config, run, client, benchmark_dir=REPO_ROOT / config.benchmark)
    summary = analyze_run(run.dir)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())
    if summary["n_rows"] == 0:
        print("no decisions recorded")


if __name__ == "__main__":
    main()
