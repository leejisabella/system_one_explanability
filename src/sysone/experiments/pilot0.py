"""Pilot 0: query Jev on every variant of every benchmark case, several samples each.

Usage:
  uv run python -m sysone.experiments.pilot0 --dry-run          # fake Jev, no network, no cost
  uv run python -m sysone.experiments.pilot0 --run-id <id>      # live run (needs TYPESAFE_API_KEY)

Re-running with the same --run-id resumes: rows already in decisions.jsonl are skipped,
and the run's earlier spend still counts toward its $25 cap.

The run is locked to one Jev version. The first version that answers is saved in the
manifest as `model_version` (or set it up front with `expected_model_version` in the
config). Any later answer from another version, including on resume, halts the run.
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
from sysone.benchmark.refund_policy import PolicyParams, evaluate, policy_text
from sysone.benchmark.render import render_case_text
from sysone.benchmark.store import load_benchmark
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.clients.base import (
    GuardedClient,
    ModelNotConfigured,
    ModelSpec,
    ModelVersionChanged,
    ProviderError,
    VersionLock,
)
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
    # Version the run must stay on, e.g. "jev-1.14.0". None: lock to whichever version
    # answers first. Either way the run halts if a different version answers.
    expected_model_version: str | None = None
    # Rotate the option order with the sample index, so each option is listed first
    # equally often. Set false for a pure noise-floor run, where repeats are identical.
    rotate_option_order: bool = True
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
    params = PolicyParams(**meta["policy"])
    policy = policy_text(params)
    if not config.use_cache:
        client.cache = None

    # A resumed run stays on the version it started with.
    locked = run.manifest().get("model_version")
    if locked and config.expected_model_version and locked != config.expected_model_version:
        raise ValueError(
            f"run {run.run_id} is locked to {locked}, but the config expects {config.expected_model_version}; "
            "start a new run_id to use another version"
        )
    lock = VersionLock(locked or config.expected_model_version, on_lock=lambda v: run.update_manifest(model_version=v))
    client.version_lock = lock
    jev = JevDecisionModel(client, rotate_options=config.rotate_option_order)
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
        model_version=lock.expected,
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
                # The rule that decides this variant's record (the case's decisive_rule
                # is the one that decides the true record).
                variant_rule = evaluate(variant.record, params).rule
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
                        "decisive_rule": case.decisive_rule,
                        "variant_rule": variant_rule,
                        "model_requested": spec.model,
                    }
                    tags = {"experiment": EXPERIMENT, "case_id": case.case_id,
                            "variant_id": variant.variant_id, "sample_index": str(sample)}
                    try:
                        result = jev.decide(policy, case_text, sample_index=sample, tags=tags)
                    except ModelVersionChanged as e:
                        # The call is in the ledger; record it here too, then halt.
                        writer.append({**base, "status": "error", "error": str(e), "model_used": e.got, "ts": now_iso()})
                        raise
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
                        "p_approve": d.p_approve,
                        "probabilities": d.probabilities,
                        "confidence": d.confidence,
                        "parse_ok": d.parse_ok,
                        "parse_error": d.parse_error,
                        "raw_answer": d.raw_answer,
                        "option_order": list(result.option_order),
                        # The exact version that answered (the request may name an alias).
                        "model_used": resp.model_used,
                        "cost_usd": resp.cost_usd,
                        "cached": resp.cached,
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "request_key": result.request_key,
                        "ts": now_iso(),
                    })
    except (BudgetError, _TooManyErrors, ModelVersionChanged) as e:
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
        spec = ModelSpec("jev", "fake", "jev-latest", Price(0.0, 0.0))
        client = GuardedClient(spec, FakeDecisionProvider(), registry.guard, cache=None)
    else:
        try:
            client = registry.client(config.model_alias)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without a key.") from None

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
