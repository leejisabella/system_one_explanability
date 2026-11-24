"""E.119: capability check. Does a stronger reviewer fix the cascade?

E.102's design on E.102's frozen plan, with gpt-oss-120b as the reviewer instead of gpt-oss-20b:
the same prompt, settings, items, Jev labels, matched random labels and subsample. Only the
reviewer's size changes. No Jev calls.

  review  every reviewed item alone x alone_samples, plus one review of each proposal (Jev's wrong
          label, the matched random wrong label, the subsample's correct label), exactly as E.102.

The run folder gets copies of E.102's plan.jsonl and jev_decisions.jsonl, so E.102's own analysis
(sysone.analysis.e102) runs on it unchanged; sysone.analysis.e119 then compares the two reviewers
on the same items and applies the decision rule fixed in configs/experiments/e119.yaml.

Usage:
  uv run python -m sysone.experiments.e119 --dry-run --max-items 20
  uv run python -m sysone.experiments.e119 --run-id e119-v1-<date>
  uv run python -m sysone.experiments.e119 --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e102 import analyze_run as analyze_e102
from sysone.analysis.e119 import analyze_run
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, load_samples, review_work, run_parallel
from sysone.experiments.e102 import _key, review_tasks
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e119"
# Per-call cost assumed for the projection: E.105's gpt-oss-120b calls (medium effort, flex) averaged
# $0.000102-0.000158 by pool.
ASSUMED_USD_PER_CALL = 0.00013


@dataclass(frozen=True)
class E119Config:
    source_run: str
    samples: list[str]
    reviewer: str = "reviewer_120b"
    alone_samples: int = 5
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    rule: dict[str, Any] = field(default_factory=dict)
    workers: int = 8
    max_items: int | None = None   # reviewed items per pool, for a pilot
    use_cache: bool = False
    max_consecutive_errors: int = 10
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 5.0, "typesafe": 0.01})
    bootstrap: int = 2000

    @classmethod
    def load(cls, path: str | Path) -> E119Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def pilot_plan(plan: list[dict[str, Any]], max_items: int | None) -> list[dict[str, Any]]:
    """The plan, or for a pilot the first max_items reviewed rows per pool (the rest kept unreviewed)."""
    if not max_items:
        return plan
    from sysone.experiments.e102 import reviewed
    seen: dict[str, int] = {}
    out = []
    for row in plan:
        if reviewed(row):
            seen[row["sample"]] = seen.get(row["sample"], 0) + 1
            if seen[row["sample"]] > max_items:
                continue
        out.append(row)
    return out


def run_e119(config: E119Config, run: Run, client: GuardedClient, *, plan: list[dict[str, Any]],
             items: dict[tuple[str, str], Item], source_dir: Path) -> str:
    guard = client.guard
    if not config.use_cache:
        client.cache = None
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        source_run={"path": str(source_dir), "run_id": source_dir.name}, config=dataclasses.asdict(config),
        models={config.reviewer: _spec_info(client.spec)},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))})
    models = {(config.reviewer, name): Reviewer(client, TASKS[task], temperature=config.reviewer_temperature,
                                                reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                                max_output_tokens=config.reviewer_max_output_tokens)
              for name, task in {(k[0], i.task) for k, i in items.items()}}
    writer = run.writer("reviews")
    done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"])
            for r in writer.rows() if r["status"] == "ok"}
    status = "completed"
    try:
        pending = (t for t in review_tasks(config, plan, items, config.reviewer) if _key(t) not in done)
        run_parallel(pending, review_work(run, models, experiment=EXPERIMENT), writer,
                     workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
    except (BudgetError, _TooManyErrors) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={client.spec.provider: round(guard.run_spent_usd(client.spec.provider), 6)})
    return status


def n_calls(plan: list[dict[str, Any]], alone_samples: int) -> int:
    from sysone.experiments.e102 import reviewed
    return sum(alone_samples + (not r["jev_correct"]) + (not r["matched_correct"]) + bool(r["in_subsample"])
               for r in plan if reviewed(r))


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e119.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--max-items", type=int, help="reviewed items per pool, for a pilot")
    parser.add_argument("--dry-run", action="store_true", help="fake reviewer: no network, no cost")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E119Config.load(args.config)
    if args.max_items:
        config = dataclasses.replace(config, max_items=args.max_items)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    source_dir = Path(args.results_dir) / "runs" / config.source_run
    run = Run(run_id, args.results_dir)
    if not args.report_only:
        full_plan = JsonlWriter(source_dir / "plan.jsonl").rows()
        if not full_plan:
            raise SystemExit(f"{source_dir} has no plan.jsonl; E.119 needs E.102's frozen plan")
        plan = pilot_plan(full_plan, config.max_items)
        # E.102's analysis reads the plan and Jev's decisions from the run folder.
        if not (run.dir / "plan.jsonl").exists():
            (run.dir / "plan.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in plan))
        if not (run.dir / "jev_decisions.jsonl").exists():
            shutil.copy(source_dir / "jev_decisions.jsonl", run.dir / "jev_decisions.jsonl")
        samples, _ = load_samples(dataclasses.replace(config, max_items=None))
        items = {(name, i.item_id): i for name, its in samples.items() for i in its}
        registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
        if args.dry_run:
            client = GuardedClient(ModelSpec(config.reviewer, "fake", f"fake-{config.reviewer}", Price(0.0, 0.0)),
                                   FakeChatProvider(), registry.guard, cache=None)
        else:
            try:
                client = registry.client(config.reviewer)
            except (ModelNotConfigured, RuntimeError) as e:
                raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None
        calls = n_calls(plan, config.alone_samples)
        projected = round(calls * ASSUMED_USD_PER_CALL, 2)
        project = registry.project_status()
        run.update_manifest(project_spend_at_start=project, cost_projection={"calls": calls, "expected_usd": projected})
        print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
        print(f"{calls} calls; projected ${projected:.2f} (run cap ${config.run_provider_caps_usd.get('bedrock', 0):.2f}, enforced per call)")
        status = run_e119(config, run, client, plan=plan, items=items, source_dir=source_dir)
        print(f"run {run_id}: {status}")
    analyze_e102(run.dir)
    analyze_run(run.dir, source_dir, config)
    print(f"results: {run.dir}\n{(run.dir / 'comparison.md').read_text()}")


if __name__ == "__main__":
    main()
