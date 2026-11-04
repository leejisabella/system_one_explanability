"""E.106–E.108 evaluation: base and post-trained reviewers on E.102's test plan (research_docs/14).

Every reviewer, the untrained base included, runs on the same self-hosted stack (vLLM), so a
difference between them is training, not serving. Each one, on E.102's frozen test pools:
  alone     decides alone every item it will review: the difficulty reviewer (the base, by default)
            alone_samples times, which is the difficulty scale for all of them; the others once.
  review    reviews E.102's proposals: Jev's wrong labels (jev_label), the matched random wrong
            labels (matched_label) and the subsample's correct labels (correct_label).
  upstream  reviews the held-out upstream models' wrong labels (E.103's Llama, Qwen and Nova
            decisions; no model was trained on their mistakes): a seeded random set of
            upstream_wrong_per_sample wrong decisions per model and pool (condition
            upstream_wrong:<model>). Their right labels are the gold label, which correct_label
            already reviews.
On a binary task every wrong proposal on an item is the same label, so the prompt is identical
whoever proposed it; only which items are wrong differs.

Serve one model at a time and pass --reviewers; a run resumes per reviewer, so one run_id holds them all.

Usage:
  uv run python -m sysone.experiments.e106 --dry-run --source-run <e102 dry-run> --upstream-run <e103 dry-run>
  uv run python -m sysone.experiments.e106 --run-id e106-v1-<date> --reviewers base_local
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.proposals import ALONE, JEV_LABEL, MATCHED_LABEL, Proposal
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, load_samples, review_work, run_parallel
from sysone.experiments.e102 import CORRECT_LABEL, _round_robin, reviewed
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e106"
STEPS = ("alone", "review", "upstream")
UPSTREAM_WRONG = "upstream_wrong"


@dataclass(frozen=True)
class E106Config:
    source_run: str      # the E.102 run: its plan is the test set
    upstream_run: str    # the E.103 run: its upstream.jsonl holds the held-out upstream decisions
    samples: list[str]
    reviewers: list[str] = field(default_factory=lambda: ["base_local", "sft_review", "sft_plain"])
    difficulty_reviewer: str = "base_local"
    upstreams: list[str] = field(default_factory=lambda: ["explainer_open", "reviewer_qwen", "decision_alt"])
    upstream_wrong_per_sample: int = 150
    # The difficulty reviewer decides alone alone_samples times (the difficulty scale, as E.102's 5);
    # the others once each, which is all their alone accuracy needs.
    alone_samples: int = 5
    alone_samples_others: int = 1
    seed: int = 106
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    workers: int = 32
    max_items: int | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    # Evaluation runs on the GPU: any paid call is a mistake, so paid providers get a cent.
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 0.01, "typesafe": 0.01})
    # Pre-registered criteria (research_docs/14), read by sysone.analysis.e106.
    criteria: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> E106Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def upstream_wrong(plan: list[dict[str, Any]], upstream: list[dict[str, Any]], models: list[str], n: int,
                   seed: int) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(sample, model) -> [(item_id, wrong label)]: a seeded random set of n items per pool where the
    model's decision was wrong, drawn from the whole pool."""
    in_plan = {(p["sample"], p["item_id"]) for p in plan}
    wrong: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in upstream:
        if (r["reviewer"] in models and r["status"] == "ok" and r["parse_ok"] and r["final_label"] != r["label"]
                and (r["sample"], r["item_id"]) in in_plan):
            wrong[(r["sample"], r["reviewer"])].append((r["item_id"], r["final_label"]))
    out = {}
    for key, rows in wrong.items():
        rows = sorted(rows)
        digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}:upstream".encode()).digest()
        out[key] = rows if len(rows) <= n else sorted(random.Random(int.from_bytes(digest[:8], "big")).sample(rows, n))
    return out


def eval_tasks(config: E106Config, plan: list[dict[str, Any]], items: dict[tuple[str, str], Item],
               picked: dict[tuple[str, str], list[tuple[str, str]]], alias: str):
    """One reviewer's tasks, item by item, pools interleaved: alone samples first, then its reviews."""
    up_by_item: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for (sample, model), rows in picked.items():
        for item_id, label in rows:
            up_by_item[(sample, item_id)].append((model, label))
    by_sample: dict[str, list] = defaultdict(list)
    for row in plan:
        key = (row["sample"], row["item_id"])
        item = items.get(key)
        if item is None or not (reviewed(row) or key in up_by_item):
            continue
        tasks = []
        if "alone" in config.steps:
            n_alone = config.alone_samples if alias == config.difficulty_reviewer else config.alone_samples_others
            tasks += [(alias, row["sample"], item, None, k) for k in range(n_alone)]
        if "review" in config.steps:
            if not row["jev_correct"]:
                tasks.append((alias, row["sample"], item, Proposal(item.item_id, JEV_LABEL, row["jev_label"], False, None), 0))
            if not row["matched_correct"]:
                tasks.append((alias, row["sample"], item,
                              Proposal(item.item_id, MATCHED_LABEL, row["matched_label"], False, None), 0))
            if row["in_subsample"]:
                tasks.append((alias, row["sample"], item, Proposal(item.item_id, CORRECT_LABEL, item.label, True, None), 0))
        if "upstream" in config.steps:
            for model, label in sorted(up_by_item.get(key, [])):
                tasks.append((alias, row["sample"], item,
                              Proposal(item.item_id, f"{UPSTREAM_WRONG}:{model}", label, False, None), 0))
        by_sample[row["sample"]].append(tasks)
    for tasks in _round_robin(by_sample):
        yield from tasks


def _key(t) -> tuple:
    alias, name, item, p, k = t
    return alias, name, item.item_id, p.condition if p else ALONE, k


def run_e106(config: E106Config, run: Run, reviewers: dict[str, GuardedClient], *, samples: dict[str, list[Item]],
             source_dir: Path, upstream_dir: Path) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not config.use_cache:
        for c in reviewers.values():
            c.cache = None
    plan = JsonlWriter(source_dir / "plan.jsonl").rows()
    if not plan:
        raise ValueError(f"{source_dir} has no plan.jsonl; the evaluation needs a completed E.102 run")
    upstream = JsonlWriter(upstream_dir / "upstream.jsonl").rows()
    picked = upstream_wrong(plan, upstream, config.upstreams, config.upstream_wrong_per_sample, config.seed)
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    guard = next(iter(reviewers.values())).guard
    manifest = run.manifest()
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=manifest.get("started_at", now_iso()),
        source_run={"run_id": source_dir.name, **{k: v for k, v in Run(source_dir.name, source_dir.parent.parent)
                                                    .manifest().items() if k in ("git", "status")}},
        upstream_run=upstream_dir.name,
        upstream_wrong_counts={f"{s}/{m}": len(v) for (s, m), v in sorted(picked.items())},
        config=dataclasses.asdict(config),
        # Each reviewer's model is recorded as it is evaluated; earlier ones stay.
        models={**manifest.get("models", {}), **{a: _spec_info(c.spec) for a, c in reviewers.items()}},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    writer = run.writer("reviews")
    done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"])
            for r in writer.rows() if r["status"] == "ok"}
    status = "completed"
    try:
        for alias, client in reviewers.items():
            models = {(alias, name): Reviewer(client, TASKS[its[0].task], temperature=config.reviewer_temperature,
                                              reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                              max_output_tokens=config.reviewer_max_output_tokens)
                      for name, its in samples.items()}
            pending = (t for t in eval_tasks(config, plan, items, picked, alias) if _key(t) not in done)
            run_parallel(pending, review_work(run, models, experiment=EXPERIMENT), writer, workers=config.workers,
                         max_consecutive_errors=config.max_consecutive_errors)
    except (BudgetError, _TooManyErrors) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        providers = sorted({c.spec.provider for c in reviewers.values()})
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6) for p in providers})
    return status


def main() -> None:
    from sysone.analysis.e106 import analyze_run

    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e106.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--reviewers", nargs="+", help="evaluate only these aliases (the one being served)")
    parser.add_argument("--source-run", help="override the config's E.102 run")
    parser.add_argument("--upstream-run", help="override the config's E.103 run")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dry-run", action="store_true", help="fake reviewers: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E106Config.load(args.config)
    for name, value in (("source_run", args.source_run), ("upstream_run", args.upstream_run), ("steps", args.steps)):
        if value:
            config = dataclasses.replace(config, **{name: value})
    aliases = args.reviewers or config.reviewers
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, _ = load_samples(config)
    runs_dir = Path(args.results_dir) / "runs"

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    if args.dry_run:
        reviewers = {a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(),
                                      registry.guard, cache=None) for a in aliases}
    else:
        try:
            reviewers = {a: registry.client(a) for a in aliases}
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without a server.") from None

    run = Run(run_id, args.results_dir)
    status = run_e106(config, run, reviewers, samples=samples, source_dir=runs_dir / config.source_run,
                      upstream_dir=runs_dir / config.upstream_run)
    analyze_run(run.dir, runs_dir / config.source_run, runs_dir / config.upstream_run, config)
    print(f"run {run_id}: {status}\nresults: {run.dir}")
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
