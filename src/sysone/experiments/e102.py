"""E.102: powered test of deference beyond shared difficulty, with a reference model (research_docs/13).

Jev screens large pools; reviews then go where they can tell deference apart:
  jev        Jev decides every item of every sample, jev_samples times (as in E.101).
  review     the plan (saved once to plan.jsonl, refused if a resume would change it), then the reviewer:
             - Jev-wrong items: alone x alone_samples, and a review of Jev's label (jev_label);
             - matched-wrong items (random errors with Jev's confusion matrix, drawn on the whole
               pool, as in E.101): alone x alone_samples, and a review of that label (matched_label);
             - a random subsample of subsample_size items: alone x alone_samples, and a review of the
               correct label (correct_label). On a binary task a right proposal is always the gold
               label, whether Jev's or random, so this one review gives accept-when-right for both arms.
  reference  each reference model decides the subsample alone, reference_samples times. A reference
             never reviews; it is the third model of the error-overlap comparison.

Usage:
  uv run python -m sysone.experiments.e102 --dry-run       # fake models, no network, no cost
  uv run python -m sysone.experiments.e102 --run-id <id>   # live; set steps: [jev] to stop after the screen

Re-running with the same --run-id resumes, as in E.101.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e102 import analyze_run
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.proposals import JEV_LABEL, MATCHED_LABEL, Proposal, build_proposals, combine_jev
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ModelVersionChanged, VersionLock
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.env import load_env
from sysone.experiments.e101 import (
    ALONE,
    _JevIncomplete,
    _spec_info,
    _TooManyErrors,
    jev_step,
    load_samples,
    review_work,
    run_parallel,
)
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import Run, git_info, now_iso

EXPERIMENT = "e102"
STEPS = ("jev", "review", "reference")
CORRECT_LABEL = "correct_label"


@dataclass(frozen=True)
class E102Config:
    samples: list[str]
    jev: str = "jev"
    reviewer: str = "reviewer"
    # Models that only decide the subsample alone (non-reasoning on Bedrock: no reasoning budget).
    references: list[str] = field(default_factory=lambda: ["reviewer_qwen"])
    jev_samples: int = 3
    alone_samples: int = 5
    reference_samples: int = 3
    subsample_size: int = 400
    seed: int = 102
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    reference_max_output_tokens: int = 256
    workers: int = 8
    max_items: int | None = None
    expected_model_version: str | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 8.0, "typesafe": 1.5})

    @classmethod
    def load(cls, path: str | Path) -> E102Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def subsample(item_ids: list[str], n: int, *, seed: int, sample: str) -> set[str]:
    """A random subset of n item IDs, independent of every other draw in the run."""
    digest = hashlib.sha256(f"{seed}:{sample}:subsample".encode()).digest()
    ids = sorted(item_ids)
    return set(ids if len(ids) <= n else random.Random(int.from_bytes(digest[:8], "big")).sample(ids, n))


def plan_step(config: E102Config, run: Run, samples: dict[str, list[Item]]) -> list[dict[str, Any]]:
    """One plan row per item Jev answered: its Jev and matched proposals and whether it is in
    the subsample. Saved once; a resume that would change it is refused."""
    parsed: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    answered: dict[tuple[str, str], int] = defaultdict(int)
    for r in run.writer("jev_decisions").rows():
        if r["status"] == "ok":
            answered[(r["sample"], r["item_id"])] += 1
            if r["parse_ok"]:
                parsed[(r["sample"], r["item_id"])].append(r["probabilities"])
    missing = sum(answered[(name, i.item_id)] < config.jev_samples for name, items in samples.items() for i in items)
    if missing:
        raise _JevIncomplete(f"{missing} items lack some of Jev's {config.jev_samples} answers; resume with "
                             "steps [jev] to retry the failed requests before reviewing")
    plan: list[dict[str, Any]] = []
    for name, items in samples.items():
        labels = TASKS[items[0].task].labels
        jev = {i.item_id: d for i in items if (d := combine_jev(parsed[(name, i.item_id)], labels)) is not None}
        props = {(p.item_id, p.condition): p for p in build_proposals(items, jev, labels, seed=config.seed, flip_ps=[],
                                                                      dataset=name)}
        sub = subsample(list(jev), config.subsample_size, seed=config.seed, sample=name)
        for i in sorted(items, key=lambda i: i.item_id):
            if i.item_id not in jev:
                continue
            j, m = props[(i.item_id, JEV_LABEL)], props[(i.item_id, MATCHED_LABEL)]
            plan.append({"sample": name, "item_id": i.item_id, "label": i.label, "strata": i.strata,
                         "jev_label": j.label, "jev_correct": j.correct, "jev_confidence": jev[i.item_id].confidence,
                         "jev_split": jev[i.item_id].split, "matched_label": m.label, "matched_correct": m.correct,
                         "in_subsample": i.item_id in sub})
    writer = run.writer("plan")
    saved = writer.rows()
    if saved and saved != json.loads(json.dumps(plan)):
        raise ValueError(f"{writer.path} differs from the plan Jev's rows give now; start a new run_id")
    if not saved:
        for row in plan:
            writer.append(row)
    return plan


def reviewed(row: dict[str, Any]) -> bool:
    """Whether the reviewer sees this item at all."""
    return not row["jev_correct"] or not row["matched_correct"] or row["in_subsample"]


def _round_robin(per_sample: dict[str, list[Any]]):
    """Interleave the samples, so a run that stops early has covered every sample evenly."""
    iters = [iter(v) for v in per_sample.values()]
    while iters:
        for it in list(iters):
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)


def review_tasks(config: E102Config, plan: list[dict[str, Any]], items: dict[tuple[str, str], Item], alias: str):
    by_sample: dict[str, list] = defaultdict(list)
    for row in plan:
        if not reviewed(row):
            continue
        item = items[(row["sample"], row["item_id"])]
        tasks = [(alias, row["sample"], item, None, k) for k in range(config.alone_samples)]
        if not row["jev_correct"]:
            tasks.append((alias, row["sample"], item, Proposal(item.item_id, JEV_LABEL, row["jev_label"], False, None), 0))
        if not row["matched_correct"]:
            tasks.append((alias, row["sample"], item,
                          Proposal(item.item_id, MATCHED_LABEL, row["matched_label"], False, None), 0))
        if row["in_subsample"]:
            tasks.append((alias, row["sample"], item, Proposal(item.item_id, CORRECT_LABEL, item.label, True, None), 0))
        by_sample[row["sample"]].append(tasks)
    for tasks in _round_robin(by_sample):
        yield from tasks


def reference_tasks(config: E102Config, plan: list[dict[str, Any]], items: dict[tuple[str, str], Item], aliases: list[str]):
    by_sample: dict[str, list] = defaultdict(list)
    for row in plan:
        if row["in_subsample"]:
            item = items[(row["sample"], row["item_id"])]
            by_sample[row["sample"]].append([(a, row["sample"], item, None, k) for a in aliases
                                             for k in range(config.reference_samples)])
    for tasks in _round_robin(by_sample):
        yield from tasks


def _key(t) -> tuple:
    alias, name, item, p, k = t
    return alias, name, item.item_id, p.condition if p else ALONE, k


def run_e102(config: E102Config, run: Run, jev_client: GuardedClient, reviewer: GuardedClient,
             references: dict[str, GuardedClient], *, samples: dict[str, list[Item]], manifests: dict[str, dict]) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not config.use_cache:
        for c in (jev_client, reviewer, *references.values()):
            c.cache = None
    locked = run.manifest().get("model_version")
    if locked and config.expected_model_version and locked != config.expected_model_version:
        raise ValueError(f"run {run.run_id} is locked to {locked}, but the config expects "
                         f"{config.expected_model_version}; start a new run_id to use another version")
    jev_client.version_lock = VersionLock(locked or config.expected_model_version,
                                          on_lock=lambda v: run.update_manifest(model_version=v))
    guard = jev_client.guard
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        resumed_at=now_iso() if (run.dir / "jev_decisions.jsonl").exists() else None,
        samples={name: {k: m[k] for k in ("name", "sha256", "n_items", "source", "task")} for name, m in manifests.items()},
        config=dataclasses.asdict(config),
        models={"jev": _spec_info(jev_client.spec), config.reviewer: _spec_info(reviewer.spec),
                **{a: _spec_info(c.spec) for a, c in references.items()}},
        model_version=jev_client.version_lock.expected,
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    status = "completed"
    try:
        if "jev" in config.steps:
            jev_step(config, run, jev_client, samples, experiment=EXPERIMENT)
        if {"review", "reference"} & set(config.steps):
            plan = plan_step(config, run, samples)
            writer = run.writer("reviews")
            done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"])
                    for r in writer.rows() if r["status"] == "ok"}
            if "review" in config.steps:
                models = {(config.reviewer, name): Reviewer(reviewer, TASKS[its[0].task],
                                                            temperature=config.reviewer_temperature,
                                                            reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                                            max_output_tokens=config.reviewer_max_output_tokens)
                          for name, its in samples.items()}
                pending = (t for t in review_tasks(config, plan, items, config.reviewer) if _key(t) not in done)
                run_parallel(pending, review_work(run, models, experiment=EXPERIMENT), writer,
                             workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
            if "reference" in config.steps and references:
                models = {(a, name): Reviewer(c, TASKS[its[0].task], max_output_tokens=config.reference_max_output_tokens)
                          for a, c in references.items() for name, its in samples.items()}
                pending = (t for t in reference_tasks(config, plan, items, list(references)) if _key(t) not in done)
                run_parallel(pending, review_work(run, models, experiment=EXPERIMENT, role="reference"), writer,
                             workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
    except (BudgetError, _TooManyErrors, _JevIncomplete, ModelVersionChanged) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        providers = sorted({c.spec.provider for c in (jev_client, reviewer, *references.values())})
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6) for p in providers})
    return status


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e102.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--steps", nargs="+", choices=STEPS, help="override the config's steps")
    parser.add_argument("--dry-run", action="store_true", help="fake models: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E102Config.load(args.config)
    if args.steps:
        config = dataclasses.replace(config, steps=args.steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, manifests = load_samples(config)

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    if args.dry_run:
        fake = lambda a, p: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), p, registry.guard, cache=None)
        jev_client = fake("jev", FakeDecisionProvider())
        reviewer = fake(config.reviewer, FakeChatProvider())
        references = {a: fake(a, FakeChatProvider()) for a in config.references}
    else:
        try:
            jev_client = registry.client(config.jev)
            reviewer = registry.client(config.reviewer)
            references = {a: registry.client(a) for a in config.references}
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e102(config, run, jev_client, reviewer, references, samples=samples, manifests=manifests)
    analyze_run(run.dir)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
