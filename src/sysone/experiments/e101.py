"""E.101: go/no-go pilot for Part 1 of the cascade direction (research_docs/13).

Does a reviewer accept Jev's wrong decisions more often than equally many random wrong
decisions? Three steps, each resumable:
  jev       Jev decides every item of every sample, jev_samples times (rotated option order).
  proposals Jev's decisions and the random error processes, built from the jev step's rows
            (sysone.cascade.proposals) and saved once; a resume that would change them is refused.
  review    each reviewer decides every item alone (alone_samples times) and reviews every
            proposal condition once.

Usage:
  uv run python -m sysone.experiments.e101 --dry-run        # fake models, no network, no cost
  uv run python -m sysone.experiments.e101 --run-id <id>    # live (TypeSafe and Bedrock)

Re-running with the same --run-id resumes: rows already saved are skipped, failed requests
are retried, and the run's earlier spend still counts toward its caps. The run's per-provider
caps (run_provider_caps_usd: E.101's $8 Bedrock, $1 TypeSafe) apply on top of the $25 run cap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e101 import analyze_run
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item, load_items
from sysone.cascade.proposals import (
    ALONE,
    JEV_CONF,
    JEV_LABEL,
    MATCHED_CONF,
    MATCHED_LABEL,
    Proposal,
    build_proposals,
    combine_jev,
    flip_condition,
)
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ModelVersionChanged, ProviderError, VersionLock
from sysone.clients.fake import FakeChatProvider, FakeDecisionProvider
from sysone.decision.jev import JevDecisionModel
from sysone.env import load_env
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e101"
STEPS = ("jev", "review")


@dataclass(frozen=True)
class E101Config:
    # Frozen samples (benchmarks/cascade/<name>), one per dataset.
    samples: list[str]
    jev: str = "jev"
    reviewers: list[str] = field(default_factory=lambda: ["reviewer"])
    jev_samples: int = 3
    alone_samples: int = 3
    conditions: list[str] = field(default_factory=lambda: [JEV_LABEL, JEV_CONF, MATCHED_LABEL, MATCHED_CONF])
    flip_ps: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.35])
    seed: int = 101
    reviewer_temperature: float | None = None
    # Reasoning tokens allowed on top of the visible answer (gpt-oss counts both in maxTokens).
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    # Requests in flight at once. The cost guard reserves each one's worst case.
    workers: int = 8
    # Only the first n items of each sample (by item ID), for a partial run; None for all.
    max_items: int | None = None
    expected_model_version: str | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 8.0, "typesafe": 1.0})

    @classmethod
    def load(cls, path: str | Path) -> E101Config:
        return cls(**yaml.safe_load(Path(path).read_text()))

    @property
    def review_conditions(self) -> list[str]:
        return [*self.conditions, *(flip_condition(p) for p in self.flip_ps)]


class _TooManyErrors(RuntimeError):
    pass


class _JevIncomplete(RuntimeError):
    pass


# Errors that halt the whole run rather than one request.
_FATAL = (BudgetError, ModelVersionChanged)


def run_parallel(tasks: Iterable[Any], work: Callable[[Any], dict[str, Any]], writer: JsonlWriter, *, workers: int,
                 max_consecutive_errors: int) -> None:
    """Run `work` on each task with up to `workers` in flight, appending each row as it
    completes (from this thread only). A row with status "error" counts toward
    max_consecutive_errors; a fatal error stops new work, lets in-flight requests finish
    and is re-raised."""
    tasks = iter(tasks)
    consecutive = 0
    fatal: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future] = set()
        while True:
            while fatal is None and len(pending) < workers and (task := next(tasks, None)) is not None:
                pending.add(pool.submit(work, task))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    row = future.result()
                except _FATAL as e:
                    fatal = fatal or e
                    continue
                writer.append(row)
                if row["status"] == "error":
                    consecutive += 1
                    if consecutive >= max_consecutive_errors and fatal is None:
                        fatal = _TooManyErrors(f"{consecutive} failed requests in a row; last: {row['error']}")
                else:
                    consecutive = 0
    if fatal is not None:
        raise fatal


def _usage(resp) -> dict[str, Any]:
    return {"model_used": resp.model_used, "cost_usd": resp.cost_usd, "cached": resp.cached,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
            "reasoning_tokens": resp.usage.reasoning_tokens, "finish_reason": resp.finish_reason,
            "service_tier": resp.service_tier, "latency_s": resp.latency_s, "wall_s": resp.wall_s}


def jev_step(config: E101Config, run: Run, client: GuardedClient, samples: dict[str, list[Item]]) -> None:
    writer = run.writer("jev_decisions")
    done = {(r["sample"], r["item_id"], r["sample_index"]) for r in writer.rows() if r["status"] == "ok"}
    models = {name: JevDecisionModel(client, task=TASKS[items[0].task].jev) for name, items in samples.items()}
    tasks = [(name, item, k) for name, items in samples.items() for item in items for k in range(config.jev_samples)
             if (name, item.item_id, k) not in done]

    def work(t: tuple[str, Item, int]) -> dict[str, Any]:
        name, item, k = t
        base = {"run_id": run.run_id, "sample": name, "item_id": item.item_id, "task": item.task,
                "sample_index": k, "label": item.label, "strata": item.strata}
        tags = {"experiment": EXPERIMENT, "step": "jev", "sample": name, "item_id": item.item_id,
                "sample_index": str(k)}
        try:
            result = models[name].decide(TASKS[item.task].policy, item.text, sample_index=k, tags=tags)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        d = result.decision
        return {**base, "status": "ok", "decision": d.label, "probabilities": d.probabilities,
                "confidence": d.confidence, "parse_ok": d.parse_ok, "parse_error": d.parse_error,
                "raw_answer": None if d.parse_ok else d.raw_answer, "option_order": list(result.option_order),
                "request_key": result.request_key, **_usage(result.response), "ts": now_iso()}

    run_parallel(tasks, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)


def proposals_step(config: E101Config, run: Run, samples: dict[str, list[Item]]) -> dict[str, dict[str, list[Proposal]]]:
    """Each sample's proposals by item, built from the jev step's rows. The first build is
    saved; a later build that differs (Jev's rows changed after reviews could start) is refused."""
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
    built: list[dict[str, Any]] = []
    out: dict[str, dict[str, list[Proposal]]] = {}
    for name, items in samples.items():
        labels = TASKS[items[0].task].labels
        jev = {i.item_id: d for i in items if (d := combine_jev(parsed[(name, i.item_id)], labels)) is not None}
        proposals = build_proposals(items, jev, labels, seed=config.seed, flip_ps=config.flip_ps, dataset=name)
        by_item: dict[str, list[Proposal]] = defaultdict(list)
        for p in proposals:
            by_item[p.item_id].append(p)
            built.append({"sample": name, **p.to_dict(), **({"jev_split": jev[p.item_id].split,
                          "jev_n_samples": jev[p.item_id].n_samples} if p.condition == JEV_LABEL else {})})
        out[name] = dict(by_item)
    writer = run.writer("proposals")
    saved = writer.rows()
    if saved and saved != json.loads(json.dumps(built)):
        raise ValueError(f"{writer.path} differs from the proposals Jev's rows give now; start a new run_id")
    if not saved:
        for row in built:
            writer.append(row)
    return out


def review_step(config: E101Config, run: Run, reviewers: dict[str, GuardedClient], samples: dict[str, list[Item]],
                proposals: dict[str, dict[str, list[Proposal]]]) -> None:
    writer = run.writer("reviews")
    done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"])
            for r in writer.rows() if r["status"] == "ok"}
    models = {(alias, name): Reviewer(client, TASKS[items[0].task], temperature=config.reviewer_temperature,
                                      reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                      max_output_tokens=config.reviewer_max_output_tokens)
              for alias, client in reviewers.items() for name, items in samples.items()}
    wanted = set(config.review_conditions)

    def tasks():
        # Item by item, so a run that stops early has every condition for the items it reached.
        for name, items in samples.items():
            for item in items:
                if item.item_id not in proposals[name]:
                    continue  # Jev never answered it, so no condition covers it
                for alias in reviewers:
                    for k in range(config.alone_samples):
                        yield alias, name, item, None, k
                    for p in proposals[name][item.item_id]:
                        if p.condition in wanted:
                            yield alias, name, item, p, 0

    def work(t: tuple[str, str, Item, Proposal | None, int]) -> dict[str, Any]:
        alias, name, item, p, k = t
        condition = p.condition if p else ALONE
        base = {"run_id": run.run_id, "reviewer": alias, "sample": name, "item_id": item.item_id, "task": item.task,
                "condition": condition, "sample_index": k, "label": item.label, "strata": item.strata,
                "proposed_label": p.label if p else None, "proposed_correct": p.correct if p else None,
                "confidence_shown": p.confidence if p else None}
        tags = {"experiment": EXPERIMENT, "step": "review", "reviewer": alias, "sample": name,
                "item_id": item.item_id, "condition": condition, "sample_index": str(k)}
        try:
            r = models[(alias, name)].decide(item.text, p.label if p else None, p.confidence if p else None,
                                             sample_index=k, tags=tags)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict,
                "consistent": r.consistent, "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                "raw_text": None if r.parse_ok else r.response.text[:2000], "request_key": r.request_key,
                **_usage(r.response), "ts": now_iso()}

    pending = (t for t in tasks() if (t[0], t[1], t[2].item_id, t[3].condition if t[3] else ALONE, t[4]) not in done)
    run_parallel(pending, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)


def _spec_info(spec: ModelSpec) -> dict[str, Any]:
    return {"alias": spec.alias, "provider": spec.provider, "model": spec.model, "service_tier": spec.service_tier,
            "reasoning_effort": spec.reasoning_effort,
            "price_per_m": dataclasses.asdict(spec.price) if spec.price else None,
            "tier_price_per_m": dataclasses.asdict(spec.tier_price) if spec.tier_price else None}


def run_e101(config: E101Config, run: Run, jev_client: GuardedClient, reviewers: dict[str, GuardedClient], *,
             samples: dict[str, list[Item]], manifests: dict[str, dict]) -> str:
    """Run (or resume) E.101 and return the final run status."""
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not config.use_cache:
        jev_client.cache = None
        for c in reviewers.values():
            c.cache = None
    locked = run.manifest().get("model_version")
    if locked and config.expected_model_version and locked != config.expected_model_version:
        raise ValueError(f"run {run.run_id} is locked to {locked}, but the config expects "
                         f"{config.expected_model_version}; start a new run_id to use another version")
    jev_client.version_lock = VersionLock(locked or config.expected_model_version,
                                          on_lock=lambda v: run.update_manifest(model_version=v))
    guard = jev_client.guard
    run.update_manifest(
        experiment=EXPERIMENT,
        status="running",
        started_at=run.manifest().get("started_at", now_iso()),
        resumed_at=now_iso() if (run.dir / "jev_decisions.jsonl").exists() else None,
        samples={name: {k: m[k] for k in ("name", "sha256", "n_items", "source", "task")} for name, m in manifests.items()},
        config=dataclasses.asdict(config),
        models={"jev": _spec_info(jev_client.spec), **{a: _spec_info(c.spec) for a, c in reviewers.items()}},
        model_version=jev_client.version_lock.expected,
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(),
        config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    status = "completed"
    try:
        if "jev" in config.steps:
            jev_step(config, run, jev_client, samples)
        if "review" in config.steps:
            proposals = proposals_step(config, run, samples)
            review_step(config, run, reviewers, samples, proposals)
    except (BudgetError, _TooManyErrors, _JevIncomplete, ModelVersionChanged) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6)
                                               for p in sorted({jev_client.spec.provider,
                                                                *(c.spec.provider for c in reviewers.values())})})
    return status


def load_samples(config: E101Config) -> tuple[dict[str, list[Item]], dict[str, dict]]:
    samples, manifests = {}, {}
    for name in config.samples:
        items, manifest = load_items(name)
        items = sorted(items, key=lambda i: i.item_id)
        samples[name] = items[: config.max_items] if config.max_items else items
        manifests[name] = manifest
    return samples, manifests


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e101.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="fake Jev and reviewers: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E101Config.load(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, manifests = load_samples(config)

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    if args.dry_run:
        jev_client = GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(0.0, 0.0)), FakeDecisionProvider(),
                                   registry.guard, cache=None)
        reviewers = {a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(),
                                      registry.guard, cache=None) for a in config.reviewers}
    else:
        try:
            jev_client = registry.client(config.jev)
            reviewers = {a: registry.client(a) for a in config.reviewers}
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e101(config, run, jev_client, reviewers, samples=samples, manifests=manifests)
    analyze_run(run.dir)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
