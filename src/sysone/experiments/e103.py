"""E.103: cheap alternatives to post-training, on E.102's frozen plan (research_docs/13).

No new Jev calls. Every review reuses an E.102 run's plan: the same items, Jev's wrong labels, the
matched random wrong labels and the random subsample's correct labels, so each protocol is compared
with the source run's standard review on identical items. Difficulty is the source run's: the
reviewer's 5 alone samples.

The source run is E.102's for gpt-oss-20b (configs/experiments/e103.yaml). For Part 2's Qwen
reviewer (configs/experiments/e103_qwen.yaml) it is the E.106 run once base_local has run there:
its standard reviews, alone record and blind-first answers are base_local's rows, and plan_run
points at E.102's run for the plan. Its result is criterion 2's bar (research_docs/14).

  reviews   each protocol reviews Jev-wrong (jev_label), matched-wrong (matched_label) and
            subsample (correct_label) items once:
              blind_first         the reviewer's own E.102 alone answer (sample 0) comes first,
                                  then Jev's decision is shown and the review asked
              warning             "this automated model is often wrong"
              calibrated_warning  the automated model's measured error rate on the pool
              consider_opposite   argue for the other label before deciding
  upstream  other upstream models decide every item alone once, for Part 2's held-out tests
            and the error-overlap comparison on whole pools.

Usage:
  uv run python -m sysone.experiments.e103 --dry-run --source-run <e102 dry-run id>
  uv run python -m sysone.experiments.e103 --run-id <id>        # source_run from the config
  uv run python -m sysone.experiments.e103 --config configs/experiments/e103_qwen.yaml --run-id <id>
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e103 import analyze_run
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.proposals import JEV_LABEL, MATCHED_LABEL
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import ALONE, _spec_info, _TooManyErrors, _usage, load_samples, review_work, run_parallel
from sysone.experiments.e102 import CORRECT_LABEL, _round_robin, reviewed
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e103"
STEPS = ("reviews", "upstream")
BLIND_FIRST, WARNING, CALIBRATED, OPPOSITE = "blind_first", "warning", "calibrated_warning", "consider_opposite"
PROTOCOLS = (BLIND_FIRST, WARNING, CALIBRATED, OPPOSITE)
WARNING_NOTE = "Note: this automated model is often wrong."
CALIBRATED_NOTE = "Note: on content like this, this automated model is wrong about {rate} of the time."
OPPOSITE_PREFIX = ("Before you decide, make the strongest case you can for the other label. Then judge which label "
                   "the policy really supports.")


@dataclass(frozen=True)
class E103Config:
    # The run whose reviews are the standard prompt (and the difficulty record and blind-first's
    # committed answers): E.102's, or E.106's for a Part 2 reviewer.
    source_run: str
    samples: list[str]
    # The E.102 run whose plan.jsonl is reviewed, if source_run isn't an E.102 run; None: source_run.
    plan_run: str | None = None
    reviewer: str = "reviewer"
    protocols: list[str] = field(default_factory=lambda: list(PROTOCOLS))
    upstreams: list[str] = field(default_factory=lambda: ["explainer_open", "reviewer_qwen", "decision_alt"])
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    upstream_max_output_tokens: int = 256
    workers: int = 8
    max_items: int | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 6.0, "typesafe": 0.01})

    @classmethod
    def load(cls, path: str | Path) -> E103Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def protocol_kwargs(protocol: str, *, jev_error_rate: float, own_answer: str | None) -> dict[str, Any]:
    """The Reviewer.decide options for one protocol. The calibrated note states Jev's error rate on
    the pool; the matched arm has the same rate by construction, so both arms read alike."""
    if protocol == BLIND_FIRST:
        return {"own_answer": own_answer}
    if protocol == WARNING:
        return {"note": WARNING_NOTE}
    if protocol == CALIBRATED:
        return {"note": CALIBRATED_NOTE.format(rate=f"{jev_error_rate:.0%}")}
    if protocol == OPPOSITE:
        return {"task_prefix": OPPOSITE_PREFIX}
    raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")


def first_alone_answers(source_reviews: list[dict[str, Any]], reviewer: str) -> dict[tuple[str, str], str]:
    """Each item's earliest parsed alone answer from the source run: blind-first's committed answer."""
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for r in source_reviews:
        if (r["status"] == "ok" and r["parse_ok"] and r["condition"] == ALONE and r["reviewer"] == reviewer
                and r.get("role", "reviewer") == "reviewer"):
            key = (r["sample"], r["item_id"])
            if key not in best or r["sample_index"] < best[key][0]:
                best[key] = (r["sample_index"], r["final_label"])
    return {k: v for k, (_, v) in best.items()}


def review_tasks(config: E103Config, plan: list[dict], items: dict[tuple[str, str], Item],
                 own: dict[tuple[str, str], str], error_rate: dict[str, float]):
    """(sample, item, condition, proposed label, correct, protocol, kwargs), item by item, samples interleaved."""
    by_sample: dict[str, list] = defaultdict(list)
    for row in plan:
        if not reviewed(row):
            continue
        key = (row["sample"], row["item_id"])
        item = items.get(key)
        if item is None:
            continue
        proposals = []
        if not row["jev_correct"]:
            proposals.append((JEV_LABEL, row["jev_label"], False))
        if not row["matched_correct"]:
            proposals.append((MATCHED_LABEL, row["matched_label"], False))
        if row["in_subsample"]:
            proposals.append((CORRECT_LABEL, row["label"], True))
        tasks = []
        for protocol in config.protocols:
            if protocol == BLIND_FIRST and key not in own:
                continue  # no committed answer to start from
            kw = protocol_kwargs(protocol, jev_error_rate=error_rate[row["sample"]], own_answer=own.get(key))
            tasks += [(row["sample"], item, cond, label, correct, protocol, kw) for cond, label, correct in proposals]
        by_sample[row["sample"]].append(tasks)
    for tasks in _round_robin(by_sample):
        yield from tasks


def run_e103(config: E103Config, run: Run, reviewer: GuardedClient, upstreams: dict[str, GuardedClient], *,
             samples: dict[str, list[Item]], source_dir: Path, plan_dir: Path | None = None) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    for p in config.protocols:
        protocol_kwargs(p, jev_error_rate=0.0, own_answer="x")  # fail fast on an unknown protocol
    if not config.use_cache:
        for c in (reviewer, *upstreams.values()):
            c.cache = None
    plan_dir = plan_dir or source_dir
    plan = JsonlWriter(plan_dir / "plan.jsonl").rows()
    if not plan:
        raise ValueError(f"{plan_dir} has no plan.jsonl; E.103 needs a completed E.102 run (plan_run)")
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    source_reviews = JsonlWriter(source_dir / "reviews.jsonl").rows()
    if not any(r.get("reviewer") == config.reviewer for r in source_reviews):
        raise ValueError(f"{source_dir} has no reviews by {config.reviewer!r}: the standard prompt to compare "
                         "with is missing (for a Part 2 reviewer, run E.106 with it first)")
    own = first_alone_answers(source_reviews, config.reviewer)
    error_rate = {s: sum(not p["jev_correct"] for p in plan if p["sample"] == s) / max(1, sum(p["sample"] == s for p in plan))
                  for s in samples}
    guard = reviewer.guard
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        source_run={"path": str(source_dir), "run_id": source_dir.name, **{
            k: v for k, v in Run(source_dir.name, source_dir.parent.parent).manifest().items() if k in ("git", "status")}},
        plan_run=plan_dir.name,
        jev_error_rate_by_sample={k: round(v, 4) for k, v in error_rate.items()},
        config=dataclasses.asdict(config),
        models={config.reviewer: _spec_info(reviewer.spec), **{a: _spec_info(c.spec) for a, c in upstreams.items()}},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    models = {name: Reviewer(reviewer, TASKS[its[0].task], temperature=config.reviewer_temperature,
                             reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                             max_output_tokens=config.reviewer_max_output_tokens) for name, its in samples.items()}
    status = "completed"
    try:
        if "reviews" in config.steps:
            writer = run.writer("reviews")
            done = {(r["sample"], r["item_id"], r["condition"], r["protocol"]) for r in writer.rows() if r["status"] == "ok"}

            def work(t) -> dict[str, Any]:
                name, item, cond, label, correct, protocol, kw = t
                base = {"run_id": run.run_id, "reviewer": config.reviewer, "role": "reviewer", "sample": name,
                        "item_id": item.item_id, "task": item.task, "condition": cond, "protocol": protocol,
                        "sample_index": 0, "label": item.label, "strata": item.strata, "proposed_label": label,
                        "proposed_correct": correct, "own_answer": kw.get("own_answer")}
                tags = {"experiment": EXPERIMENT, "step": "review", "sample": name, "item_id": item.item_id,
                        "condition": cond, "protocol": protocol}
                try:
                    r = models[name].decide(item.text, label, **kw, tags=tags)
                except ProviderError as e:
                    return {**base, "status": "error", "error": str(e), "ts": now_iso()}
                return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict,
                        "consistent": r.consistent, "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                        "raw_text": None if r.parse_ok else r.response.text[:2000], "request_key": r.request_key,
                        **_usage(r.response), "ts": now_iso()}

            pending = (t for t in review_tasks(config, plan, items, own, error_rate)
                       if (t[0], t[1].item_id, t[2], t[5]) not in done)
            run_parallel(pending, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
        if "upstream" in config.steps and upstreams:
            writer = run.writer("upstream")
            done = {(r["reviewer"], r["sample"], r["item_id"]) for r in writer.rows() if r["status"] == "ok"}
            up_models = {(a, name): Reviewer(c, TASKS[its[0].task], max_output_tokens=config.upstream_max_output_tokens)
                         for a, c in upstreams.items() for name, its in samples.items()}
            by_sample: dict[str, list] = defaultdict(list)
            for row in plan:
                key = (row["sample"], row["item_id"])
                if key in items:
                    by_sample[row["sample"]].append([(a, row["sample"], items[key], None, 0) for a in upstreams])
            pending = (t for tasks in _round_robin(by_sample) for t in tasks
                       if (t[0], t[1], t[2].item_id) not in done)
            run_parallel(pending, review_work(run, up_models, experiment=EXPERIMENT, role="upstream"), writer,
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
        providers = sorted({c.spec.provider for c in (reviewer, *upstreams.values())})
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6) for p in providers})
    return status


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e103.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--source-run", help="override the config's source run (E.102's, or E.106's for base_local)")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dry-run", action="store_true", help="fake models: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E103Config.load(args.config)
    if args.source_run:
        config = dataclasses.replace(config, source_run=args.source_run)
    if args.steps:
        config = dataclasses.replace(config, steps=args.steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, _ = load_samples(config)
    source_dir = Path(args.results_dir) / "runs" / config.source_run
    plan_dir = Path(args.results_dir) / "runs" / (config.plan_run or config.source_run)

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    if args.dry_run:
        fake = lambda a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(),
                                       registry.guard, cache=None)
        reviewer, upstreams = fake(config.reviewer), {a: fake(a) for a in config.upstreams}
    else:
        try:
            reviewer = registry.client(config.reviewer)
            upstreams = {a: registry.client(a) for a in config.upstreams}
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e103(config, run, reviewer, upstreams, samples=samples, source_dir=source_dir, plan_dir=plan_dir)
    analyze_run(run.dir, source_dir, plan_dir, reviewer=config.reviewer)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
