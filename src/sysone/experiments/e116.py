"""E.116: prompt-engineering development screen (research_docs/16). Bedrock only, no Jev calls.

Does a different prompt help gpt-oss-20b apply WildGuard's policy to Jev's mistakes? Each arm
(sysone.prompteng.arms) reviews the same development items once, in the review format with Jev's
label shown, and is compared with the standard prompt on the same items (paired).

Items: E.104's mix, WildGuard training pools only (never the test pools):
  jev_wrong  every Jev mistake in the mix (over-flagged: gold unharmful; let through: gold harmful)
  jev_right  right_per_pool of Jev's right labels per pool, first by hash
Items used as worked examples (configs/prompteng/examples.yaml) are kept out of the scored set.

The selection rule is fixed in configs/experiments/e116.yaml before the run; the analysis
(sysone.analysis.e116) applies it and names at most one arm for E.117.

Usage:
  uv run python -m sysone.experiments.e116 --dry-run --arms standard grounded
  uv run python -m sysone.experiments.e116 --run-id e116-v1-<date>
  uv run python -m sysone.experiments.e116 --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis.e116 import analyze_run
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, _usage, load_samples, run_parallel
from sysone.posttrain.mix import JEV_RIGHT, JEV_WRONG, REVIEW
from sysone.prompteng.arms import (ARMS, STANDARD, WILDGUARD_TASKS, ArmTexts, arm_kwargs, check_test_overlap,
                                   example_keys, load_texts)
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e116"
OVER_FLAGGED, LET_THROUGH, RIGHT = "over_flagged", "let_through", "right"
# Tokens per call for the cost projection: E.103's WildGuard reviews averaged about 740 in and 310 out;
# the clarify and examples arms add up to about 1,500 input tokens.
ASSUMED_TOKENS = (1500, 400)


@dataclass(frozen=True)
class E116Config:
    source_run: str                  # E.104: the mix (Jev's labels on the training pools)
    samples: list[str]               # WildGuard training pools only
    # E.117's test pools: hand-written examples are checked against them for shared wording.
    test_samples: list[str] = field(default_factory=lambda: ["wildguard_prompt_full_v1", "wildguard_response_full_v1"])
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    right_per_pool: int = 300
    reviewer: str = "reviewer"
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    seed: str = "e116-v1"
    selection: dict[str, Any] = field(default_factory=dict)
    workers: int = 8
    max_items: int | None = None     # per pool and group, for a pilot
    use_cache: bool = False
    max_consecutive_errors: int = 10
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 2.0, "typesafe": 0.01})
    bootstrap: int = 2000

    @classmethod
    def load(cls, path: str | Path) -> E116Config:
        return cls(**yaml.safe_load(Path(path).read_text()))

    def check(self) -> None:
        bad = [s for s in self.samples if not s.startswith("wildguard_") or "_train_" not in s]
        if bad:
            raise ValueError(f"E.116 develops on WildGuard training pools only; got {bad}")
        if STANDARD not in self.arms:
            raise ValueError("the standard arm is the paired control and must run")
        unknown = set(self.arms) - set(ARMS)
        if unknown:
            raise ValueError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")


def _rank(seed: str, sample: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample}:{item_id}".encode()).hexdigest()


def cell(label: str, jev_label: str) -> str:
    if label == jev_label:
        return RIGHT
    return OVER_FLAGGED if label == "unharmful" else LET_THROUGH


def select(mix: list[dict[str, Any]], config: E116Config, exclude: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """The scored items, the same for every arm: every Jev mistake, plus right_per_pool Jev-right items by hash."""
    picked = []
    for sample in config.samples:
        rows = [r for r in mix if r["sample"] == sample and r["format"] == REVIEW
                and (sample, r["item_id"]) not in exclude]
        wrong = sorted((r for r in rows if r["group"] == JEV_WRONG), key=lambda r: r["item_id"])
        right = sorted((r for r in rows if r["group"] == JEV_RIGHT), key=lambda r: _rank(config.seed, sample, r["item_id"]))
        right = right[: config.right_per_pool]
        if config.max_items:
            wrong, right = wrong[: config.max_items], right[: config.max_items]
        for r in [*wrong, *sorted(right, key=lambda r: r["item_id"])]:
            picked.append({"sample": sample, "item_id": r["item_id"], "task": r["task"], "label": r["label"],
                           "jev_label": r["jev_label"], "jev_confidence": r["jev_confidence"],
                           "group": r["group"], "cell": cell(r["label"], r["jev_label"])})
    return picked


def pool_counts(mix: list[dict[str, Any]], samples: dict[str, list[Item]], error_rate: dict[str, float]) -> dict[str, dict[str, int]]:
    """Each pool's Jev-answered items per cell, to weight the scored items back to the pool.

    The mix holds every Jev mistake, so the two wrong cells are exact. Jev-right items per label are
    the pool's items of that label less its mistakes, scaled by the share Jev answered."""
    out = {}
    for name, items in samples.items():
        wrong = Counter(cell(r["label"], r["jev_label"]) for r in mix if r["sample"] == name and r["group"] == JEV_WRONG)
        n_wrong = sum(wrong.values())
        answered = round(n_wrong / error_rate[name]) if error_rate.get(name) else len(items)
        share = min(1.0, answered / max(1, len(items)))
        by_label = Counter(i.label for i in items)
        right_harmful = round(by_label["harmful"] * share) - wrong[LET_THROUGH]
        right_unharmful = round(by_label["unharmful"] * share) - wrong[OVER_FLAGGED]
        out[name] = {OVER_FLAGGED: wrong[OVER_FLAGGED], LET_THROUGH: wrong[LET_THROUGH],
                     "right_harmful": max(0, right_harmful), "right_unharmful": max(0, right_unharmful)}
    return out


def project_cost(client: GuardedClient, n_reviews: int, config: E116Config) -> dict[str, float]:
    tin, tout = ASSUMED_TOKENS
    price = client.spec.tier_price or client.spec.price
    cap = (config.reviewer_reasoning_max_tokens or 0) + config.reviewer_max_output_tokens
    return {"expected_usd": round(n_reviews * (tin * price.input_per_m + tout * price.output_per_m) / 1e6, 4),
            "worst_usd": round(n_reviews * (tin * client.spec.price.input_per_m + cap * client.spec.price.output_per_m) / 1e6, 4)}


def run_e116(config: E116Config, run: Run, client: GuardedClient, *, items: dict[tuple[str, str], Item],
             picked: list[dict[str, Any]], texts: ArmTexts, counts: dict[str, dict[str, int]]) -> str:
    guard = client.guard
    if not config.use_cache:
        client.cache = None
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        config=dataclasses.asdict(config), models={config.reviewer: _spec_info(client.spec)},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        selection={"n": len(picked), "counts": dict(Counter(f"{p['sample']}/{p['cell']}" for p in picked)),
                   "sha256": hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()},
        pool_counts=counts, arm_texts={"sha256": texts.fingerprint(), "clarify": texts.clarify,
                                       "examples": {k: [dataclasses.asdict(e) for e in v] for k, v in texts.examples.items()}},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))})
    reviewers = {task: Reviewer(client, TASKS[task], temperature=config.reviewer_temperature,
                                reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                max_output_tokens=config.reviewer_max_output_tokens) for task in WILDGUARD_TASKS}
    kwargs = {(arm, task): arm_kwargs(arm, task, texts, items) for arm in config.arms for task in WILDGUARD_TASKS}
    writer = run.writer("reviews")
    done = {(r["arm"], r["sample"], r["item_id"]) for r in writer.rows() if r["status"] == "ok"}

    def work(t: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        arm, p = t
        item = items[(p["sample"], p["item_id"])]
        base = {"run_id": run.run_id, "reviewer": config.reviewer, "arm": arm, **p, "proposed_label": p["jev_label"],
                "proposed_correct": p["cell"] == RIGHT, "sample_index": 0}
        tags = {"experiment": EXPERIMENT, "step": "review", "arm": arm, "sample": p["sample"], "item_id": p["item_id"]}
        try:
            r = reviewers[p["task"]].decide(item.text, p["jev_label"], **kwargs[(arm, p["task"])], tags=tags)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                "raw_text": None if r.parse_ok else r.response.text[:2000], "request_key": r.request_key,
                **_usage(r.response), "ts": now_iso()}

    # Item by item, every arm of an item together, so a halted run leaves paired rows.
    todo = [(arm, p) for p in picked for arm in config.arms if (arm, p["sample"], p["item_id"]) not in done]
    status = "completed"
    try:
        run_parallel(todo, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
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


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e116.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--arms", nargs="+", choices=ARMS, help="override the config's arms (standard always runs)")
    parser.add_argument("--max-items", type=int, help="per pool and group, for a pilot")
    parser.add_argument("--dry-run", action="store_true", help="fake reviewer: no network, no cost")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E116Config.load(args.config)
    if args.arms:
        config = dataclasses.replace(config, arms=sorted(set(args.arms) | {STANDARD}, key=ARMS.index))
    if args.max_items:
        config = dataclasses.replace(config, max_items=args.max_items)
    config.check()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    run = Run(run_id, args.results_dir)
    if not args.report_only:
        texts = load_texts(config.arms)
        source = Path(args.results_dir) / "runs" / config.source_run
        mix = JsonlWriter(source / "mix.jsonl").rows()
        if not mix:
            raise SystemExit(f"{source} has no mix.jsonl; E.116 needs E.104's mix")
        error_rate = json.loads((source / "manifest.json").read_text())["jev_error_rate_by_sample"]
        # Pools are loaded in full (the examples may be any item); max_items limits the scored set only.
        samples, _ = load_samples(dataclasses.replace(config, max_items=None))
        items = {(name, i.item_id): i for name, its in samples.items() for i in its}
        if any(e.handwritten for rows in texts.examples.values() for e in rows):
            tests, _ = load_samples(dataclasses.replace(config, samples=config.test_samples, max_items=None))
            check_test_overlap(texts, [i for its in tests.values() for i in its])
        picked = select(mix, config, example_keys(texts))
        counts = pool_counts(mix, samples, error_rate)
        registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
        if args.dry_run:
            client = GuardedClient(ModelSpec(config.reviewer, "fake", f"fake-{config.reviewer}", Price(0.0, 0.0)),
                                   FakeChatProvider(), registry.guard, cache=None)
        else:
            try:
                client = registry.client(config.reviewer)
            except (ModelNotConfigured, RuntimeError) as e:
                raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None
        n_reviews = len(picked) * len(config.arms)
        cost = project_cost(client, n_reviews, config)
        project = registry.project_status()
        run.update_manifest(project_spend_at_start=project, cost_projection=cost)
        print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
        print(f"{len(picked)} items x {len(config.arms)} arms = {n_reviews} reviews; projected {cost} "
              f"(the run's Bedrock cap of ${config.run_provider_caps_usd.get('bedrock', 0):.2f} is enforced per call)")
        status = run_e116(config, run, client, items=items, picked=picked, texts=texts, counts=counts)
        print(f"run {run_id}: {status}")
    analyze_run(run.dir, config)
    print(f"results: {run.dir}\n{(run.dir / 'report.md').read_text()}")


if __name__ == "__main__":
    main()
