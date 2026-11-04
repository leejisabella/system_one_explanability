"""E.104: Jev screens the Part 2 training pools, and the training mix is saved (research_docs/14).

  jev   Jev decides every item of every training pool, jev_samples times (as in E.101/E.102).
  mix   the training mix (sysone.posttrain.mix), built from Jev's decisions and saved once to
        mix.jsonl; a resume that would change it is refused.

The report compares Jev's error rate on each training pool with its rate on the matching E.102
test pool. A much lower rate on a training pool would suggest Jev was trained on it (TypeSafe's
"Jev boost" sets include Civil Comments' train split, which the pools already avoid).

Usage:
  uv run python -m sysone.experiments.e104 --dry-run       # fake Jev, no network, no cost
  uv run python -m sysone.experiments.e104 --run-id <id>   # live (TypeSafe only)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.proposals import combine_jev
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ModelVersionChanged, VersionLock
from sysone.clients.fake import FakeDecisionProvider
from sysone.env import load_env
from sysone.experiments.e101 import _JevIncomplete, _spec_info, _TooManyErrors, jev_step, load_samples
from sysone.posttrain.mix import MixSettings, build_mix, mix_counts
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e104"
STEPS = ("jev", "mix")


@dataclass(frozen=True)
class E104Config:
    # Training pools (benchmarks/cascade/<name>), never test samples.
    samples: list[str]
    jev: str = "jev"
    jev_samples: int = 3
    mix: dict[str, Any] = field(default_factory=dict)
    # The E.102 run whose Jev error rates the report compares against, by test pool.
    test_run: str | None = None
    test_pool_of: dict[str, str] = field(default_factory=dict)
    workers: int = 8
    max_items: int | None = None
    expected_model_version: str | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"typesafe": 2.0})

    @classmethod
    def load(cls, path: str | Path) -> E104Config:
        return cls(**yaml.safe_load(Path(path).read_text()))

    @property
    def mix_settings(self) -> MixSettings:
        return MixSettings(**self.mix)


def jev_decisions(run: Run, samples: dict[str, list[Item]], jev_samples: int) -> dict[str, dict[str, dict[str, Any]]]:
    """Jev's combined decision per item, by sample. Refuses if any item lacks some of its answers."""
    parsed: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    answered: dict[tuple[str, str], int] = defaultdict(int)
    for r in run.writer("jev_decisions").rows():
        if r["status"] == "ok":
            answered[(r["sample"], r["item_id"])] += 1
            if r["parse_ok"]:
                parsed[(r["sample"], r["item_id"])].append(r["probabilities"])
    missing = sum(answered[(name, i.item_id)] < jev_samples for name, items in samples.items() for i in items)
    if missing:
        raise _JevIncomplete(f"{missing} items lack some of Jev's {jev_samples} answers; resume with "
                             "steps [jev] to retry the failed requests before building the mix")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, items in samples.items():
        labels = TASKS[items[0].task].labels
        out[name] = {i.item_id: {"label": d.label, "confidence": d.confidence, "split": d.split}
                     for i in items if (d := combine_jev(parsed[(name, i.item_id)], labels)) is not None}
    return out


def mix_step(config: E104Config, run: Run, samples: dict[str, list[Item]]) -> list[dict[str, Any]]:
    jev = jev_decisions(run, samples, config.jev_samples)
    rows = [r for name, items in samples.items()
            for r in build_mix(name, items, jev[name], TASKS[items[0].task].labels, config.mix_settings)]
    writer = run.writer("mix")
    saved = writer.rows()
    if saved and saved != json.loads(json.dumps(rows)):
        raise ValueError(f"{writer.path} differs from the mix Jev's rows give now; start a new run_id")
    if not saved:
        for row in rows:
            writer.append(row)
    error_rate = {name: round(sum(d["label"] != i.label for i in items if (d := jev[name].get(i.item_id)))
                              / max(1, len(jev[name])), 4) for name, items in samples.items()}
    run.update_manifest(jev_error_rate_by_sample=error_rate, mix_counts=mix_counts(rows))
    return rows


def report(run: Run, config: E104Config, results_dir: Path) -> str:
    m = run.manifest()
    test: dict[str, float] = {}
    plan_path = results_dir / "runs" / str(config.test_run) / "plan.jsonl"
    if config.test_run and plan_path.exists():
        plan = JsonlWriter(plan_path).rows()
        for s in {p["sample"] for p in plan}:
            rows = [p for p in plan if p["sample"] == s]
            test[s] = sum(not p["jev_correct"] for p in rows) / len(rows)
    lines = [f"# E.104 report: {run.run_id}", "", f"Status: {m.get('status')}. Spent ${m.get('spent_usd', 0):.4f}.", "",
             "| Training pool | Jev error rate | Test pool | Jev error rate on test |", "|---|---|---|---|"]
    for s, rate in (m.get("jev_error_rate_by_sample") or {}).items():
        t = config.test_pool_of.get(s)
        lines.append(f"| {s} | {rate:.1%} | {t or '–'} | {f'{test[t]:.1%}' if t in test else '–'} |")
    lines += ["", "| Training pool | Mix (group/split: rows) |", "|---|---|"]
    for s, counts in (m.get("mix_counts") or {}).items():
        lines.append(f"| {s} | {', '.join(f'{k} {v}' for k, v in counts.items())} |")
    return "\n".join(lines) + "\n"


def run_e104(config: E104Config, run: Run, jev_client: GuardedClient, *, samples: dict[str, list[Item]],
             manifests: dict[str, dict]) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    config.mix_settings  # fail fast on an unknown mix setting
    if not config.use_cache:
        jev_client.cache = None
    locked = run.manifest().get("model_version")
    if locked and config.expected_model_version and locked != config.expected_model_version:
        raise ValueError(f"run {run.run_id} is locked to {locked}, but the config expects "
                         f"{config.expected_model_version}; start a new run_id to use another version")
    jev_client.version_lock = VersionLock(locked or config.expected_model_version,
                                          on_lock=lambda v: run.update_manifest(model_version=v))
    guard = jev_client.guard
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        samples={name: {k: m[k] for k in ("name", "sha256", "n_items", "source", "task")} for name, m in manifests.items()},
        config=dataclasses.asdict(config), models={"jev": _spec_info(jev_client.spec)},
        model_version=jev_client.version_lock.expected,
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    status = "completed"
    try:
        if "jev" in config.steps:
            jev_step(config, run, jev_client, samples, experiment=EXPERIMENT)
        if "mix" in config.steps:
            mix_step(config, run, samples)
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
                            spent_by_provider={jev_client.spec.provider: round(guard.run_spent_usd(jev_client.spec.provider), 6)})
    return status


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e104.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dry-run", action="store_true", help="fake Jev: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E104Config.load(args.config)
    if args.steps:
        config = dataclasses.replace(config, steps=args.steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, manifests = load_samples(config)

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    if args.dry_run:
        jev_client = GuardedClient(ModelSpec("jev", "fake", "jev-latest", Price(0.0, 0.0)), FakeDecisionProvider(),
                                   registry.guard, cache=None)
    else:
        try:
            jev_client = registry.client(config.jev)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e104(config, run, jev_client, samples=samples, manifests=manifests)
    text = report(run, config, Path(args.results_dir))
    (run.dir / "report.md").write_text(text)
    print(f"run {run_id}: {status}\nresults: {run.dir}\n{text}")


if __name__ == "__main__":
    main()
