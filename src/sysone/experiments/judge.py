"""Judge a completed Pilot 1 run's frozen explanations (contestability and justification coding).

Reads the Pilot 1 run, the Pilot 0 run it builds on, and their benchmark; writes
judgments.jsonl into its own run folder, so the Pilot 1 run stays as it was. One
judgment per explanation written by an explainer (the decision_probabilities baseline
has no text to judge). Resumable: judgments already saved are skipped.

Usage:
  uv run python -m sysone.experiments.judge --pilot1-run <id> --dry-run   # fake judge, no cost
  uv run python -m sysone.experiments.judge --pilot1-run <id> --run-id <id>

The judge is locked to the first version that answers, and must not share a model
family with any explainer it scores.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.analysis import pipelines
from sysone.analysis.judge import analyze_run
from sysone.analysis.pilot1 import _latest_ok, jev_decisions
from sysone.analysis.pilot0 import _read_jsonl
from sysone.benchmark.adapter import adapter_for
from sysone.benchmark.store import load_benchmark
from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ModelVersionChanged, ProviderError, VersionLock
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.pilot1 import model_family
from sysone.explain.judge import Judge, evaluator_facts, quote_found
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import Run, git_info, now_iso

EXPERIMENT = "judge"


@dataclass(frozen=True)
class JudgeConfig:
    pilot1_run: str | None = None
    judge: str = "judge"
    temperature: float | None = None
    reasoning_max_tokens: int | None = 1024
    use_cache: bool = False
    max_consecutive_errors: int = 10

    @classmethod
    def load(cls, path: str | Path) -> JudgeConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))


class _TooManyErrors(RuntimeError):
    pass


def error_kind(jev: str | None, policy: str, truth: str) -> str:
    """Why Jev's decision on the original case is right or wrong."""
    if jev is None:
        return "jev tied"
    if jev == truth:
        return "jev correct"
    # Jev applied the policy to a corrupted or incomplete record (a planted defect).
    return "follows presented record" if jev == policy else "misapplies policy"


def load_pilot1(pilot1_dir: Path) -> dict[str, Any]:
    path = pilot1_dir / "manifest.json"
    if not path.exists():
        raise ValueError(f"no Pilot 1 run at {pilot1_dir}")
    p1 = json.loads(path.read_text())
    if p1.get("experiment") != "pilot1" or p1.get("status") != "completed":
        raise ValueError(f"{pilot1_dir} is not a completed Pilot 1 run (status: {p1.get('status')})")
    return p1


def run_judge(config: JudgeConfig, run: Run, client: GuardedClient, *, pilot1_dir: Path) -> str:
    """Run (or resume) the judge over a Pilot 1 run and return the final status."""
    p1 = load_pilot1(pilot1_dir)
    cases, meta = load_benchmark(p1["benchmark"]["path"])
    if meta["sha256"] != p1["benchmark"]["sha256"]:
        raise ValueError(f"benchmark {p1['benchmark']['path']} no longer matches the one Pilot 1 used")
    adapter = adapter_for(meta)
    by_case = {c.case_id: c for c in cases}
    jev = jev_decisions(Path(p1["pilot0"]["dir"]))

    families = {alias: m["family"] for alias, m in p1["models"].items() if alias in p1["config"]["explainers"]}
    if model_family(client.spec) in set(families.values()) or model_family(client.spec) == "typesafe":
        raise ValueError(f"judge {client.spec.model} shares a family with an explainer it scores ({families})")

    if not config.use_cache:
        client.cache = None
    locked = run.manifest().get("model_version")
    client.version_lock = VersionLock(locked, on_lock=lambda v: run.update_manifest(model_version=v))
    judge = Judge(client, temperature=config.temperature, reasoning_max_tokens=config.reasoning_max_tokens,
                  setting=adapter.setting)
    writer = run.writer("judgments")
    done = {r["explanation_id"] for r in writer.rows() if r["status"] == "ok"}
    explanations = [e for e in _latest_ok(_read_jsonl(pilot1_dir / "explanations.jsonl"), "explanation_id")
                    if e["parse_ok"] and e["explanation_text"] and e["explainer"] != "none"]

    run.update_manifest(
        experiment=EXPERIMENT,
        status="running",
        started_at=run.manifest().get("started_at", now_iso()),
        resumed_at=now_iso() if done else None,
        pilot1={"run_id": pilot1_dir.name, "dir": str(pilot1_dir)},
        benchmark=p1["benchmark"],
        domain=adapter.domain,
        config=dataclasses.asdict(config),
        model={"alias": client.spec.alias, "provider": client.spec.provider, "model": client.spec.model,
               "family": model_family(client.spec),
               "price_per_m": dataclasses.asdict(client.spec.price) if client.spec.price else None},
        model_version=client.version_lock.expected,
        budget={"per_call_cap_usd": client.guard.per_call_cap_usd, "run_cap_usd": client.guard.run_cap_usd},
        git=git_info(),
        config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )

    status, errors = "completed", 0
    try:
        for e in explanations:
            if e["explanation_id"] in done:
                continue
            case = by_case[e["case_id"]]
            original = case.original
            policy_outcome = str(original.policy_outcome)
            # Jev's decision, or (own_decision condition) the explainer's own.
            jev_label = e.get("explained_decision") or e["jev_decision"]
            base = {
                "run_id": run.run_id, "explanation_id": e["explanation_id"], "explainer": e["explainer"],
                "condition": e["condition"], "case_id": case.case_id, "case_type": str(case.case_type),
                "jev_decision": e["jev_decision"], "explained_decision": jev_label, "policy_outcome": policy_outcome, "true_outcome": str(case.true_outcome),
                "error_kind": error_kind(jev_label, policy_outcome, str(case.true_outcome)),
                "defect_fields": [d.field for d in case.defects if d.type in ("I", "II")],
                "model_requested": client.spec.model,
            }
            tags = {"experiment": EXPERIMENT, "step": "judgment", "explanation_id": e["explanation_id"]}
            try:
                j = judge.judge(adapter.policy, adapter.render(original), jev_label, e["explanation_text"],
                                evaluator_facts(case, jev_label, policy_outcome), tags=tags)
            except ModelVersionChanged as exc:
                writer.append({**base, "status": "error", "error": str(exc), "model_used": exc.got, "ts": now_iso()})
                raise
            except ProviderError as exc:
                writer.append({**base, "status": "error", "error": str(exc), "ts": now_iso()})
                errors += 1
                if errors >= config.max_consecutive_errors:
                    raise _TooManyErrors(f"{errors} failed requests in a row; last: {exc}") from exc
                continue
            errors = 0
            r = j.response
            writer.append({
                **base, "status": "ok", "justification": j.justification,
                "names_defect_field": j.names_defect_field, "questions_defect_value": j.questions_defect_value,
                "cites_correcting_evidence": j.cites_correcting_evidence, "recommends_review": j.recommends_review,
                "quote": j.quote, "quote_found": quote_found(j.quote, e["explanation_text"]),
                "parse_ok": j.parse_ok, "parse_error": j.parse_error, "repaired": j.repaired, "raw_text": r.text,
                "model_used": r.model_used, "cost_usd": r.cost_usd, "cached": r.cached,
                "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                "reasoning_tokens": r.usage.reasoning_tokens, "finish_reason": r.finish_reason,
                "service_tier": r.service_tier, "latency_s": r.latency_s, "wall_s": r.wall_s,
                "request_key": j.request_key, "ts": now_iso(),
            })
    except (BudgetError, _TooManyErrors, ModelVersionChanged) as exc:
        status = f"halted: {exc}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status = f"failed: {exc!r}"
        raise
    finally:
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(client.guard.spent_usd, 6))
    return status


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "judge.yaml"))
    parser.add_argument("--pilot1-run", help="run_id of a completed Pilot 1 run (overrides the config)")
    parser.add_argument("--run-id")
    parser.add_argument("--labels", nargs="*", default=[], help="human label CSVs to score the judge against")
    parser.add_argument("--dry-run", action="store_true", help="use a fake judge: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = JudgeConfig.load(args.config)
    pilot1_run = args.pilot1_run or config.pilot1_run
    if not pilot1_run:
        raise SystemExit("set --pilot1-run (or pilot1_run in the config) to a completed Pilot 1 run")
    pilot1_dir = Path(args.results_dir) / "runs" / pilot1_run
    try:
        load_pilot1(pilot1_dir)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    registry = Registry(run_id, results_dir=Path(args.results_dir))
    if args.dry_run:
        spec = ModelSpec(config.judge, "fake", f"fake-{config.judge}", Price(0.0, 0.0))
        client = GuardedClient(spec, FakeChatProvider(), registry.guard, cache=None)
    else:
        try:
            client = registry.client(config.judge)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None
    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_judge(config, run, client, pilot1_dir=pilot1_dir)
    analyze_run(run.dir, labels=[Path(p) for p in args.labels])
    pipelines.analyze_run(pilot1_dir, run.dir)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())
    print((run.dir / "pipelines.md").read_text())


if __name__ == "__main__":
    main()
