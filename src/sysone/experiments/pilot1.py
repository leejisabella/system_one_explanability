"""Pilot 1 (Experiment 1 pilot): Jev vs. explainer, with frozen explanations scored on hidden variants.

Builds on a completed Pilot 0 run, whose saved decisions are Jev's: Jev is not called
again. Jev's decision on each variant is the majority across its Pilot 0 samples, with
probabilities averaged; a tie leaves it undetermined, and a case whose original ties is
skipped. Three steps, each resumable:

1. Independent decisions (explainer_decisions.jsonl). Each explainer decides every
   original case and variant itself, several samples each, without seeing Jev's answer.
2. Frozen explanations (explanations.jsonl). Given the policy, the original case and
   Jev's decision, each explainer explains that decision, in each explained condition
   (free_form, structured). The decision_probabilities baseline needs no explainer: its
   "explanation" is Jev's own decision and probabilities. Explanations never see the
   variants and are never rewritten; a resumed run keeps the first one saved.
3. Simulations (simulations.jsonl). For every hidden variant of a case, the simulator
   predicts Jev's decision once without any explanation (the control) and once with
   each frozen explanation.

Usage:
  uv run python -m sysone.experiments.pilot1 --pilot0-run <id> --dry-run   # fake models, no cost
  uv run python -m sysone.experiments.pilot1 --pilot0-run <id> --run-id <id>

The benchmark's domain (from its meta; refund by default) sets the policy, the case
text, the options and the explainer and simulator prompts (sysone.benchmark.adapter).

Every model is locked to the first version that answers (saved per alias as
`model_versions` in the manifest), and the run halts if one changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml

from sysone.analysis.pilot1 import JevVariantDecision, analyze_run, jev_decisions
from sysone.benchmark.adapter import adapter_for
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
from sysone.clients.fake import FakeChatProvider
from sysone.explain.explainer import (
    EXPLAINED_CONDITIONS,
    ExplainerModel,
    decision_probabilities_text,
    explanation_record,
)
from sysone.explain.simulator import Simulator
from sysone.env import load_env
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso
from sysone.types import ChatResponse

EXPERIMENT = "pilot1"
DECISION_PROBABILITIES = "decision_probabilities"
CONDITIONS = (*EXPLAINED_CONDITIONS, DECISION_PROBABILITIES)
CONTROL = "control"
# The explainer recorded for explanations no explainer wrote (the decision_probabilities baseline).
NO_EXPLAINER = "none"

T = TypeVar("T")


@dataclass(frozen=True)
class Pilot1Config:
    # run_id of a completed Pilot 0 run under <results-dir>/runs. --pilot0-run overrides it.
    pilot0_run: str | None = None
    explainers: tuple[str, ...] = ("explainer_open", "explainer_gemini")
    conditions: tuple[str, ...] = CONDITIONS
    simulator: str = "simulator"
    # Independent decisions per explainer, per case variant (Jev has 3 per variant in Pilot 0).
    explainer_decision_samples: int = 3
    # None leaves the provider's default.
    explainer_temperature: float | None = None
    explainer_reasoning_max_tokens: int | None = None
    simulator_temperature: float | None = None
    simulator_reasoning_max_tokens: int | None = 2048
    # Bedrock's Converse API (the simulator's) has no seed parameter, so None.
    simulator_seed: int | None = None
    # Off by default, as in Pilot 0. Resuming a run never needs the cache.
    use_cache: bool = False
    max_consecutive_errors: int = 10

    def __post_init__(self) -> None:
        unknown = set(self.conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}; expected some of {CONDITIONS}")

    @classmethod
    def load(cls, path: str | Path) -> Pilot1Config:
        raw = yaml.safe_load(Path(path).read_text())
        for key in ("explainers", "conditions"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)


class _TooManyErrors(RuntimeError):
    pass


def model_family(spec: ModelSpec) -> str:
    """The model family, for keeping the simulator out of each explainer's family."""
    if spec.provider == "fake":
        return f"fake:{spec.alias}"
    if spec.provider == "typesafe":
        return "typesafe"
    if spec.provider == "gemini":
        return "google"
    if spec.provider == "openrouter":
        return spec.model.split("/")[0]
    if spec.provider == "bedrock":
        # "us.anthropic.claude-..." or "anthropic.claude-...": skip any routing prefix.
        parts = spec.model.split(".")
        return parts[1] if parts[0] in {"us", "eu", "apac", "global"} else parts[0]
    return spec.provider


def check_families(config: Pilot1Config, clients: dict[str, GuardedClient]) -> None:
    sim = clients[config.simulator].spec
    if model_family(sim) == "typesafe":
        raise ValueError("the simulator must not be Jev")
    for alias in config.explainers:
        if clients[alias].spec.model == sim.model or model_family(clients[alias].spec) == model_family(sim):
            raise ValueError(
                f"simulator {sim.model} is from the same family as explainer {alias} "
                f"({clients[alias].spec.model}); it would be scoring its own family"
            )


def _response_fields(response: ChatResponse, request_key: str) -> dict[str, Any]:
    return {
        "model_used": response.model_used,
        "cost_usd": response.cost_usd,
        "cached": response.cached,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "reasoning_tokens": response.usage.reasoning_tokens,
        "finish_reason": response.finish_reason,
        "service_tier": response.service_tier,
        "request_key": request_key,
    }


def _latest_ok(writer: JsonlWriter, key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    return {key(r): r for r in writer.rows() if r["status"] == "ok"}


def load_pilot0(pilot0_dir: Path) -> tuple[dict[str, Any], Path, list[Any], dict[str, Any]]:
    """A completed Pilot 0 run's manifest, and the benchmark it used (checked against its hash)."""
    manifest_path = pilot0_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no Pilot 0 run at {pilot0_dir}")
    p0 = json.loads(manifest_path.read_text())
    if p0.get("experiment") != "pilot0" or p0.get("status") != "completed":
        raise ValueError(f"{pilot0_dir} is not a completed Pilot 0 run (status: {p0.get('status')})")
    benchmark_dir = Path(p0["benchmark"]["path"])
    cases, meta = load_benchmark(benchmark_dir)
    if meta["sha256"] != p0["benchmark"]["sha256"]:
        raise ValueError(f"benchmark {benchmark_dir} no longer matches the one Pilot 0 used")
    return p0, benchmark_dir, cases, meta


def run_pilot1(config: Pilot1Config, run: Run, clients: dict[str, GuardedClient], *, pilot0_dir: Path) -> str:
    """Run (or resume) Pilot 1 and return the final run status."""
    p0, benchmark_dir, cases, meta = load_pilot0(pilot0_dir)
    adapter = adapter_for(meta)
    policy, setting = adapter.policy, adapter.setting
    jev = jev_decisions(pilot0_dir)
    none = JevVariantDecision(None, None, 0)

    needed = {config.simulator, *config.explainers}
    missing = needed - set(clients)
    if missing:
        raise ValueError(f"no client for {sorted(missing)}")
    check_families(config, clients)

    versions = dict(run.manifest().get("model_versions") or {})

    def record_version(alias: str, version: str) -> None:
        versions[alias] = version
        run.update_manifest(model_versions=versions)

    for alias in needed:
        client = clients[alias]
        if not config.use_cache:
            client.cache = None
        client.version_lock = VersionLock(versions.get(alias), on_lock=lambda v, a=alias: record_version(a, v))

    decisions_w, explanations_w, simulations_w = (
        run.writer("explainer_decisions"), run.writer("explanations"), run.writer("simulations"))
    resumed = bool(decisions_w.rows() or explanations_w.rows() or simulations_w.rows())
    skipped = sorted(c.case_id for c in cases if jev.get(c.original.variant_id, none).label is None)
    guard = clients[config.simulator].guard
    run.update_manifest(
        experiment=EXPERIMENT,
        status="running",
        started_at=run.manifest().get("started_at", now_iso()),
        resumed_at=now_iso() if resumed else None,
        pilot0={"run_id": pilot0_dir.name, "dir": str(pilot0_dir), "model_version": p0.get("model_version")},
        benchmark={"path": str(benchmark_dir), "name": meta["name"], "sha256": meta["sha256"]},
        domain=adapter.domain,
        config=dataclasses.asdict(config),
        models={
            alias: {
                "provider": c.spec.provider,
                "model": c.spec.model,
                "family": model_family(c.spec),
                "price_per_m": dataclasses.asdict(c.spec.price) if c.spec.price else None,
            }
            for alias, c in sorted(clients.items()) if alias in needed
        },
        model_versions=versions,
        # Cases whose original Jev decision tied across Pilot 0 samples: not explained or simulated.
        skipped_cases_jev_tie=skipped,
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd},
        git=git_info(),
        config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )

    consecutive_errors = 0

    def attempt(writer: JsonlWriter, base: dict[str, Any], fn: Callable[[], T]) -> T | None:
        nonlocal consecutive_errors
        try:
            result = fn()
        except ModelVersionChanged as e:
            # The call is in the ledger; record it here too, then halt.
            writer.append({**base, "status": "error", "error": str(e), "model_used": e.got, "ts": now_iso()})
            raise
        except ProviderError as e:
            writer.append({**base, "status": "error", "error": str(e), "ts": now_iso()})
            consecutive_errors += 1
            if consecutive_errors >= config.max_consecutive_errors:
                raise _TooManyErrors(f"{consecutive_errors} failed requests in a row; last: {e}") from e
            return None
        consecutive_errors = 0
        return result

    texts = {v.variant_id: adapter.render(v) for c in cases for v in c.variants}
    explainable = [c for c in cases if c.case_id not in skipped]

    status = "completed"
    try:
        # 1. Independent decisions.
        done = set(_latest_ok(decisions_w, lambda r: (r["explainer"], r["variant_id"], r["sample_index"])))
        for alias in config.explainers:
            explainer = ExplainerModel(clients[alias], temperature=config.explainer_temperature,
                                       reasoning_max_tokens=config.explainer_reasoning_max_tokens, setting=setting)
            for case in cases:
                for variant in case.variants:
                    for sample in range(config.explainer_decision_samples):
                        if (alias, variant.variant_id, sample) in done:
                            continue
                        base = {
                            "run_id": run.run_id, "explainer": alias, "case_id": case.case_id,
                            "case_type": str(case.case_type), "variant_id": variant.variant_id,
                            "variant_kind": str(variant.kind), "sample_index": sample,
                            "policy_outcome": str(variant.policy_outcome), "true_outcome": str(case.true_outcome),
                            "jev_decision": jev.get(variant.variant_id, none).label,
                            "model_requested": clients[alias].spec.model,
                        }
                        tags = {"experiment": EXPERIMENT, "step": "explainer_decision", "explainer": alias,
                                "case_id": case.case_id, "variant_id": variant.variant_id,
                                "sample_index": str(sample)}
                        result = attempt(decisions_w, base, lambda: explainer.decide(
                            policy, texts[variant.variant_id], sample_index=sample, tags=tags))
                        if result is None:
                            continue
                        decisions_w.append({
                            **base, "status": "ok", "decision": result.label, "parse_ok": result.parse_ok,
                            "parse_error": result.parse_error, "raw_text": result.response.text,
                            "option_order": list(result.option_order),
                            **_response_fields(result.response, result.request_key), "ts": now_iso(),
                        })

        # 2. Frozen explanations.
        frozen = _latest_ok(explanations_w, lambda r: r["explanation_id"])
        for condition in config.conditions:
            for alias in ([NO_EXPLAINER] if condition == DECISION_PROBABILITIES else config.explainers):
                for case in explainable:
                    explanation_id = f"{alias}/{condition}/{case.case_id}"
                    if explanation_id in frozen:
                        continue
                    jev_original = jev[case.original.variant_id]
                    base = {
                        "run_id": run.run_id, "explanation_id": explanation_id, "explainer": alias,
                        "condition": condition, "case_id": case.case_id, "case_type": str(case.case_type),
                        "variant_id": case.original.variant_id, "jev_decision": jev_original.label,
                        "jev_probabilities": jev_original.probabilities,
                    }
                    if condition == DECISION_PROBABILITIES:
                        text = decision_probabilities_text(jev_original.label, jev_original.probabilities,
                                                           setting.labels)
                        row = {**base, "status": "ok", "explanation_text": text, "basis": None, "summary": None,
                               "parse_ok": True, "parse_error": None, "raw_text": None, "cost_usd": 0.0,
                               "ts": now_iso()}
                    else:
                        explainer = ExplainerModel(clients[alias], temperature=config.explainer_temperature,
                                                   reasoning_max_tokens=config.explainer_reasoning_max_tokens,
                                                   setting=setting)
                        tags = {"experiment": EXPERIMENT, "step": "explanation", "explainer": alias,
                                "condition": condition, "case_id": case.case_id}
                        result = attempt(explanations_w, base, lambda: explainer.explain(
                            condition, policy, texts[case.original.variant_id], jev_original.label, tags=tags))
                        if result is None:
                            continue
                        row = {**base, "status": "ok", **explanation_record(result),
                               "model_requested": clients[alias].spec.model,
                               **_response_fields(result.response, result.request_key), "ts": now_iso()}
                    explanations_w.append(row)
                    frozen[explanation_id] = row

        # 3. Simulations on the hidden variants: the control, then each usable explanation.
        simulator = Simulator(clients[config.simulator], temperature=config.simulator_temperature,
                              reasoning_max_tokens=config.simulator_reasoning_max_tokens, seed=config.simulator_seed,
                              setting=setting)
        by_case = {c.case_id: c for c in explainable}
        targets = [(f"{CONTROL}/{c.case_id}", CONTROL, CONTROL, c.case_id, None) for c in explainable]
        targets += [
            (r["explanation_id"], r["explainer"], r["condition"], r["case_id"], r["explanation_text"])
            for r in frozen.values()
            if r["parse_ok"] and r["condition"] in config.conditions and r["case_id"] in by_case
        ]
        done = set(_latest_ok(simulations_w, lambda r: (r["explanation_id"], r["variant_id"])))
        for explanation_id, alias, condition, case_id, explanation_text in targets:
            case = by_case[case_id]
            jev_original = jev[case.original.variant_id]
            for variant in case.variants[1:]:
                if (explanation_id, variant.variant_id) in done:
                    continue
                base = {
                    "run_id": run.run_id, "explanation_id": explanation_id, "explainer": alias,
                    "condition": condition, "with_explanation": explanation_text is not None,
                    "case_id": case_id, "case_type": str(case.case_type), "variant_id": variant.variant_id,
                    "variant_kind": str(variant.kind), "policy_outcome": str(variant.policy_outcome),
                    "true_outcome": str(case.true_outcome), "jev_original_decision": jev_original.label,
                    # Jev's decision on this variant. Saved for scoring; never shown to the simulator.
                    "jev_decision": jev.get(variant.variant_id, none).label,
                    "model_requested": simulator.client.spec.model,
                }
                tags = {"experiment": EXPERIMENT, "step": "simulation", "explanation_id": explanation_id,
                        "variant_id": variant.variant_id}
                result = attempt(simulations_w, base, lambda: simulator.simulate(
                    policy, texts[case.original.variant_id], jev_original.label, explanation_text,
                    texts[variant.variant_id], tags=tags))
                if result is None:
                    continue
                simulations_w.append({
                    **base, "status": "ok", "predicted_decision": result.predicted,
                    "probabilities": result.probabilities, "basis": result.basis_raw,
                    "basis_ok": result.basis is not None, "basis_error": result.basis_error,
                    "covered": result.covered, "parse_ok": result.parse_ok, "parse_error": result.parse_error,
                    "repaired": result.repaired,
                    "raw_text": result.response.text, **_response_fields(result.response, result.request_key),
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
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6))
    return status


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "pilot1.yaml"))
    parser.add_argument("--pilot0-run", help="run_id of a completed Pilot 0 run (overrides the config)")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="use fake explainers and simulator: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = Pilot1Config.load(args.config)
    pilot0_run = args.pilot0_run or config.pilot0_run
    if not pilot0_run:
        raise SystemExit("set --pilot0-run (or pilot0_run in the config) to a completed Pilot 0 run")
    pilot0_dir = Path(args.results_dir) / "runs" / pilot0_run
    try:
        load_pilot0(pilot0_dir)  # fail before creating a run folder
    except ValueError as e:
        raise SystemExit(str(e)) from None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"

    registry = Registry(run_id, results_dir=Path(args.results_dir))
    aliases = [*config.explainers, config.simulator]
    clients: dict[str, GuardedClient] = {}
    for alias in aliases:
        if args.dry_run:
            spec = ModelSpec(alias, "fake", f"fake-{alias}", Price(0.0, 0.0))
            clients[alias] = GuardedClient(spec, FakeChatProvider(), registry.guard, cache=None)
            continue
        try:
            clients[alias] = registry.client(alias)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    # Spend across all runs so far, per provider, against the project caps in configs/budget.yaml.
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_pilot1(config, run, clients, pilot0_dir=pilot0_dir)
    analyze_run(run.dir)
    print(f"run {run_id}: {status}")
    print(f"results: {run.dir}")
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
