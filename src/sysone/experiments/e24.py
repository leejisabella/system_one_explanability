"""E.24: minimal-pair probes of the shared blind spot (rules in configs/experiments/e24.yaml).

Steps, all on the frozen probe set (benchmarks/probes/minimal_pairs_v1):
  jev      Jev (pinned jev-1.14.0) decides every item jev_samples times.
  alone    gpt-oss-20b x5, gpt-oss-120b x5, Mistral Large 3 x3 (written reasoning), Qwen3-32B x1, alone.
  cascade  gpt-oss-20b reviews Jev's combined label once per item.

Usage:
  uv run python -m sysone.experiments.e24 --run-id e24-pilot-<date> --max-items 6
  uv run python -m sysone.experiments.e24 --run-id e24-v1-<date>
  uv run python -m sysone.experiments.e24 --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.cascade.items import Item
from sysone.cascade.proposals import Proposal, combine_jev
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import ProviderError
from sysone.env import load_env
from sysone.experiments.e101 import _TooManyErrors, _usage, jev_step, run_parallel
from sysone.experiments.e112 import WRITTEN_PREFIX, parse_teacher
from sysone.experiments.e22 import ALONE_PREFIX
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, Registry
from sysone.runs import Run, git_info, now_iso

EXPERIMENT = "e24"
MISTRAL = "reviewer_mistral"
USD_PER_CALL = {"jev_1_14": 0.00003, "reviewer": 0.00008, "reviewer_120b": 0.00013, MISTRAL: 0.00077, "reviewer_qwen": 0.00003}


def load_probes(probe_dir: Path) -> dict[str, list[Item]]:
    text = (probe_dir / "items.jsonl").read_text()
    man = json.loads((probe_dir / "manifest.json").read_text())
    if hashlib.sha256(text.encode()).hexdigest() != man["items_sha256"]:
        raise SystemExit("probe items.jsonl doesn't match its manifest's sha256")
    out: dict[str, list[Item]] = defaultdict(list)
    for line in text.splitlines():
        d = json.loads(line)
        out[d["dataset"]].append(Item(**d))
    return dict(out)


def work_fn(run: Run, models: dict[tuple[str, str], Reviewer]):
    def work(t):
        alias, name, item, p, k = t
        condition = p.condition if p else "alone"
        base = {"run_id": run.run_id, "reviewer": alias, "role": "reviewer", "sample": name, "item_id": item.item_id,
                "task": item.task, "condition": condition, "sample_index": k, "label": item.label, "strata": item.strata,
                "proposed_label": p.label if p else None, "proposed_correct": p.correct if p else None}
        rv = models[(alias, name)]
        tags = {"experiment": EXPERIMENT, "step": condition, "reviewer": alias, "sample": name, "item_id": item.item_id,
                "sample_index": str(k)}
        prefix = (WRITTEN_PREFIX if p else ALONE_PREFIX) if alias == MISTRAL else None
        req = rv.build_request(item.text, p.label if p else None, task_prefix=prefix, sample_index=k, tags=tags)
        try:
            resp = rv.client.complete(req)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        r, written = parse_teacher(resp, rv.client.request_key(req), rv.task.labels, p.label if p else None)
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error, "reasoning": written if alias == MISTRAL else None,
                "raw_text": None if r.parse_ok else resp.text[:2000], "request_key": r.request_key, **_usage(resp), "ts": now_iso()}
    return work


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e24.yaml"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-items", type=int, help="pilot: items per probe dataset")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run = Run(a.run_id, RESULTS_DIR)
    if not a.report_only:
        samples = load_probes(REPO_ROOT / cfg["probe_set"])
        if a.max_items:
            samples = {n: its[:a.max_items] for n, its in samples.items()}
        n_items = sum(len(v) for v in samples.values())
        registry = Registry(a.run_id, run_provider_caps_usd=cfg["run_provider_caps_usd"])
        clients = {alias: registry.client(alias) for alias in cfg["models"]}
        for c in clients.values():
            c.cache = None
        calls = {alias: n_items * k for alias, k in cfg["models"].items()}
        calls["reviewer"] += n_items * cfg["cascade"]["samples"]
        projected = sum(USD_PER_CALL[al] * n for al, n in calls.items())
        project = registry.project_status()
        print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items() if v["cap_usd"]))
        print(f"{n_items} items; calls {calls}; projected ${projected:.2f} (caps {cfg['run_provider_caps_usd']})")
        run.update_manifest(experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
                            config=cfg, git=git_info(), project_spend_at_start=project, n_items=n_items,
                            probe_manifest=json.loads((REPO_ROOT / cfg["probe_set"] / "manifest.json").read_text())["items_sha256"],
                            cost_projection={"calls": calls, "expected_usd": round(projected, 2)})
        status = "completed"
        try:
            jev_cfg = SimpleNamespace(jev_samples=cfg["jev_samples"], workers=cfg["workers"], max_consecutive_errors=cfg["max_consecutive_errors"])
            jev_step(jev_cfg, run, clients["jev_1_14"], samples, experiment=EXPERIMENT)
            models: dict[tuple[str, str], Reviewer] = {}
            for alias in cfg["models"]:
                if alias == "jev_1_14":
                    continue
                for name, its in samples.items():
                    native = alias in ("reviewer", "reviewer_120b")
                    models[(alias, name)] = Reviewer(
                        clients[alias], TASKS[its[0].task], temperature=None,
                        reasoning_max_tokens=cfg["reviewer_reasoning_max_tokens"] if native else None,
                        max_output_tokens=cfg["written_max_output_tokens"] if alias == MISTRAL else cfg["reviewer_max_output_tokens"])
            writer = run.writer("reviews")
            done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"]) for r in writer.rows() if r["status"] == "ok"}
            # Jev's combined label per item, for the cascade
            probs: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for r in run.writer("jev_decisions").rows():
                if r["status"] == "ok" and r["parse_ok"]:
                    probs[(r["sample"], r["item_id"])].append(r["probabilities"])

            def tasks():
                for name, its in samples.items():
                    for item in its:
                        for alias, k_n in cfg["models"].items():
                            if alias == "jev_1_14":
                                continue
                            for k in range(k_n):
                                if (alias, name, item.item_id, "alone", k) not in done:
                                    yield (alias, name, item, None, k)
                        d = combine_jev(probs.get((name, item.item_id), []), TASKS[item.task].labels)
                        if d is not None and ("reviewer", name, item.item_id, "jev_label", 0) not in done:
                            yield ("reviewer", name, item, Proposal(item.item_id, "jev_label", d.label, d.label == item.label, None), 0)

            run_parallel(tasks(), work_fn(run, models), writer, workers=cfg["workers"], max_consecutive_errors=cfg["max_consecutive_errors"])
        except (BudgetError, _TooManyErrors) as e:
            status = f"halted: {e}"
        finally:
            run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(registry.guard.spent_usd, 6),
                                spent_by_provider={p: round(registry.guard.run_spent_usd(p), 6) for p in ("bedrock", "typesafe")})
        print(f"run {a.run_id}: {status}; spent ${registry.guard.spent_usd:.4f}")
    from sysone.analysis.e24 import analyze_run
    analyze_run(run.dir, cfg)
    print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
