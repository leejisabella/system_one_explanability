"""E.22: a reviewer from another family (Mistral Large 3) on WildGuard (rules in configs/experiments/e22.yaml).

E.119's design on E.102's frozen plan, WildGuard prompts and responses only: every reviewed item alone x
alone_samples, plus one review of each proposal (Jev's wrong label, the matched random wrong label, the
subsample's correct label). Mistral writes its reasoning before the JSON (E.112's "written" mode).
Result rows have E.102's schema, so E.102's and E.119's analyses run on them.

Usage:
  uv run python -m sysone.experiments.e22 --run-id e22-pilot-<date> --max-items 20
  uv run python -m sysone.experiments.e22 --run-id e22-v1-<date>
  uv run python -m sysone.experiments.e22 --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.cascade.items import Item, load_items
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import ProviderError
from sysone.env import load_env
from sysone.experiments.e101 import _TooManyErrors, _usage, run_parallel
from sysone.experiments.e102 import _key, review_tasks
from sysone.experiments.e112 import WRITTEN_PREFIX, parse_teacher
from sysone.experiments.e119 import pilot_plan
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e22"
ALONE_PREFIX = (
    "Think it through before you answer: first write out your reasoning step by step as plain text (not "
    "inside the JSON object and not in a code block), applying the policy to the content. Then end your reply "
    "with the JSON object described below, and nothing after it."
)
ASSUMED_USD_PER_CALL = 0.0006   # E.112's Mistral calls in written mode averaged $0.00059


class _Cfg:
    def __init__(self, d: dict[str, Any]):
        self.__dict__.update(d)


def work_fn(run: Run, models: dict[str, Reviewer]):
    def work(t):
        alias, name, item, p, k = t
        condition = p.condition if p else "alone"
        base = {"run_id": run.run_id, "reviewer": alias, "role": "reviewer", "sample": name, "item_id": item.item_id,
                "task": item.task, "condition": condition, "sample_index": k, "label": item.label, "strata": item.strata,
                "proposed_label": p.label if p else None, "proposed_correct": p.correct if p else None,
                "confidence_shown": None}
        rv = models[name]
        tags = {"experiment": EXPERIMENT, "step": "review", "reviewer": alias, "sample": name, "item_id": item.item_id,
                "condition": condition, "sample_index": str(k)}
        req = rv.build_request(item.text, p.label if p else None, task_prefix=WRITTEN_PREFIX if p else ALONE_PREFIX,
                               sample_index=k, tags=tags)
        try:
            resp = rv.client.complete(req)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        r, written = parse_teacher(resp, rv.client.request_key(req), rv.task.labels, p.label if p else None)
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error, "reasoning": written,
                "raw_text": None if r.parse_ok else resp.text[:2000], "request_key": r.request_key,
                **_usage(resp), "ts": now_iso()}
    return work


def n_calls(plan: list[dict], alone: int) -> int:
    from sysone.experiments.e102 import reviewed
    return sum(alone + (not r["jev_correct"]) + (not r["matched_correct"]) + bool(r["in_subsample"]) for r in plan if reviewed(r))


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e22.yaml"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    d = yaml.safe_load(Path(a.config).read_text())
    cfg = _Cfg(d)
    source = RESULTS_DIR / "runs" / cfg.source_run
    run = Run(a.run_id, RESULTS_DIR)
    if not a.report_only:
        full = [r for r in JsonlWriter(source / "plan.jsonl").rows() if r["sample"] in cfg.samples]
        plan = pilot_plan(full, a.max_items)
        if not (run.dir / "plan.jsonl").exists():
            (run.dir / "plan.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in plan))
        if not (run.dir / "jev_decisions.jsonl").exists():
            rows = [r for r in JsonlWriter(source / "jev_decisions.jsonl").rows() if r["sample"] in cfg.samples]
            (run.dir / "jev_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        items: dict[tuple[str, str], Item] = {(s, i.item_id): i for s in cfg.samples for i in load_items(s)[0]}
        registry = Registry(a.run_id, run_provider_caps_usd=cfg.run_provider_caps_usd)
        client = registry.client(cfg.reviewer)
        client.cache = None
        models = {s: Reviewer(client, TASKS[next(i for (n, _), i in items.items() if n == s).task], temperature=None,
                              reasoning_max_tokens=None, max_output_tokens=cfg.max_output_tokens) for s in cfg.samples}
        calls = n_calls(plan, cfg.alone_samples)
        project = registry.project_status()
        print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()
                                                    if v["cap_usd"]))
        print(f"{calls} calls; projected ${calls * ASSUMED_USD_PER_CALL:.2f} (cap ${cfg.run_provider_caps_usd['bedrock']:.2f})")
        run.update_manifest(experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
                            config=d, git=git_info(), source_run=cfg.source_run, project_spend_at_start=project,
                            cost_projection={"calls": calls, "expected_usd": round(calls * ASSUMED_USD_PER_CALL, 2)},
                            prompts={"review_prefix": WRITTEN_PREFIX, "alone_prefix": ALONE_PREFIX})
        writer = run.writer("reviews")
        done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"], r["sample_index"]) for r in writer.rows() if r["status"] == "ok"}
        status = "completed"
        try:
            pending = (t for t in review_tasks(cfg, plan, items, cfg.reviewer) if _key(t) not in done)
            run_parallel(pending, work_fn(run, models), writer, workers=cfg.workers, max_consecutive_errors=cfg.max_consecutive_errors)
        except (BudgetError, _TooManyErrors) as e:
            status = f"halted: {e}"
        finally:
            run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(client.guard.spent_usd, 6))
        print(f"run {a.run_id}: {status}; spent ${client.guard.spent_usd:.4f}")
    rows = [r for r in JsonlWriter(run.dir / "reviews.jsonl").rows() if r["status"] == "ok"]
    print(f"parsed {sum(bool(r['parse_ok']) for r in rows)}/{len(rows)}")
    from sysone.analysis.e22 import analyze_run
    analyze_run(run.dir, source, d)
    print((run.dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
