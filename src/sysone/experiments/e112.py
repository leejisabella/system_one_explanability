"""E.112: teacher screen for distillation (research_docs/15, revised 2026-11-21).

Does any affordable teacher catch the Jev mistakes gpt-oss-20b misses? Each candidate teacher
reviews a fixed sample of training-pool items once, in the reviewer's own review format (item,
policy, Jev's label), without being told the gold label. Training data only: the teacher is never a
stage in the cascade.

The sample, per training pool, drawn by hash from E.104's mix:
  unsolved  Jev's mistakes that no model got right in any E.105 review try (the training-pool
            analogue of group C)
  solved    Jev's other mistakes
  right     items Jev got right
gpt-oss-20b's baseline is free: its first E.105 review try on the same items.

Teachers that don't reason natively (mode "written") are asked to write their reasoning before
the JSON answer; that text is the trace a distillation dataset would train on.

The pass rule is fixed in configs/experiments/e112.yaml before the run; `report` scores it.

Usage:
  uv run python -m sysone.experiments.e112 --dry-run
  uv run python -m sysone.experiments.e112 --run-id e112-v1-<date>
  uv run python -m sysone.experiments.e112 --report-only --run-id <id>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import Item
from sysone.cascade.reviewer import Reviewer, parse_review
from sysone.cascade.tasks import TASKS
from sysone.explain.jsonout import parse_json_object
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, _usage, load_samples, run_parallel
from sysone.posttrain import formats
from sysone.posttrain.formats import REVIEWER_SOURCE
from sysone.posttrain.mix import REVIEW
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e112"
UNSOLVED, SOLVED, RIGHT = "unsolved", "solved", "right"
STRATA = (UNSOLVED, SOLVED, RIGHT)
NATIVE, WRITTEN = "native", "written"
BASELINE = "gpt-oss-20b (E.105, first try)"
WRITTEN_PREFIX = (
    "Think it through before you answer: first write out your reasoning step by step as plain text (not "
    "inside the JSON object and not in a code block), applying the policy to the content and checking the "
    "automated model's decision. Then end your reply with the JSON object described below, and nothing after it."
)
# Tokens per call assumed for the cost projection (E.105 averaged 460 in, 332 out for gpt-oss-20b).
ASSUMED_TOKENS = {NATIVE: (460, 1500), WRITTEN: (460, 600)}


@dataclass(frozen=True)
class E112Config:
    source_run: str            # E.104: the mix (Jev's labels on the training pools)
    traces_run: str            # E.105: gpt-oss-20b's review traces (baseline and the unsolved stratum)
    samples: list[str]
    per_stratum: dict[str, int]
    teachers: dict[str, str]   # alias -> "native" or "written"
    native_reasoning_max_tokens: int = 8192
    native_max_output_tokens: int = 256
    written_max_output_tokens: int = 1500
    seed: str = "e112-v1"
    criteria: dict[str, Any] = field(default_factory=dict)
    workers: int = 8
    max_items: int | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 3.0, "typesafe": 0.01})
    bootstrap: int = 2000

    @classmethod
    def load(cls, path: str | Path) -> E112Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def _rank(seed: str, sample: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample}:{item_id}".encode()).hexdigest()


def strata(mix: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Each review-format mix item's stratum. A Jev mistake is unsolved if no E.105 review try, by
    either model, reached the gold label."""
    solved = {(t["sample"], t["item_id"]) for t in traces
              if t["format"] == REVIEW and t["condition"] == "jev_label" and formats.correct(t)}
    out = {}
    for r in mix:
        if r["format"] != REVIEW:
            continue
        key = (r["sample"], r["item_id"])
        out[key] = RIGHT if r["jev_correct"] else SOLVED if key in solved else UNSOLVED
    return out


def select(mix: list[dict[str, Any]], traces: list[dict[str, Any]], config: E112Config) -> list[dict[str, Any]]:
    """The screen's items, the same for every teacher: per pool and stratum, the first n by hash."""
    st = strata(mix, traces)
    rows = {(r["sample"], r["item_id"]): r for r in mix if r["format"] == REVIEW}
    picked = []
    for sample in config.samples:
        for s in STRATA:
            keys = sorted((k for k, v in st.items() if k[0] == sample and v == s), key=lambda k: _rank(config.seed, *k))
            for k in sorted(keys[: config.per_stratum[s]]):
                r = rows[k]
                picked.append({"sample": k[0], "item_id": k[1], "stratum": s, "label": r["label"],
                               "proposed_label": r["proposed_label"]})
    return picked


def weights(mix: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Jev's mistakes per pool and stratum in the whole mix, to weight the screen back to the pool."""
    w: dict[str, Counter] = defaultdict(Counter)
    for (sample, _), s in strata(mix, traces).items():
        if s != RIGHT:
            w[sample][s] += 1
    return {k: dict(v) for k, v in w.items()}


def parse_teacher(response, key: str, labels: tuple[str, ...], proposal: str) -> tuple[Any, str | None]:
    """The review, tolerating written reasoning before the JSON; and that written reasoning."""
    r = parse_review(response, key, labels, proposal)
    cut = response.text.rfind("{")
    if not r.parse_ok and cut > 0:
        r = parse_review(dataclasses.replace(response, text=response.text[cut:]), key, labels, proposal)
    written = response.text[:cut].strip().removesuffix("```json").removesuffix("```").strip() if cut > 0 else ""
    if not written:  # reasoning put inside the JSON object instead
        obj = parse_json_object(response.text[cut:] if cut > 0 else response.text) or {}
        written = next((v for k, v in obj.items() if k not in ("verdict", "final_label") and isinstance(v, str)), "")
    return r, written or None


def baseline_rows(traces: list[dict[str, Any]], picked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    want = {(p["sample"], p["item_id"]): p for p in picked}
    out = []
    for t in traces:
        key = (t["sample"], t["item_id"])
        if (key in want and t["source"] == REVIEWER_SOURCE and t["format"] == REVIEW and t["condition"] == "jev_label"
                and t["sample_index"] == 0 and t["status"] == "ok"):
            out.append({**want[key], "teacher": BASELINE, "parse_ok": t["parse_ok"], "final_label": t["final_label"],
                        "verdict": t["verdict"], "status": "ok"})
    return out


def run_e112(config: E112Config, run: Run, clients: dict[str, GuardedClient], *, samples: dict[str, list[Item]],
             picked: list[dict[str, Any]]) -> str:
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    guard = next(iter(clients.values())).guard
    if not config.use_cache:
        for c in clients.values():
            c.cache = None
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        config=dataclasses.asdict(config), models={a: _spec_info(c.spec) for a, c in clients.items()},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        selection={"n": len(picked), "counts": dict(Counter(f"{p['sample']}/{p['stratum']}" for p in picked)),
                   "sha256": hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))})
    reviewers = {}
    for alias, c in clients.items():
        native = config.teachers[alias] == NATIVE
        for name, its in samples.items():
            reviewers[(alias, name)] = Reviewer(
                c, TASKS[its[0].task], reasoning_max_tokens=config.native_reasoning_max_tokens if native else None,
                max_output_tokens=config.native_max_output_tokens if native else config.written_max_output_tokens)
    writer = run.writer("reviews")
    done = {(r["teacher"], r["sample"], r["item_id"]) for r in writer.rows() if r["status"] == "ok"}

    def work(t: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        alias, p = t
        item = items[(p["sample"], p["item_id"])]
        base = {"run_id": run.run_id, "teacher": alias, **p, "ts": now_iso()}
        rv = reviewers[(alias, p["sample"])]
        prefix = WRITTEN_PREFIX if config.teachers[alias] == WRITTEN else None
        tags = {"experiment": EXPERIMENT, "step": "review", "teacher": alias, "sample": p["sample"], "item_id": p["item_id"]}
        request = rv.build_request(item.text, p["proposed_label"], task_prefix=prefix, tags=tags)
        try:
            response = rv.client.complete(request)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e)}
        r, written = parse_teacher(response, rv.client.request_key(request), rv.task.labels, p["proposed_label"])
        usage = _usage(response)
        if written and not usage.get("reasoning"):
            usage["reasoning"] = written
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                "raw_text": response.text[:8000], **usage}

    todo = [(a, p) for p in picked for a in clients if (a, p["sample"], p["item_id"]) not in done]
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
        providers = sorted({c.spec.provider for c in clients.values()})
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6) for p in providers})
    return status


def project_cost(config: E112Config, clients: dict[str, GuardedClient], n_items: int) -> dict[str, dict[str, float]]:
    """Expected (assumed tokens) and worst-case (every call at its output cap, standard price) cost per teacher."""
    out = {}
    for alias, c in clients.items():
        mode = config.teachers[alias]
        tin, tout = ASSUMED_TOKENS[mode]
        price = c.spec.tier_price or c.spec.price
        cap = (config.native_reasoning_max_tokens + config.native_max_output_tokens if mode == NATIVE
               else config.written_max_output_tokens)
        out[alias] = {"expected_usd": round(n_items * (tin * price.input_per_m + tout * price.output_per_m) / 1e6, 4),
                      "worst_usd": round(n_items * (tin * c.spec.price.input_per_m + cap * c.spec.price.output_per_m) / 1e6, 4)}
    return out


# ---------------------------------------------------------------------------------------------- report

def _overrides(r: dict[str, Any]) -> bool:
    """Reaches the gold label (on a Jev mistake: overrides it). Unparsed counts as not."""
    return bool(r.get("parse_ok")) and r["final_label"] == r["label"]


def _weighted(rows: list[dict[str, Any]], w: dict[str, dict[str, int]], pools: list[str]) -> float | None:
    """Override rate on Jev's mistakes, each pool's strata weighted to the whole mix, pools by their mistakes."""
    num = den = 0.0
    for pool in pools:
        for s in (UNSOLVED, SOLVED):
            cell = [_overrides(r) for r in rows if r["sample"] == pool and r["stratum"] == s]
            if cell and w.get(pool, {}).get(s):
                num += w[pool][s] * sum(cell) / len(cell)
                den += w[pool][s]
    return num / den if den else None


def _paired_boot(a: list[dict], b: list[dict], w, pools, n: int, seed: str) -> list[float]:
    """Bootstrap the weighted override difference a - b, resampling items within each pool and stratum."""
    rng = random.Random(seed)
    ka = {(r["sample"], r["item_id"]): r for r in a}
    kb = {(r["sample"], r["item_id"]): r for r in b}
    cells = defaultdict(list)
    for k in ka.keys() & kb.keys():
        if ka[k]["stratum"] != RIGHT:
            cells[(k[0], ka[k]["stratum"])].append(k)
    diffs = []
    for _ in range(n):
        ra, rb = [], []
        for keys in cells.values():
            pick = [rng.choice(keys) for _ in keys]
            ra += [ka[k] for k in pick]
            rb += [kb[k] for k in pick]
        x, y = _weighted(ra, w, pools), _weighted(rb, w, pools)
        if x is not None and y is not None:
            diffs.append(x - y)
    diffs.sort()
    return [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if diffs else [float("nan")] * 2


def _rate(xs: list[bool]) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "–"


def analyze(rows: list[dict[str, Any]], base: list[dict[str, Any]], w: dict[str, dict[str, int]],
            config: E112Config) -> dict[str, Any]:
    c = config.criteria
    pools = config.samples
    wg = [p for p in pools if p.startswith("wildguard")]
    out: dict[str, Any] = {}
    b_rate = _weighted(base, w, pools)
    for alias in [*config.teachers, BASELINE]:
        mine = base if alias == BASELINE else [r for r in rows if r["teacher"] == alias and r["status"] == "ok"]
        if not mine:
            continue
        right = [r for r in mine if r["stratum"] == RIGHT]
        res = {"n": len(mine), "parse_rate": sum(bool(r["parse_ok"]) for r in mine) / len(mine),
               "override_weighted": _weighted(mine, w, pools),
               "accept_right": sum(_overrides(r) for r in right) / len(right) if right else None,
               "unsolved_wg": [_overrides(r) for r in mine if r["sample"] in wg and r["stratum"] == UNSOLVED],
               "cells": {f"{p}/{s}": [_overrides(r) for r in mine if r["sample"] == p and r["stratum"] == s]
                         for p in pools for s in STRATA},
               "trace_chars": sorted(len(r.get("reasoning") or "") for r in mine)}
        if alias != BASELINE:
            lo, hi = _paired_boot(mine, base, w, pools, config.bootstrap, config.seed)
            diff = res["override_weighted"] - b_rate
            uw = res["unsolved_wg"]
            res["vs_baseline"] = {"diff": diff, "ci95": [lo, hi]}
            res["pass"] = {
                "override_gain": lo > 0 and diff >= c["min_override_gain"],
                "accept_right": res["accept_right"] is not None and res["accept_right"] >= c["min_accept_right"],
                "unsolved_wg": bool(uw) and sum(uw) / len(uw) >= c["min_unsolved_wg"],
                "parse": res["parse_rate"] >= c["min_parse_rate"],
            }
            res["passes"] = all(res["pass"].values())
        out[alias] = res
    return out


def report(run: Run, config: E112Config, mix, traces) -> str:
    rows = [r for r in run.writer("reviews").rows() if r["status"] == "ok"]
    picked = [{k: r[k] for k in ("sample", "item_id", "stratum", "label", "proposed_label")}
              for r in {(r["sample"], r["item_id"]): r for r in rows}.values()]
    base = baseline_rows(traces, picked)
    w = weights(mix, traces)
    res = analyze(rows, base, w, config)
    m = run.manifest()
    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    lines = [f"# E.112 teacher screen: {run.run_id}", "",
             f"Status: {m.get('status')}. Spent ${m.get('spent_usd', 0):.4f} ({m.get('spent_by_provider', {})}). "
             f"{len(rows)} teacher reviews on {len(picked)} items.", "",
             "Override = reaches the gold label on Jev's mistakes, weighted to the whole training mix "
             f"(Jev's mistakes per pool and stratum: {w}). Accept-when-right = keeps Jev's right labels.", "",
             "| Model | Override, weighted | vs. 20b [95%] | Accept when right | WildGuard unsolved | Parsed | Median trace (chars) | Passes |",
             "|---|---|---|---|---|---|---|---|"]
    for alias, r in res.items():
        vs = r.get("vs_baseline")
        tc = r["trace_chars"]
        lines.append(
            f"| {alias} | {pct(r['override_weighted'])} | "
            + (f"{100 * vs['diff']:+.1f} [{100 * vs['ci95'][0]:+.1f}, {100 * vs['ci95'][1]:+.1f}]" if vs else "–")
            + f" | {pct(r['accept_right'])} | {_rate(r['unsolved_wg'])} | {pct(r['parse_rate'])} | {tc[len(tc) // 2] if tc else 0} | "
            + ("–" if "passes" not in r else ("**yes**" if r["passes"] else "no (" + ", ".join(k for k, v in r["pass"].items() if not v) + ")"))
            + " |")
    lines += ["", "Per pool and stratum (reaches the gold label):", "",
              "| Model | " + " | ".join(f"{p.split('_')[0]}{'-' + p.split('_')[1] if p.startswith('wildguard') else ''} {s}"
                                       for p in config.samples for s in STRATA) + " |",
              "|---|" + "---|" * (len(config.samples) * len(STRATA))]
    for alias, r in res.items():
        lines.append(f"| {alias} | " + " | ".join(_rate(r["cells"][f"{p}/{s}"]) for p in config.samples for s in STRATA) + " |")
    lines += ["", f"Criteria (fixed before the run): {config.criteria}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e112.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="fake teachers: no network, no cost")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E112Config.load(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    runs = Path(args.results_dir) / "runs"
    mix = JsonlWriter(runs / config.source_run / "mix.jsonl").rows()
    traces = [t for t in JsonlWriter(runs / config.traces_run / "traces.jsonl").rows() if t["status"] == "ok"]
    run = Run(run_id, args.results_dir)
    if not args.report_only:
        samples, _ = load_samples(config)
        picked = select(mix, traces, config)
        registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
        if args.dry_run:
            clients = {a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(),
                                        registry.guard, cache=None) for a in config.teachers}
        else:
            try:
                clients = {a: registry.client(a) for a in config.teachers}
            except (ModelNotConfigured, RuntimeError) as e:
                raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None
        project = registry.project_status()
        cost = project_cost(config, clients, len(picked))
        run.update_manifest(project_spend_at_start=project, cost_projection=cost)
        print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
        print(f"{len(picked)} items x {len(clients)} teachers; projected cost: {cost}")
        worst = sum(v["worst_usd"] for v in cost.values())
        cap = config.run_provider_caps_usd.get("bedrock", 0)
        if not args.dry_run and worst > cap:
            raise SystemExit(f"worst case ${worst:.2f} exceeds the run's Bedrock cap ${cap:.2f}; nothing sent")
        status = run_e112(config, run, clients, samples=samples, picked=picked)
        print(f"run {run_id}: {status}")
    text = report(run, config, mix, traces)
    (run.dir / "report.md").write_text(text)
    print(f"results: {run.dir}\n{text}")


if __name__ == "__main__":
    main()
