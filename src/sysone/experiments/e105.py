"""E.105: reasoning traces for the training mix, then the training datasets (research_docs/14).

gpt-oss reasons before it answers, and training on bare labels would teach it to stop. So every
example gets a reasoning trace that reaches the gold label, sampled from the reviewer itself
(rejection sampling); the teacher (gpt-oss-120b) writes traces only for examples the reviewer
never solves.

  traces   in rounds: the reviewer (Bedrock gpt-oss-20b, as in Part 1) samples each example until
           it has a correct trace and either a wrong one (for a DPO pair) or min_samples tries, up
           to max_samples. Examples are every review-format row of E.104's mix, and every mix item
           in alone format (the plain-format control arm).
  teacher  the teacher samples examples with no correct reviewer trace, up to teacher_samples.
  export   writes each dataset in `datasets` (a name and its training pools):
             data/posttrain/<name>/sft_review.{train,val}.jsonl   treatment arm: the mix as built
             data/posttrain/<name>/sft_plain.{train,val}.jsonl    control arm: the same items and gold
                                                                  labels, all in alone format
             data/posttrain/<name>/dpo_review.{train,val}.jsonl   correct vs. wrong reviewer traces
             benchmarks/posttrain/<name>/manifest.json            counts, hashes, sources (committed)
           An item enters the SFT files only if both arms have a trace for it, so the two arms
           train on exactly the same items and labels.

Usage:
  uv run python -m sysone.experiments.e105 --dry-run --source-run <e104 dry-run id>
  uv run python -m sysone.experiments.e105 --run-id <id>      # source_run from the config
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sysone.budget.guard import BudgetError
from sysone.budget.pricing import Price
from sysone.cascade.items import DATA_DIR, Item
from sysone.cascade.proposals import ALONE
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, _usage, load_samples, run_parallel
from sysone.posttrain import formats
from sysone.posttrain.formats import REVIEWER_SOURCE, TEACHER_SOURCE
from sysone.posttrain.mix import ALONE_FORMAT, REVIEW, TRAIN, VAL
from sysone.registry import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e105"
STEPS = ("traces", "teacher", "export")
EXPORT_DATA_DIR = DATA_DIR / "posttrain"
EXPORT_MANIFEST_DIR = REPO_ROOT / "benchmarks" / "posttrain"
ARMS = ("sft_review", "sft_plain", "dpo_review")


@dataclass(frozen=True)
class E105Config:
    source_run: str
    samples: list[str]
    # Each exported dataset: {"name": ..., "samples": [training pools]}. E.108's held-out-domain
    # dataset leaves a pool out.
    datasets: list[dict[str, Any]] = field(default_factory=list)
    reviewer: str = "reviewer"
    teacher: str = "teacher"
    min_samples: int = 2
    max_samples: int = 4
    teacher_samples: int = 2
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    # Only the first n mix rows of each pool (by item ID), for a pilot; None for all.
    max_rows_per_sample: int | None = None
    workers: int = 8
    max_items: int | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 10.0, "typesafe": 0.01})

    @classmethod
    def load(cls, path: str | Path) -> E105Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def targets(mix: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str | None]]:
    """(sample, item_id, format, condition, proposed label) for every example to trace: each
    review-format mix row, and each mix item once in alone format."""
    out = {(r["sample"], r["item_id"], ALONE_FORMAT, ALONE, None) for r in mix}
    out |= {(r["sample"], r["item_id"], REVIEW, r["condition"], r["proposed_label"]) for r in mix if r["format"] == REVIEW}
    return sorted(out, key=lambda t: (t[0], t[1], t[2], t[3]))


def needs_more(traces: list[dict[str, Any]], *, min_samples: int, max_samples: int) -> bool:
    """Whether the reviewer should sample an example again."""
    mine = [t for t in traces if t["source"] == REVIEWER_SOURCE and t["status"] == "ok"]
    if len(mine) >= max_samples:
        return False
    has_correct = any(formats.correct(t) and formats.usable(t) for t in mine)
    has_wrong = any(not formats.correct(t) and formats.usable(t) for t in mine)
    return not (has_correct and (has_wrong or len(mine) >= min_samples))


def _target_key(row: dict[str, Any]) -> tuple:
    return row["sample"], row["item_id"], row["format"], row["condition"]


def _select_mix(mix: list[dict[str, Any]], config: E105Config) -> list[dict[str, Any]]:
    rows = [r for r in mix if r["sample"] in config.samples]
    if config.max_rows_per_sample is None:
        return rows
    keep: list[dict[str, Any]] = []
    for s in config.samples:
        keep += sorted((r for r in rows if r["sample"] == s), key=lambda r: r["item_id"])[: config.max_rows_per_sample]
    return keep


def export_dataset(name: str, pools: list[str], mix: list[dict[str, Any]], traces: list[dict[str, Any]],
                   items: dict[tuple[str, str], Item], *, source: dict[str, Any], data_dir: Path = EXPORT_DATA_DIR,
                   manifest_dir: Path = EXPORT_MANIFEST_DIR) -> dict[str, Any]:
    """Write one dataset's files and manifest. A re-export must reproduce the committed manifest's hashes."""
    by_target: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_target[_target_key(t)].append(t)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {TRAIN: [], VAL: []} for arm in ARMS}
    dropped: Counter = Counter()
    sources: Counter = Counter()
    for row in sorted((r for r in mix if r["sample"] in pools), key=lambda r: (r["sample"], r["item_id"], r["group"])):
        text = items[(row["sample"], row["item_id"])].text
        own = formats.choose(by_target[(row["sample"], row["item_id"], row["format"], row["condition"])])
        plain = formats.choose(by_target[(row["sample"], row["item_id"], ALONE_FORMAT, ALONE)])
        if own is None or plain is None:
            dropped[f"{row['sample']}/{row['group']}/no_trace_{'review_arm' if own is None else 'plain_arm'}"] += 1
        else:
            out["sft_review"][row["split"]].append(formats.sft_record(row, text, row["proposed_label"], own, arm="review"))
            out["sft_plain"][row["split"]].append(formats.sft_record(row, text, None, plain, arm="plain"))
            sources[f"{row['sample']}/{row['group']}/{own['source']}"] += 1
        if row["format"] == REVIEW and own is not None:
            bad = formats.rejected(by_target[(row["sample"], row["item_id"], row["format"], row["condition"])])
            good = formats.choose([t for t in by_target[(row["sample"], row["item_id"], row["format"], row["condition"])]
                                   if t["source"] == REVIEWER_SOURCE])
            if bad is not None and good is not None:
                out["dpo_review"][row["split"]].append(formats.dpo_record(row, text, good, bad))

    bodies = {f"{arm}.{split}.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
              for arm, splits in out.items() for split, records in splits.items()}
    files = {fname: {"n": len(out[fname.split(".")[0]][fname.split(".")[1]]),
                     "sha256": hashlib.sha256(body.encode()).hexdigest(),
                     "groups": dict(sorted(Counter(r["meta"]["group"] for r in
                                                   out[fname.split(".")[0]][fname.split(".")[1]]).items()))}
             for fname, body in bodies.items()}
    manifest_path = manifest_dir / name / "manifest.json"
    # Checked before anything is written, so a refused export leaves the existing files intact.
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: v["sha256"] for k, v in old["files"].items()} != {k: v["sha256"] for k, v in files.items()}:
            raise ValueError(f"{manifest_path} exists with different data; export under a new dataset name")
    folder = data_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    for fname, body in bodies.items():
        (folder / fname).write_text(body)
    manifest = {"name": name, "pools": pools, "source": source, "files": files,
                "trace_sources": dict(sorted(sources.items())), "dropped": dict(sorted(dropped.items())),
                "format": "TRL conversational prompt-completion (SFT) and preference (DPO); see sysone.posttrain.formats"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_e105(config: E105Config, run: Run, reviewer: GuardedClient, teacher: GuardedClient, *,
             samples: dict[str, list[Item]], source_dir: Path, data_dir: Path = EXPORT_DATA_DIR,
             manifest_dir: Path = EXPORT_MANIFEST_DIR) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not config.use_cache:
        reviewer.cache = None
        teacher.cache = None
    mix = JsonlWriter(source_dir / "mix.jsonl").rows()
    if not mix:
        raise ValueError(f"{source_dir} has no mix.jsonl; E.105 needs a completed E.104 run")
    mix = _select_mix(mix, config)
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    guard = reviewer.guard
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        source_run={"run_id": source_dir.name, **{k: v for k, v in Run(source_dir.name, source_dir.parent.parent)
                                                    .manifest().items() if k in ("git", "status", "mix_counts")}},
        config=dataclasses.asdict(config),
        models={config.reviewer: _spec_info(reviewer.spec), config.teacher: _spec_info(teacher.spec)},
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
        git=git_info(), config_files={p.name: p.read_text() for p in sorted(CONFIG_DIR.glob("*.yaml"))},
    )
    reviewers = {(src, name): Reviewer(c, TASKS[its[0].task], temperature=config.reviewer_temperature,
                                       reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                       max_output_tokens=config.reviewer_max_output_tokens)
                 for src, c in ((REVIEWER_SOURCE, reviewer), (TEACHER_SOURCE, teacher)) for name, its in samples.items()}
    group_of = {(r["sample"], r["item_id"]): r["group"] for r in mix}
    writer = run.writer("traces")

    def work(t: tuple) -> dict[str, Any]:
        (sample, item_id, fmt, condition, proposed), source, k = t
        item = items[(sample, item_id)]
        base = {"run_id": run.run_id, "source": source, "sample": sample, "item_id": item_id, "task": item.task,
                "format": fmt, "condition": condition, "group": group_of[(sample, item_id)], "sample_index": k,
                "label": item.label, "proposed_label": proposed}
        tags = {"experiment": EXPERIMENT, "step": "review", "source": source, "sample": sample, "item_id": item_id,
                "format": fmt, "sample_index": str(k)}
        try:
            r = reviewers[(source, sample)].decide(item.text, proposed, sample_index=k, tags=tags)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                "raw_text": None if r.parse_ok else r.response.text[:2000], **_usage(r.response), "ts": now_iso()}

    def traces_by_target() -> dict[tuple, list[dict[str, Any]]]:
        out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in writer.rows():
            out[_target_key(row)].append(row)
        return out

    todo = targets(mix)
    status = "completed"
    try:
        if "traces" in config.steps:
            for k in range(config.max_samples):
                have = traces_by_target()
                done = {(_target_key(r), r["sample_index"]) for rows in have.values() for r in rows
                        if r["source"] == REVIEWER_SOURCE and r["status"] == "ok"}
                pending = [(t, REVIEWER_SOURCE, k) for t in todo if (t[:4], k) not in done
                           and needs_more(have[t[:4]], min_samples=config.min_samples, max_samples=config.max_samples)]
                run_parallel(pending, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
        if "teacher" in config.steps:
            for k in range(config.teacher_samples):
                have = traces_by_target()
                pending = [(t, TEACHER_SOURCE, k) for t in todo if formats.choose(have[t[:4]]) is None
                           and not any(r["source"] == TEACHER_SOURCE and r["sample_index"] == k and r["status"] == "ok"
                                       for r in have[t[:4]])]
                run_parallel(pending, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
        if "export" in config.steps:
            rows = [r for r in writer.rows() if r["status"] == "ok"]
            exported = {}
            for spec in config.datasets:
                m = export_dataset(spec["name"], spec["samples"], mix, rows, items, data_dir=data_dir,
                                   manifest_dir=manifest_dir, source={"e104_run": source_dir.name, "e105_run": run.run_id})
                exported[spec["name"]] = {f: v["n"] for f, v in m["files"].items()}
            run.update_manifest(exported=exported)
    except (BudgetError, _TooManyErrors) as e:
        status = f"halted: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        providers = sorted({reviewer.spec.provider, teacher.spec.provider})
        run.update_manifest(status=status, finished_at=now_iso(), spent_usd=round(guard.spent_usd, 6),
                            spent_by_provider={p: round(guard.run_spent_usd(p), 6) for p in providers})
    return status


def report(run: Run) -> str:
    """Trace yield: per pool, group and format, how many examples the reviewer solved, the teacher
    solved, or neither (the blind spots that rejection sampling can't teach)."""
    rows = [r for r in run.writer("traces").rows() if r["status"] == "ok"]
    by_target: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_target[_target_key(r)].append(r)
    table: dict[tuple, Counter] = defaultdict(Counter)
    for key, ts in by_target.items():
        best = formats.choose(ts)
        table[(key[0], ts[0]["group"], key[2])]["reviewer" if best and best["source"] == REVIEWER_SOURCE
                                                 else "teacher" if best else "unsolved"] += 1
    m = run.manifest()
    lines = [f"# E.105 report: {run.run_id}", "", f"Status: {m.get('status')}. Spent ${m.get('spent_usd', 0):.4f} "
             f"({m.get('spent_by_provider', {})}). {len(rows)} traces.", "",
             "| Pool | Group | Format | Examples | Solved by the reviewer | By the teacher | Unsolved |", "|---|---|---|---|---|---|---|"]
    for (s, g, f), c in sorted(table.items()):
        n = sum(c.values())
        lines.append(f"| {s} | {g} | {f} | {n} | {c['reviewer']} | {c['teacher']} | {c['unsolved']} |")
    for name, files in (m.get("exported") or {}).items():
        lines += ["", f"Exported `{name}`: " + ", ".join(f"{f} {n}" for f, n in files.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e105.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--source-run", help="override the config's E.104 run")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dry-run", action="store_true", help="fake models: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E105Config.load(args.config)
    if args.source_run:
        config = dataclasses.replace(config, source_run=args.source_run)
    if args.steps:
        config = dataclasses.replace(config, steps=args.steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, _ = load_samples(config)
    source_dir = Path(args.results_dir) / "runs" / config.source_run

    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    export_dirs = {}
    if args.dry_run:
        fake = lambda a: GuardedClient(ModelSpec(a, "fake", f"fake-{a}", Price(0.0, 0.0)), FakeChatProvider(),
                                       registry.guard, cache=None)
        reviewer, teacher = fake(config.reviewer), fake(config.teacher)
        # A dry run's datasets are random; keep them out of benchmarks/ and data/.
        export_dirs = {"data_dir": Path(args.results_dir) / "runs" / run_id / "export" / "data",
                       "manifest_dir": Path(args.results_dir) / "runs" / run_id / "export" / "manifests"}
    else:
        try:
            reviewer, teacher = registry.client(config.reviewer), registry.client(config.teacher)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None

    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e105(config, run, reviewer, teacher, samples=samples, source_dir=source_dir, **export_dirs)
    text = report(run)
    (run.dir / "report.md").write_text(text)
    print(f"run {run_id}: {status}\nresults: {run.dir}\n{text}")


if __name__ == "__main__":
    main()
