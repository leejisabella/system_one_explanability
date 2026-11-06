"""E.108 data: consistency training and balanced-proposal SFT, on the same items (research_docs/15).

Both arms show a proposal in every prompt and train toward the gold label; they differ only in the
reasoning used as the target, and every target is gpt-oss-20b's own (never the teacher's):
  consist_review    the reviewer's reasoning from deciding the item alone, with no proposal in view
                    (bias-augmented consistency training): seeing the label shouldn't change the answer.
  balanced_review   the reviewer's own review reasoning that reached the gold label with this
                    proposal in view: half the proposals are wrong, so the proposal alone says nothing.

Steps:
  select   from E.104's mix and E.105's traces, per pool:
             jev_wrong      Jev's mistakes the reviewer solved alone at least once (Jev's wrong label shown)
             matched_wrong  Jev-right items from the mix's alone group, shown the other label,
                            matched_wrong_per_wrong times as many
             jev_right      Jev-right items (Jev's right label shown), as many as both wrong groups
                            together, so half the proposals are wrong
           Every item needs a correct alone trace by the reviewer (arm 1's target). Items it never
           solved alone (group C-like) are left out by construction. Splits are E.104's.
  sample   review traces by the reviewer where E.105 has no correct one: up to deference_samples
           tries for jev_wrong rows (the ones where it gave in every time), review_samples otherwise,
           stopping at the first correct trace. Bedrock; sample indexes start at SAMPLE_OFFSET.
  export   data/posttrain/<dataset>/{consist_review,balanced_review}.{train,val}.jsonl, the same
           rows in both (a row without a trace for either arm is dropped from both), and
           grpo_prompts.{train,val}.jsonl for E.109's pre-pass: every review-format row of E.104's
           mix plus this mix's matched_wrong rows. Manifest with hashes in benchmarks/posttrain/<dataset>/.

Usage:
  uv run python -m sysone.experiments.e108 --dry-run --source-run <e104 dry run> --trace-run <e105 dry run>
  uv run python -m sysone.experiments.e108 --run-id e108-v1-<date>
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
from sysone.cascade.items import Item
from sysone.cascade.proposals import ALONE, MATCHED_LABEL
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, ProviderError
from sysone.clients.fake import FakeChatProvider
from sysone.env import load_env
from sysone.experiments.e101 import _spec_info, _TooManyErrors, _usage, load_samples, run_parallel
from sysone.experiments.e105 import EXPORT_DATA_DIR, EXPORT_MANIFEST_DIR
from sysone.posttrain import formats
from sysone.posttrain.formats import REVIEWER_SOURCE
from sysone.posttrain.mix import ALONE_FORMAT, ALONE_GROUP, JEV_RIGHT, JEV_WRONG, MATCHED_WRONG, REVIEW, TRAIN, VAL, _rng
from sysone.registry import CONFIG_DIR, RESULTS_DIR, Registry
from sysone.runs import JsonlWriter, Run, git_info, now_iso

EXPERIMENT = "e108"
STEPS = ("select", "sample", "export")
ARMS = ("consist_review", "balanced_review")
GRPO_PROMPTS = "grpo_prompts"
# E.105 sampled indexes 0-3; new samples start here, so no request repeats one of E.105's.
SAMPLE_OFFSET = 100


@dataclass(frozen=True)
class E108Config:
    source_run: str                   # the E.104 run: its mix.jsonl
    trace_run: str                    # the E.105 run: its traces.jsonl
    samples: list[str]
    dataset: str = "part2_round2_v1"
    reviewer: str = "reviewer"
    matched_wrong_per_wrong: float = 1 / 3
    deference_samples: int = 8
    review_samples: int = 4
    reviewer_temperature: float | None = None
    reviewer_reasoning_max_tokens: int | None = 4096
    reviewer_max_output_tokens: int = 256
    seed: int = 108
    workers: int = 8
    max_items: int | None = None
    use_cache: bool = False
    max_consecutive_errors: int = 10
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    run_provider_caps_usd: dict[str, float] = field(default_factory=lambda: {"bedrock": 2.0, "typesafe": 0.01})

    @classmethod
    def load(cls, path: str | Path) -> E108Config:
        return cls(**yaml.safe_load(Path(path).read_text()))


def _key(row: dict[str, Any]) -> tuple:
    return row["sample"], row["item_id"], row["format"], row["condition"]


def _own(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The reviewer's own trace to train on (never the teacher's)."""
    return formats.choose([t for t in traces if t["source"] == REVIEWER_SOURCE])


def select_rows(mix: list[dict[str, Any]], traces: dict[tuple, list[dict[str, Any]]], labels: dict[str, tuple[str, ...]],
                config: E108Config) -> list[dict[str, Any]]:
    """E.108's mix rows (E.104's row schema), per pool; every row has a correct alone trace by the reviewer."""
    solved_alone = lambda r: _own(traces[(r["sample"], r["item_id"], ALONE_FORMAT, ALONE)]) is not None
    rows: list[dict[str, Any]] = []
    for sample in config.samples:
        pool = [r for r in mix if r["sample"] == sample and solved_alone(r)]
        by_group = {g: sorted((r for r in pool if r["group"] == g), key=lambda r: r["item_id"])
                    for g in (JEV_WRONG, JEV_RIGHT, ALONE_GROUP)}
        wrong = by_group[JEV_WRONG]
        n_matched = round(len(wrong) * config.matched_wrong_per_wrong)
        candidates = by_group[ALONE_GROUP][:]
        _rng(config.seed, sample, MATCHED_WRONG).shuffle(candidates)
        if len(candidates) < n_matched:
            raise ValueError(f"{sample}: {len(candidates)} alone-group items for {n_matched} matched_wrong rows")
        other = lambda lab: next(x for x in labels[sample] if x != lab)
        matched = [{**r, "group": MATCHED_WRONG, "format": REVIEW, "condition": MATCHED_LABEL,
                    "proposed_label": other(r["label"]), "target_verdict": "override"}
                   for r in sorted(candidates[:n_matched], key=lambda r: r["item_id"])]
        right = by_group[JEV_RIGHT][:]
        _rng(config.seed, sample, JEV_RIGHT).shuffle(right)
        if len(right) < len(wrong) + n_matched:
            raise ValueError(f"{sample}: {len(right)} jev_right items for {len(wrong) + n_matched} wrong proposals")
        rows += wrong + matched + sorted(right[:len(wrong) + n_matched], key=lambda r: r["item_id"])
    return sorted(rows, key=lambda r: (r["sample"], r["group"], r["item_id"]))


def needs_review_sample(row: dict[str, Any], have: list[dict[str, Any]], config: E108Config) -> bool:
    """Whether the reviewer should sample this row's review again: no correct trace of its own yet,
    and tries left (counting E.105's)."""
    mine = [t for t in have if t["source"] == REVIEWER_SOURCE and t["status"] == "ok"]
    cap = config.deference_samples if row["group"] == JEV_WRONG else config.review_samples
    new = [t for t in mine if t["sample_index"] >= SAMPLE_OFFSET]
    return _own(mine) is None and len(new) < cap


def _grpo_record(row: dict[str, Any], text: str) -> dict[str, Any]:
    return {"prompt": formats.prompt_messages(row["task"], text, row["proposed_label"]),
            "meta": {"sample": row["sample"], "item_id": row["item_id"], "task": row["task"], "group": row["group"],
                     "proposed_label": row["proposed_label"], "label": row["target_label"], "split": row["split"]}}


def export_round2(name: str, rows: list[dict[str, Any]], mix: list[dict[str, Any]], traces: dict[tuple, list[dict[str, Any]]],
                  items: dict[tuple[str, str], Item], *, source: dict[str, Any], data_dir: Path = EXPORT_DATA_DIR,
                  manifest_dir: Path = EXPORT_MANIFEST_DIR) -> dict[str, Any]:
    """Write both arms, the GRPO prompts and the manifest. A re-export must reproduce the committed hashes."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {a: {TRAIN: [], VAL: []} for a in (*ARMS, GRPO_PROMPTS)}
    dropped: Counter = Counter()
    for row in rows:
        text = items[(row["sample"], row["item_id"])].text
        alone = _own(traces[(row["sample"], row["item_id"], ALONE_FORMAT, ALONE)])
        review = _own(traces[_key(row)])
        if alone is None or review is None:
            dropped[f"{row['sample']}/{row['group']}/no_trace_{'consist' if alone is None else 'balanced'}_arm"] += 1
            continue
        out["consist_review"][row["split"]].append(formats.sft_record(row, text, row["proposed_label"], alone, arm="consist"))
        out["balanced_review"][row["split"]].append(formats.sft_record(row, text, row["proposed_label"], review, arm="balanced"))
    pools = {r["sample"] for r in rows}
    prompts = [r for r in mix if r["sample"] in pools and r["format"] == REVIEW] + [r for r in rows if r["group"] == MATCHED_WRONG]
    for row in sorted(prompts, key=lambda r: (r["sample"], r["group"], r["item_id"])):
        out[GRPO_PROMPTS][row["split"]].append(_grpo_record(row, items[(row["sample"], row["item_id"])].text))

    bodies = {f"{a}.{s}.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs)
              for a, splits in out.items() for s, recs in splits.items()}
    files = {f: {"n": len(out[f.split(".")[0]][f.split(".")[1]]), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                 "groups": dict(sorted(Counter(r["meta"]["group"] for r in out[f.split(".")[0]][f.split(".")[1]]).items()))}
             for f, body in bodies.items()}
    manifest_path = manifest_dir / name / "manifest.json"
    # Checked before anything is written, so a refused export leaves the existing files intact.
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: v["sha256"] for k, v in old["files"].items()} != {k: v["sha256"] for k, v in files.items()}:
            raise ValueError(f"{manifest_path} exists with different data; export under a new dataset name")
    folder = data_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    for f, body in bodies.items():
        (folder / f).write_text(body)
    trace_sources = Counter(f"{r['meta']['sample']}/{r['meta']['group']}/index_{'new' if r['meta']['trace_sample_index'] >= SAMPLE_OFFSET else 'e105'}"
                            for r in out["balanced_review"][TRAIN] + out["balanced_review"][VAL])
    manifest = {"name": name, "pools": sorted(pools), "source": source, "files": files, "dropped": dict(sorted(dropped.items())),
                "balanced_review_traces": dict(sorted(trace_sources.items())),
                "format": "TRL conversational prompt-completion (both arms); prompt-only with meta (grpo_prompts); "
                          "see sysone.posttrain.formats and research_docs/15"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_e108(config: E108Config, run: Run, reviewer: GuardedClient, *, samples: dict[str, list[Item]],
             source_dir: Path, trace_dir: Path, data_dir: Path = EXPORT_DATA_DIR,
             manifest_dir: Path = EXPORT_MANIFEST_DIR) -> str:
    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    if not config.use_cache:
        reviewer.cache = None
    mix = JsonlWriter(source_dir / "mix.jsonl").rows()
    old = [t for t in JsonlWriter(trace_dir / "traces.jsonl").rows() if t["status"] == "ok"]
    if not mix or not old:
        raise ValueError(f"E.108 needs a completed E.104 run ({source_dir}) and E.105 run ({trace_dir})")
    items = {(name, i.item_id): i for name, its in samples.items() for i in its}
    mix = [r for r in mix if r["sample"] in config.samples and (r["sample"], r["item_id"]) in items]
    guard = reviewer.guard
    run.update_manifest(
        experiment=EXPERIMENT, status="running", started_at=run.manifest().get("started_at", now_iso()),
        source_runs={"e104": source_dir.name, "e105": trace_dir.name}, config=dataclasses.asdict(config),
        models={config.reviewer: _spec_info(reviewer.spec)}, git=git_info(),
        budget={"per_call_cap_usd": guard.per_call_cap_usd, "run_cap_usd": guard.run_cap_usd,
                "run_provider_caps_usd": guard.run_provider_caps_usd},
    )
    reviewers = {name: Reviewer(reviewer, TASKS[its[0].task], temperature=config.reviewer_temperature,
                                reasoning_max_tokens=config.reviewer_reasoning_max_tokens,
                                max_output_tokens=config.reviewer_max_output_tokens) for name, its in samples.items()}
    rows_writer, writer = run.writer("rows"), run.writer("traces")

    def all_traces() -> dict[tuple, list[dict[str, Any]]]:
        out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for t in old + [t for t in writer.rows() if t["status"] == "ok"]:
            out[_key(t)].append(t)
        return out

    def work(task: tuple) -> dict[str, Any]:
        row, k = task
        item = items[(row["sample"], row["item_id"])]
        base = {"run_id": run.run_id, "source": REVIEWER_SOURCE, "sample": row["sample"], "item_id": row["item_id"],
                "task": item.task, "format": REVIEW, "condition": row["condition"], "group": row["group"],
                "sample_index": k, "label": item.label, "proposed_label": row["proposed_label"]}
        tags = {"experiment": EXPERIMENT, "step": "review", "sample": row["sample"], "item_id": row["item_id"],
                "condition": row["condition"], "sample_index": str(k)}
        try:
            r = reviewers[row["sample"]].decide(item.text, row["proposed_label"], sample_index=k, tags=tags)
        except ProviderError as e:
            return {**base, "status": "error", "error": str(e), "ts": now_iso()}
        return {**base, "status": "ok", "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent,
                "parse_ok": r.parse_ok, "parse_error": r.parse_error,
                "raw_text": None if r.parse_ok else r.response.text[:2000], **_usage(r.response), "ts": now_iso()}

    status = "completed"
    try:
        rows = rows_writer.rows()
        if "select" in config.steps and not rows:
            labels = {name: TASKS[its[0].task].labels for name, its in samples.items()}
            rows = select_rows(mix, all_traces(), labels, config)
            for r in rows:
                rows_writer.append(r)
            run.update_manifest(row_counts={s: dict(sorted(Counter(r["group"] for r in rows if r["sample"] == s).items()))
                                            for s in config.samples})
        if "sample" in config.steps:
            for k in range(SAMPLE_OFFSET, SAMPLE_OFFSET + max(config.deference_samples, config.review_samples)):
                have = all_traces()
                done = {(_key(t), t["sample_index"]) for ts in have.values() for t in ts}
                pending = [(r, k) for r in rows if (_key(r), k) not in done and needs_review_sample(r, have[_key(r)], config)]
                run_parallel(pending, work, writer, workers=config.workers, max_consecutive_errors=config.max_consecutive_errors)
        if "export" in config.steps:
            m = export_round2(config.dataset, rows, mix, all_traces(), items,
                              source={"e104_run": source_dir.name, "e105_run": trace_dir.name, "e108_run": run.run_id},
                              data_dir=data_dir, manifest_dir=manifest_dir)
            run.update_manifest(exported={f: v["n"] for f, v in m["files"].items()}, dropped=m["dropped"])
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
                            spent_by_provider={reviewer.spec.provider: round(guard.run_spent_usd(reviewer.spec.provider), 6)})
    return status


def report(run: Run) -> str:
    """Rows per pool and group, the new review samples and how many reached the gold label, and the export."""
    m = run.manifest()
    rows = run.writer("rows").rows()
    new = [t for t in run.writer("traces").rows() if t["status"] == "ok"]
    group = {(t["sample"], t["item_id"], t["condition"]): t["group"] for t in new}
    tried = Counter((k[0], g) for k, g in group.items())
    got = Counter((k[0], group[k]) for k in {(t["sample"], t["item_id"], t["condition"]) for t in new
                                             if formats.correct(t) and formats.usable(t)})
    lines = [f"# E.108 data report: {run.run_id}", "", f"Status: {m.get('status')}. Spent ${m.get('spent_usd', 0):.4f}. "
             f"{len(new)} new review samples.", "", "| Pool | Group | Rows | Rows sampled again | Of those, now solved |",
             "|---|---|---|---|---|"]
    for (s, g), n in sorted(Counter((r["sample"], r["group"]) for r in rows).items()):
        lines.append(f"| {s} | {g} | {n} | {tried[(s, g)]} | {got[(s, g)]} |")
    if m.get("exported"):
        lines += ["", "Exported: " + ", ".join(f"{f} {n}" for f, n in m["exported"].items()),
                  "", "Dropped from both arms: " + (", ".join(f"{k} {v}" for k, v in m.get("dropped", {}).items()) or "none")]
    return "\n".join(lines) + "\n"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DIR / "experiments" / "e108.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--source-run", help="override the config's E.104 run")
    parser.add_argument("--trace-run", help="override the config's E.105 run")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dry-run", action="store_true", help="fake reviewer: no network, no cost")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    config = E108Config.load(args.config)
    for name in ("source_run", "trace_run", "steps"):
        if getattr(args, name):
            config = dataclasses.replace(config, **{name: getattr(args, name)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{EXPERIMENT}-{'dryrun-' if args.dry_run else ''}{stamp}"
    samples, _ = load_samples(config)
    runs = Path(args.results_dir) / "runs"
    registry = Registry(run_id, results_dir=Path(args.results_dir), run_provider_caps_usd=config.run_provider_caps_usd)
    export_dirs = {}
    if args.dry_run:
        reviewer = GuardedClient(ModelSpec(config.reviewer, "fake", f"fake-{config.reviewer}", Price(0.0, 0.0)),
                                 FakeChatProvider(), registry.guard, cache=None)
        # A dry run's datasets are random; keep them out of benchmarks/ and data/.
        export_dirs = {"data_dir": runs / run_id / "export" / "data", "manifest_dir": runs / run_id / "export" / "manifests"}
    else:
        try:
            reviewer = registry.client(config.reviewer)
        except (ModelNotConfigured, RuntimeError) as e:
            raise SystemExit(f"{e}\nUse --dry-run to test the pipeline without keys.") from None
    run = Run(run_id, args.results_dir)
    project = registry.project_status()
    run.update_manifest(project_spend_at_start=project)
    print("project spend so far: " + ", ".join(f"{p} ${v['spent_usd']:.4f} of ${v['cap_usd']:.2f}" for p, v in project.items()))
    status = run_e108(config, run, reviewer, samples=samples, source_dir=runs / config.source_run,
                      trace_dir=runs / config.trace_run, **export_dirs)
    text = report(run)
    (run.dir / "report.md").write_text(text)
    print(f"run {run_id}: {status}\nresults: {run.dir}\n{text}")


if __name__ == "__main__":
    main()
