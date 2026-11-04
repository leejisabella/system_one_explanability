"""Frozen item samples for the cascade experiments, drawn from public moderation datasets.

Each sample has two halves:
  data/cascade/<name>/items.jsonl         the items, with their text (gitignored: WildGuardMix
                                          is gated, so its text is not redistributed)
  benchmarks/cascade/<name>/manifest.json the source revision, sampling settings, every item's
                                          ID, label and strata, and the sha256 of items.jsonl
                                          (committed)
Building is deterministic from the pinned source revision and the seed, so anyone with
access to the sources rebuilds byte-identical items, and loading refuses items whose hash
doesn't match the committed manifest.

Usage:
  uv run --group data python -m sysone.cascade.items build    # download the sources, freeze both samples

  uv run --group data python -m sysone.cascade.items build civil_toxicity_pool_v1   # one sample

Samples:
- wildguard_prompt_v1 (E.101): WildGuardTest (human labels; the GPT-4-labeled training split is
  not used), prompt harmfulness, 75 items in each cell of adversarial x label.
- civil_toxicity_v1 (E.101): Civil Comments with identity annotations (the WILDS release),
  toxicity >= 0.5, 75 items in each cell of mentions-an-identity x label. Only comments from
  Jigsaw's test sets (ID >= 7,000,000) whose text is not in google/civil_comments' train split:
  TypeSafe's "Jev boost" post-training sets include that train split (as of 2026-10-24).
- wildguard_prompt_full_v1 (E.102): every labeled WildGuardTest prompt.
- wildguard_response_full_v1 (E.102): every WildGuardTest response with a harmfulness label,
  shown with its prompt; conversations over 12,000 characters are left out.
- civil_toxicity_pool_v1 (E.102): 750 per cell, as civil_toxicity_v1 but none of its items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sysone.cascade.tasks import CIVIL_TOXICITY, TASKS, WILDGUARD_PROMPT, WILDGUARD_RESPONSE
from sysone.registry import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
MANIFEST_DIR = REPO_ROOT / "benchmarks" / "cascade"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

# Pinned source revisions (read 2026-11-03).
WILDGUARD_SOURCE = {"repo": "allenai/wildguardmix", "revision": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
                    "path": "test/wildguard_test.parquet", "license": "ODC-BY (gated)"}
CIVIL_SOURCE = {"repo": "pietrolesci/civilcomments-wilds", "revision": "c227534cc0a34cf21db6a0bf0edd0f9050c8b305",
                "path": "raw/train-00000-of-00001.parquet", "license": "CC0 (Civil Comments)"}
# The revision the Jev-boost set names: clm-project/jev-boost-kn-civil-comments-sata-*.
CIVIL_TRAIN_SOURCES = [
    {"repo": "google/civil_comments", "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
     "path": f"data/train-0000{i}-of-00002.parquet"} for i in range(2)
]
# Jigsaw Unintended Bias comment IDs: train below 7,000,000, the public and private test sets above.
JIGSAW_TEST_MIN_ID = 7_000_000
CIVIL_TOXIC_THRESHOLD = 0.5
# Annotator agreement, the difficulty slice: low when the label is close to a coin flip.
# WildGuardTest has three annotators per prompt; agreement 2 of 3 is low. Civil Comments
# toxicity is the fraction of raters (usually 4 to 10) who called the comment toxic.
CIVIL_LOW_AGREEMENT = (0.3, 0.7)
CIVIL_IDENTITY_GROUPS = ("male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white")


@dataclass(frozen=True)
class Item:
    item_id: str
    dataset: str
    # The task name in sysone.cascade.tasks.
    task: str
    text: str
    # Ground truth from human annotators.
    label: str
    # Categorical slices the sample is stratified on and analyzed by.
    strata: dict[str, str]
    # Everything else kept from the source: soft labels, annotator agreement, subcategory.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Item:
        return Item(**d)


def stratified_sample(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], tuple], n_per_cell: int,
                      seed: int, *, row_id: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    """n_per_cell rows from each cell of `key`, drawn without replacement. A cell with fewer
    rows contributes all of them. Cells and the rows within them are sorted (by `row_id`)
    before drawing, so the draw depends only on the seed and the rows, not on their order."""
    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[key(row)].append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for cell in sorted(cells):
        pool = sorted(cells[cell], key=row_id)
        out += pool if len(pool) <= n_per_cell else rng.sample(pool, n_per_cell)
    return out


def wildguard_items(rows: Iterable[dict[str, Any]], *, n_per_cell: int | None, seed: int,
                    max_chars: int) -> tuple[list[Item], dict]:
    """WildGuardTest rows (in file order) to prompt-harmfulness items; n_per_cell None takes all."""
    pool, dropped = [], Counter()
    for i, r in enumerate(rows):
        if r.get("prompt_harm_label") not in WILDGUARD_PROMPT.labels:
            dropped["no_prompt_label"] += 1
        elif not r.get("prompt") or len(r["prompt"]) > max_chars:
            dropped["too_long_or_empty"] += 1
        else:
            pool.append({**r, "_index": i})
    picked = pool if n_per_cell is None else stratified_sample(
        pool, lambda r: (bool(r["adversarial"]), r["prompt_harm_label"]), n_per_cell, seed, row_id=lambda r: r["_index"])
    items = [Item(
        item_id=f"wgt-{r['_index']:04d}",
        dataset="wildguard_prompt",
        task=WILDGUARD_PROMPT.name,
        text=r["prompt"],
        label=r["prompt_harm_label"],
        strata={"adversarial": "adversarial" if r["adversarial"] else "vanilla",
                "annotator_agreement": "low" if (r.get("prompt_harm_agreement") or 3) < 3 else "high"},
        meta={"source_index": r["_index"], "subcategory": r.get("subcategory"),
              "prompt_harm_agreement": r.get("prompt_harm_agreement")},
    ) for r in picked]
    return items, {"pool": len(pool), "dropped": dict(dropped)}


def wildguard_response_items(rows: Iterable[dict[str, Any]], *, max_chars: int) -> tuple[list[Item], dict]:
    """WildGuardTest rows (in file order) to response-harmfulness items: the conversation is the
    prompt and the response, and the label is the response's."""
    items, dropped = [], Counter()
    for i, r in enumerate(rows):
        if r.get("response_harm_label") not in WILDGUARD_RESPONSE.labels or not r.get("response"):
            dropped["no_response_label"] += 1
            continue
        text = f"User prompt:\n{r['prompt']}\n\nAI assistant's response:\n{r['response']}"
        if len(text) > max_chars:
            dropped["too_long"] += 1
            continue
        items.append(Item(
            item_id=f"wgr-{i:04d}",
            dataset="wildguard_response",
            task=WILDGUARD_RESPONSE.name,
            text=text,
            label=r["response_harm_label"],
            strata={"adversarial": "adversarial" if r["adversarial"] else "vanilla",
                    "annotator_agreement": "low" if (r.get("response_harm_agreement") or 3) < 3 else "high",
                    "response": r.get("response_refusal_label") or "unknown"},
            meta={"source_index": i, "subcategory": r.get("subcategory"), "prompt_harm_label": r.get("prompt_harm_label"),
                  "response_harm_agreement": r.get("response_harm_agreement")},
        ))
    return items, {"pool": len(items), "dropped": dict(dropped)}


def civil_items(rows: Iterable[dict[str, Any]], train_texts: set[str], *, n_per_cell: int, seed: int,
                max_chars: int, exclude_ids: frozenset[int] = frozenset()) -> tuple[list[Item], dict]:
    """Identity-annotated Civil Comments rows to toxicity items, keeping only comments that
    never appeared in google/civil_comments' train split (nor in exclude_ids, e.g. an earlier sample)."""
    pool, dropped = [], Counter()
    for r in rows:
        if r["id"] in exclude_ids:
            dropped["in_earlier_sample"] += 1
        elif r["id"] < JIGSAW_TEST_MIN_ID:
            dropped["jigsaw_train_id"] += 1
        elif r["comment_text"] in train_texts:
            dropped["text_in_hf_train"] += 1
        elif not r["comment_text"].strip() or len(r["comment_text"]) > max_chars:
            dropped["too_long_or_empty"] += 1
        else:
            pool.append(r)
    label = lambda r: CIVIL_TOXICITY.labels[0] if r["toxicity"] >= CIVIL_TOXIC_THRESHOLD else CIVIL_TOXICITY.labels[1]
    picked = stratified_sample(pool, lambda r: (bool(r["identity_any"]), label(r)), n_per_cell, seed,
                               row_id=lambda r: r["id"])
    items = [Item(
        item_id=f"cc-{r['id']}",
        dataset="civil_toxicity",
        task=CIVIL_TOXICITY.name,
        text=r["comment_text"],
        label=label(r),
        strata={"identity": "identity" if r["identity_any"] else "no_identity",
                "annotator_agreement": "low" if CIVIL_LOW_AGREEMENT[0] <= r["toxicity"] < CIVIL_LOW_AGREEMENT[1] else "high"},
        meta={"jigsaw_id": r["id"], "toxicity": r["toxicity"], "identity_attack": r["identity_attack"],
              "toxicity_annotator_count": r["toxicity_annotator_count"],
              "identity_groups": [g for g in CIVIL_IDENTITY_GROUPS if r.get(g)]},
    ) for r in picked]
    return items, {"pool": len(pool), "dropped": dict(dropped)}


def _body(items: list[Item]) -> str:
    return "".join(json.dumps(i.to_dict(), sort_keys=True) + "\n" for i in sorted(items, key=lambda i: i.item_id))


def freeze_items(items: list[Item], name: str, meta: dict[str, Any], *, data_dir: Path = DATA_DIR,
                 manifest_dir: Path = MANIFEST_DIR, force: bool = False) -> Path:
    items_path = data_dir / "cascade" / name / "items.jsonl"
    manifest_path = manifest_dir / name / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} exists; frozen samples are not overwritten (use a new name)")
    body = _body(items)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(body)
    manifest = {
        **meta,
        "name": name,
        "n_items": len(items),
        "label_counts": dict(sorted(Counter(i.label for i in items).items())),
        "strata_counts": {k: dict(sorted(Counter(i.strata[k] for i in items).items())) for k in sorted(items[0].strata)},
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "items": [{"item_id": i.item_id, "label": i.label, "strata": i.strata} for i in sorted(items, key=lambda i: i.item_id)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_items(name: str, *, data_dir: Path = DATA_DIR, manifest_dir: Path = MANIFEST_DIR) -> tuple[list[Item], dict]:
    """A frozen sample, refused if its items no longer match the committed manifest."""
    manifest_path = manifest_dir / name / "manifest.json"
    items_path = data_dir / "cascade" / name / "items.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if not items_path.exists():
        raise FileNotFoundError(f"{items_path} is missing; rebuild it with "
                                "`uv run --group data python -m sysone.cascade.items build`")
    body = items_path.read_text()
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"{items_path}: hash {digest} does not match {manifest_path} ({manifest['sha256']})")
    items = [Item.from_dict(json.loads(line)) for line in body.splitlines() if line.strip()]
    unknown = {i.task for i in items} - set(TASKS)
    if unknown:
        raise ValueError(f"{name}: unknown tasks {sorted(unknown)}")
    return items, manifest


def download(source: dict[str, str], *, data_dir: Path = DATA_DIR) -> Path:
    """One file from the Hugging Face Hub at a pinned revision, into its own folder under data/raw."""
    import httpx

    out = data_dir / "raw" / source["repo"].replace("/", "__") / source["revision"][:12] / source["path"]
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    tmp = out.with_suffix(out.suffix + ".part")
    with httpx.stream("GET", HF_URL.format(**source), headers=headers, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(out)
    return out


def _parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def _civil_rows() -> tuple[list[dict[str, Any]], set[str]]:
    train_texts: set[str] = set()
    for source in CIVIL_TRAIN_SOURCES:
        train_texts.update(r["text"] for r in _parquet_rows(download(source), ["text"]))
    cols = ["id", "comment_text", "toxicity", "identity_attack", "identity_any", "toxicity_annotator_count",
            *CIVIL_IDENTITY_GROUPS]
    return _parquet_rows(download(CIVIL_SOURCE), cols), train_texts


def _civil_meta(settings: dict, stats: dict, **extra: Any) -> dict[str, Any]:
    return {"source": CIVIL_SOURCE, "excluded_if_text_in": CIVIL_TRAIN_SOURCES, "task": CIVIL_TOXICITY.name,
            "toxic_threshold": CIVIL_TOXIC_THRESHOLD, "jigsaw_test_min_id": JIGSAW_TEST_MIN_ID, "sampling": settings,
            "source_stats": stats, "stratified_on": ["identity", "label"], **extra}


def _build_wildguard_prompt_v1() -> tuple[list[Item], dict]:
    settings = {"n_per_cell": 75, "seed": 101, "max_chars": 8000}
    items, stats = wildguard_items(_parquet_rows(download(WILDGUARD_SOURCE)), **settings)
    return items, {"source": WILDGUARD_SOURCE, "task": WILDGUARD_PROMPT.name, "sampling": settings,
                   "source_stats": stats, "stratified_on": ["adversarial", "label"]}


def _build_civil_toxicity_v1() -> tuple[list[Item], dict]:
    rows, train_texts = _civil_rows()
    settings = {"n_per_cell": 75, "seed": 101, "max_chars": 4000}
    items, stats = civil_items(rows, train_texts, **settings)
    return items, _civil_meta(settings, stats)


def _build_wildguard_prompt_full_v1() -> tuple[list[Item], dict]:
    settings = {"n_per_cell": None, "seed": 102, "max_chars": 8000}
    items, stats = wildguard_items(_parquet_rows(download(WILDGUARD_SOURCE)), **settings)
    return items, {"source": WILDGUARD_SOURCE, "task": WILDGUARD_PROMPT.name, "sampling": settings,
                   "source_stats": stats, "stratified_on": []}


def _build_wildguard_response_full_v1() -> tuple[list[Item], dict]:
    settings = {"max_chars": 12000}
    items, stats = wildguard_response_items(_parquet_rows(download(WILDGUARD_SOURCE)), **settings)
    return items, {"source": WILDGUARD_SOURCE, "task": WILDGUARD_RESPONSE.name, "sampling": settings,
                   "source_stats": stats, "stratified_on": []}


def _build_civil_toxicity_pool_v1() -> tuple[list[Item], dict]:
    rows, train_texts = _civil_rows()
    earlier, _ = load_items("civil_toxicity_v1")
    exclude = frozenset(i.meta["jigsaw_id"] for i in earlier)
    settings = {"n_per_cell": 750, "seed": 102, "max_chars": 4000}
    items, stats = civil_items(rows, train_texts, exclude_ids=exclude, **settings)
    return items, _civil_meta(settings, stats, excluded_samples=["civil_toxicity_v1"])


BUILDERS: dict[str, Callable[[], tuple[list[Item], dict]]] = {
    "wildguard_prompt_v1": _build_wildguard_prompt_v1,
    "civil_toxicity_v1": _build_civil_toxicity_v1,
    "wildguard_prompt_full_v1": _build_wildguard_prompt_full_v1,
    "wildguard_response_full_v1": _build_wildguard_response_full_v1,
    "civil_toxicity_pool_v1": _build_civil_toxicity_pool_v1,
}


def build(names: list[str] | None = None, *, force: bool = False) -> None:
    """Build and freeze samples (all by default). A sample whose manifest exists is skipped
    unless force, which rebuilds it; a deterministic rebuild leaves the manifest unchanged."""
    from sysone.env import load_env

    load_env()
    for name in names or list(BUILDERS):
        if (MANIFEST_DIR / name / "manifest.json").exists() and not force:
            print(f"{name}: frozen already, skipped")
            continue
        items, meta = BUILDERS[name]()
        print(f"{name}: {len(items)} items from {meta['source_stats']}")
        print("  ", freeze_items(items, name, meta, force=force))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("names", nargs="*", choices=[[], *BUILDERS], help="samples to build (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild a sample whose manifest exists (must match)")
    args = parser.parse_args()
    build(args.names or None, force=args.force)


if __name__ == "__main__":
    main()
