"""The training mix: which training items become which kind of example (E.104).

Jev's mistakes on a training pool mark the items the reviewer probably gets wrong too (E.102:
their errors overlap far beyond chance), so Jev is a cheap detector of the reviewer's blind
spots. The examples always teach the dataset's gold label, never Jev's: Jev is wrong exactly there.

Groups, per pool, with disjoint items:
  jev_wrong      every item Jev gets wrong. Review format: Jev's (wrong) label is shown and the
                 target is to override it with the gold label.
  jev_right      Jev-right items, jev_right_per_wrong times as many as jev_wrong. Review format: the
                 target is to accept. Without these the model would learn "always override".
  matched_wrong  optional: Jev-right items given a random wrong label (the other label: the tasks
                 are binary), matched_wrong_per_wrong times as many. Review format, override.
  alone          other items, alone_per_wrong times as many. Alone format (no proposal), so the
                 reviewer's own judgment is trained too.
A share of items (val_fraction, by hash) is held out for validation loss. The real test set is
E.102's pools, which no training pool overlaps (sysone.cascade.items).

The plain-format control arm (research_docs/14) is built from the same rows in E.105: every item
of the mix, in alone format, with the same gold label.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sysone.cascade.items import Item
from sysone.cascade.proposals import ALONE, JEV_LABEL, MATCHED_LABEL

REVIEW, ALONE_FORMAT = "review", "alone"
JEV_WRONG, JEV_RIGHT, MATCHED_WRONG, ALONE_GROUP = "jev_wrong", "jev_right", "matched_wrong", "alone"
TRAIN, VAL = "train", "val"


@dataclass(frozen=True)
class MixSettings:
    jev_right_per_wrong: float = 2.0
    matched_wrong_per_wrong: float = 0.0
    alone_per_wrong: float = 1.0
    val_fraction: float = 0.05
    # Cap on jev_wrong items per pool (None: all), for a small pilot mix.
    max_jev_wrong: int | None = None
    seed: int = 104


def _rng(seed: int, sample: str, what: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample}:{what}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def split_of(sample: str, item_id: str, val_fraction: float, seed: int) -> str:
    """train or val, by a hash of the item, so it never depends on which other items exist."""
    digest = hashlib.sha256(f"{seed}:{sample}:{item_id}:split".encode()).digest()
    return VAL if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else TRAIN


def build_mix(sample: str, items: list[Item], jev: dict[str, dict[str, Any]], labels: tuple[str, ...],
              settings: MixSettings) -> list[dict[str, Any]]:
    """Mix rows for one pool. `jev` maps item_id to {"label", "confidence", "split"} for every item
    Jev answered (items it didn't are left out)."""
    by_id = {i.item_id: i for i in items}
    answered = sorted(i for i in jev if i in by_id)
    wrong = [i for i in answered if jev[i]["label"] != by_id[i].label]
    right = [i for i in answered if jev[i]["label"] == by_id[i].label]
    if settings.max_jev_wrong is not None and len(wrong) > settings.max_jev_wrong:
        wrong = sorted(_rng(settings.seed, sample, "cap").sample(wrong, settings.max_jev_wrong))
    n = len(wrong)
    pool = right[:]
    _rng(settings.seed, sample, "groups").shuffle(pool)
    want = {MATCHED_WRONG: round(n * settings.matched_wrong_per_wrong), JEV_RIGHT: round(n * settings.jev_right_per_wrong)}
    groups = {JEV_WRONG: wrong}
    start = 0
    for group in (MATCHED_WRONG, JEV_RIGHT):
        groups[group] = sorted(pool[start:start + want[group]])
        start += want[group]
    # Alone items come from the Jev-right items left over: every Jev-wrong item is already in jev_wrong.
    # A group that asks for more items than are left gets what there is.
    groups[ALONE_GROUP] = sorted(pool[start:start + round(n * settings.alone_per_wrong)])

    other = {lab: next(x for x in labels if x != lab) for lab in labels}
    rows = []
    for group, ids in groups.items():
        for item_id in ids:
            item, j = by_id[item_id], jev[item_id]
            if group == ALONE_GROUP:
                fmt, condition, proposed = ALONE_FORMAT, ALONE, None
            elif group == MATCHED_WRONG:
                fmt, condition, proposed = REVIEW, MATCHED_LABEL, other[item.label]
            else:
                fmt, condition, proposed = REVIEW, JEV_LABEL, j["label"]
            rows.append({
                "sample": sample, "item_id": item_id, "task": item.task, "label": item.label, "strata": item.strata,
                "group": group, "format": fmt, "condition": condition, "proposed_label": proposed,
                "target_label": item.label,
                "target_verdict": None if proposed is None else ("accept" if proposed == item.label else "override"),
                "jev_label": j["label"], "jev_correct": j["label"] == item.label, "jev_confidence": j.get("confidence"),
                "split": split_of(sample, item_id, settings.val_fraction, settings.seed),
            })
    return sorted(rows, key=lambda r: (r["group"], r["item_id"]))


def mix_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {}
    for r in rows:
        out.setdefault(r["sample"], Counter())[f"{r['group']}/{r['split']}"] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}
