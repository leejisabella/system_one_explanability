"""The proposals a reviewer sees: Jev's decision, or a random error process standing in for it.

Conditions (research_docs/13):
- jev_label, jev_conf: Jev's decision, without and with its confidence.
- matched_label, matched_conf: random errors with Jev's confusion matrix. Within each true
  class, exactly as many items get the wrong label as Jev got wrong, but the items are
  drawn at random. The error rate and class balance are Jev's; only *which* items are
  wrong differs, so any gap with jev_* is the part of Jev's errors that depends on the input.
  matched_conf shows a confidence drawn from Jev's own confidences, from the pool with the
  same correctness and proposed label: on average it is as calibrated as Jev, but says
  nothing about the item it is shown with.
- flip_p<NN>: each item's true label flipped independently with probability p (the
  secondary curve).

Every task is binary, so an item's wrong label is unique: a matched or flipped error that
lands on an item Jev got wrong is exactly Jev's proposal there.

Jev's decision on an item combines its samples (the option order rotates across them, so
the mean cancels Jev's lean toward the first-listed option): the label with the highest
mean probability, and that mean probability as its confidence.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from sysone.cascade.items import Item

JEV_LABEL, JEV_CONF = "jev_label", "jev_conf"
MATCHED_LABEL, MATCHED_CONF = "matched_label", "matched_conf"
ALONE = "alone"


def flip_condition(p: float) -> str:
    return f"flip_p{round(p * 100):02d}"


@dataclass(frozen=True)
class JevDecision:
    label: str
    # Mean probability of `label` over the parsed samples.
    confidence: float
    n_samples: int
    # The samples didn't all choose the same label.
    split: bool


@dataclass(frozen=True)
class Proposal:
    item_id: str
    condition: str
    label: str
    correct: bool
    # Shown to the reviewer only in the *_conf conditions.
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_jev(samples: list[dict[str, float]], labels: tuple[str, ...]) -> JevDecision | None:
    """Jev's decision from its parsed samples' probabilities; None if no sample parsed.
    Ties go to the label listed first in the task, which keeps the result deterministic."""
    if not samples:
        return None
    mean = {label: sum(s[label] for s in samples) / len(samples) for label in labels}
    label = max(labels, key=lambda lab: (mean[lab], -labels.index(lab)))
    chosen = {max(labels, key=lambda lab: (s[lab], -labels.index(lab))) for s in samples}
    return JevDecision(label, round(mean[label], 4), len(samples), len(chosen) > 1)


def _other(label: str, labels: tuple[str, ...]) -> str:
    (other,) = [lab for lab in labels if lab != label]
    return other


def _rng(seed: int, *parts: str) -> random.Random:
    """A generator for one condition of one dataset, independent of the others."""
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_proposals(items: list[Item], jev: dict[str, JevDecision], labels: tuple[str, ...], *, seed: int,
                    flip_ps: list[float], dataset: str) -> list[Proposal]:
    """Every proposal for one dataset. Items Jev never answered are left out of every
    condition, so all conditions cover the same items."""
    items = sorted((i for i in items if i.item_id in jev), key=lambda i: i.item_id)
    out: list[Proposal] = []
    for i in items:
        d = jev[i.item_id]
        out.append(Proposal(i.item_id, JEV_LABEL, d.label, d.label == i.label, None))
        out.append(Proposal(i.item_id, JEV_CONF, d.label, d.label == i.label, d.confidence))

    # Matched random errors: Jev's error count within each true class, on random items.
    by_class: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_class[i.label].append(i)
    rng = _rng(seed, dataset, "matched")
    wrong: set[str] = set()
    for label in labels:
        pool = by_class.get(label, [])
        n_errors = sum(jev[i.item_id].label != i.label for i in pool)
        wrong.update(i.item_id for i in rng.sample(pool, n_errors))
    # Jev's confidences, pooled by (correct, proposed label).
    pools: dict[tuple[bool, str], list[float]] = defaultdict(list)
    for i in items:
        d = jev[i.item_id]
        pools[(d.label == i.label, d.label)].append(d.confidence)
    conf_rng = _rng(seed, dataset, "matched_conf")
    for i in items:
        is_wrong = i.item_id in wrong
        label = _other(i.label, labels) if is_wrong else i.label
        out.append(Proposal(i.item_id, MATCHED_LABEL, label, not is_wrong, None))
        pool = pools.get((not is_wrong, label)) or pools.get((not is_wrong, _other(label, labels))) or \
            [d.confidence for d in jev.values()]
        out.append(Proposal(i.item_id, MATCHED_CONF, label, not is_wrong, conf_rng.choice(sorted(pool))))

    for p in flip_ps:
        name = flip_condition(p)
        flip_rng = _rng(seed, dataset, name)
        for i in items:
            flipped = flip_rng.random() < p
            out.append(Proposal(i.item_id, name, _other(i.label, labels) if flipped else i.label, not flipped, None))
    return out
