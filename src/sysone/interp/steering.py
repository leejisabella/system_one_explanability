"""E.110: is giving in to Jev a direction in the reviewer's activations? (research_docs/15)

A study run alongside E.108 and E.109, not a training method. Its data is training-pool items only
(Jev's mistakes that the reviewer solved alone, where E.105 has review traces with Jev's wrong label
shown and at least one of them gave in), so no direction is ever fit on the test set:
  build         select those items and all their review traces (gave in or held), split the items
                80/20 by hash, and freeze the item lists for α selection and the test. No torch: it
                runs on the laptop, where E.105's traces are.
  capture       teacher-forced passes of every selected trace (no generation), recording each decoder
                layer's output (the residual stream) at 3 positions: the token just before the final
                answer (primary), the mean over the reasoning tokens, and the last prompt token,
                which comes before any reasoning. Also each item's alone prompt, at its last token.
  directions    per layer and position, within items: for each fit item with both a give-in and a
                held trace, its mean give-in activation minus its mean held one, averaged over those
                items (so differences between items can't pass for giving in). Measurement 1: the held-out AUC of the projection onto it ("found"
                if the best layer's is at least 0.70), with a logistic probe as a check. Also the
                controls: a random direction, and "proposal present" (review prompt minus alone prompt).
  select-alpha  steered review of the held-out items with Jev's wrong label, and of held-out Jev-right
                training items, at α in {0, 0.5, 1, 2, 4} (α × the direction, so α × its norm along
                it). Rule: the smallest α with the largest drop in giving in, among those that keep
                accept-when-right within 2 points of α = 0. It is frozen to alpha.json before the test.
  steer         measurement 2: test group A with Jev's wrong label, and the Jev-right subsample, at
                the frozen α, once per direction: none (unsteered: the paired baseline on the same
                stack), steer, random and proposal (both scaled to the direction's norm). Rows follow
                E.106's reviews.jsonl, with reviewer <base>+<direction>.
  compare       measurement 3: the same traces through a trained model, and the give-in vs. held gap
                (paired within held-out items, as the direction is) along the base's direction and
                along the model's own.
  report        the three measurements, with paired intervals.

Steering subtracts α × each layer's direction from the output of the best layer and its neighbours,
at every position (prompt and generated), with forward hooks in transformers (not vLLM). Generation
samples as E.106's vLLM server did: temperature 1, no top-k or top-p, reasoning effort medium, at
most 4096 + 256 new tokens. Every prompt is rendered with one fixed "Current date" (chat_date).

Usage (build on the laptop, copy results/runs/<id> to the GPU machine, then scripts/run_e110.sh):
  uv run python -m sysone.interp.steering build --run-id e110-v1-<date>
  uv run --group train python -m sysone.interp.steering capture --run-id <id> --model checkpoints/gpt-oss-20b-base
  uv run --group train python -m sysone.interp.steering directions --run-id <id>
  uv run --group train python -m sysone.interp.steering select-alpha --run-id <id> --model checkpoints/gpt-oss-20b-base
  uv run --group train python -m sysone.interp.steering steer --run-id <id> --model checkpoints/gpt-oss-20b-base
  uv run --group train python -m sysone.interp.steering compare --run-id <id> --model <merged dir> --name consist_review
  uv run --group train python -m sysone.interp.steering report --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sysone.analysis.e101 import _fmt, rate
from sysone.analysis.e103 import _fmt_change, paired_change
from sysone.analysis.e106 import GROUP_A, base_groups
from sysone.cascade.items import Item, load_items
from sysone.cascade.proposals import ALONE, JEV_LABEL
from sysone.cascade.reviewer import parse_review
from sysone.cascade.tasks import TASKS
from sysone.experiments.e102 import CORRECT_LABEL
from sysone.posttrain.formats import REVIEWER_SOURCE, prompt_messages
from sysone.posttrain.mix import JEV_RIGHT, JEV_WRONG, VAL, split_of
from sysone.registry import RESULTS_DIR
from sysone.runs import JsonlWriter, Run, git_info, now_iso
from sysone.types import ChatResponse, Usage

EXPERIMENT = "e110"
POSITIONS = ("pre_answer", "reasoning_mean", "last_prompt")
PRIMARY = POSITIONS[0]
FIT, HELDOUT = "fit", "heldout"
WRONG_SET, JEV_RIGHT_SET = "jev_wrong", "jev_right"  # the test's jev_wrong set is group A only
NONE, STEER, RANDOM, PROPOSAL = "none", "steer", "random", "proposal"
DIRECTIONS = (NONE, STEER, RANDOM, PROPOSAL)
BASE_TAG = "base"
# gpt-oss's harmony format, as its chat template renders an assistant turn with thinking.
ANALYSIS, FINAL, END, RETURN = "<|channel|>analysis<|message|>", "<|channel|>final<|message|>", "<|end|>", "<|return|>"


@dataclass(frozen=True)
class E110Config:
    e104_run: str = "e104-v1-2026-11-04"   # the training mix
    e105_run: str = "e105-v1-2026-11-04"   # the review traces
    e102_run: str = "e102-v1-2026-11-04"   # the test plan
    e106_run: str = "e106-v1-2026-11-04"   # the base's alone record (test groups); trained models' evaluations
    base_reviewer: str = "base_local"
    heldout_fraction: float = 0.2
    seed: int = 110
    alpha_right_items: int = 150           # held-out Jev-right training items, for α's side-effect check
    alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    neighbours: int = 1                    # steer at the best layer and this many layers on each side
    found_auc: float = 0.70
    max_accept_drop: float = 0.02
    reasoning_effort: str = "medium"
    # gpt-oss's template stamps "Current date: <today>"; one fixed date keeps every capture and every
    # steered pass on the same prompt, whichever day it runs.
    chat_date: str = "2026-11-05"
    max_new_tokens: int = 4096 + 256       # E.106's reasoning budget plus its answer budget
    temperature: float = 1.0               # vLLM's default, which E.106 used (reviewer_temperature: null)
    probe_l2: float = 1e-2


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") % 2**63


# --- build: items, traces and splits (offline) --------------------------------------------------

def gave_in(t: dict[str, Any]) -> bool:
    return t["final_label"] == t["proposed_label"]


def select_traces(traces: list[dict[str, Any]], *, heldout_fraction: float, seed: int
                  ) -> tuple[list[dict[str, Any]], Counter]:
    """The reviewer's own review traces on Jev's mistakes it solved alone, for items where at least
    one gave in to Jev's label. Every trace of such an item is kept, gave in or held; the split is
    per item, so an item's traces never straddle fit and held-out. Also the items left out, by reason."""
    by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        if t["source"] == REVIEWER_SOURCE and t["group"] == JEV_WRONG and t["status"] == "ok" and t["parse_ok"]:
            by_item[(t["sample"], t["item_id"])].append(t)
    out, dropped = [], Counter()
    for (sample, item_id), ts in sorted(by_item.items()):
        review = sorted((t for t in ts if t["condition"] == JEV_LABEL and t.get("reasoning")),
                        key=lambda t: t["sample_index"])
        if not any(t["condition"] == ALONE and t["final_label"] == t["label"] for t in ts):
            dropped[f"{sample}/not_solved_alone"] += 1
            continue
        if not any(gave_in(t) for t in review):
            dropped[f"{sample}/never_gave_in"] += 1
            continue
        split = HELDOUT if split_of(sample, item_id, heldout_fraction, seed) == VAL else FIT
        out += [{"trace_id": f"{sample}/{item_id}/{t['sample_index']}", "sample": sample, "item_id": item_id,
                 "task": t["task"], "label": t["label"], "proposed_label": t["proposed_label"],
                 "sample_index": t["sample_index"], "verdict": t["verdict"], "final_label": t["final_label"],
                 "reasoning": t["reasoning"], "gave_in": gave_in(t), "split": split} for t in review]
    return out, dropped


def alpha_items(selection: list[dict[str, Any]], mix: list[dict[str, Any]], *, n_right: int, seed: int
                ) -> list[dict[str, Any]]:
    """α's items, all from the training pools: the held-out selected items with Jev's wrong label,
    and a seeded sample of Jev-right items with Jev's (correct) label, reviewed as correct_label."""
    strata = {(r["sample"], r["item_id"]): r["strata"] for r in mix}
    held = {(t["sample"], t["item_id"]): t for t in selection if t["split"] == HELDOUT}
    rows = [{"set": WRONG_SET, "sample": s, "item_id": i, "task": t["task"], "label": t["label"],
             "strata": strata.get((s, i), {}), "condition": JEV_LABEL, "proposed_label": t["proposed_label"],
             "proposed_correct": False} for (s, i), t in sorted(held.items())]
    pools = {t["sample"] for t in selection}
    right = sorted((r for r in mix if r["group"] == JEV_RIGHT and r["sample"] in pools),
                   key=lambda r: (r["sample"], r["item_id"]))
    picked = random.Random(_seed(seed, "alpha_right")).sample(right, min(n_right, len(right)))
    rows += [{"set": JEV_RIGHT_SET, "sample": r["sample"], "item_id": r["item_id"], "task": r["task"],
              "label": r["label"], "strata": r["strata"], "condition": CORRECT_LABEL, "proposed_label": r["label"],
              "proposed_correct": True} for r in sorted(picked, key=lambda r: (r["sample"], r["item_id"]))]
    return rows


def eval_items(plan: list[dict[str, Any]], groups: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    """Measurement 2's test: group A (right alone on the base's first try) with Jev's wrong label,
    and E.102's Jev-right subsample with the correct label, as E.106 reviews them."""
    out = []
    for p in sorted(plan, key=lambda p: (p["sample"], p["item_id"])):
        base = {"sample": p["sample"], "item_id": p["item_id"], "label": p["label"], "strata": p["strata"]}
        if not p["jev_correct"] and groups.get((p["sample"], p["item_id"])) == GROUP_A:
            out.append({**base, "set": WRONG_SET, "condition": JEV_LABEL, "proposed_label": p["jev_label"],
                        "proposed_correct": False})
        if p["in_subsample"] and p["jev_correct"]:
            out.append({**base, "set": JEV_RIGHT_SET, "condition": CORRECT_LABEL, "proposed_label": p["label"],
                        "proposed_correct": True})
    return out


def build_counts(selection: list[dict[str, Any]], dropped: Counter, alpha: list[dict[str, Any]],
                 tests: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, Counter] = defaultdict(Counter)
    items: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for t in selection:
        key = (t["sample"], t["item_id"])
        per[t["sample"]]["traces_gave_in" if t["gave_in"] else "traces_held"] += 1
        items[t["sample"]][t["split"]].add(key)
        items[t["sample"]]["gave_in" if t["gave_in"] else "held"].add(key)
    for s, its in items.items():
        per[s].update({"items": len(its[FIT] | its[HELDOUT]), "items_fit": len(its[FIT]),
                       "items_heldout": len(its[HELDOUT]), "items_with_both": len(its["gave_in"] & its["held"]),
                       # the items the direction is fit on, and the held-out items its gap is paired on
                       "items_with_both_fit": len(its["gave_in"] & its["held"] & its[FIT]),
                       "items_with_both_heldout": len(its["gave_in"] & its["held"] & its[HELDOUT])})
    total = Counter()
    for c in per.values():
        total.update(c)
    return {"selection": {**{s: dict(sorted(c.items())) for s, c in sorted(per.items())}, "total": dict(sorted(total.items()))},
            "left_out": dict(sorted(dropped.items())),
            "alpha_items": dict(sorted(Counter(f"{r['set']}/{r['sample']}" for r in alpha).items())),
            "test_items": dict(sorted(Counter(f"{r['set']}/{r['sample']}" for r in tests).items()))}


def _write_frozen(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if path.exists() and path.read_text() != body:
        raise ValueError(f"{path} exists with different rows; build under a new run id")
    path.write_text(body)


def build(config: E110Config, run: Run, runs_dir: Path) -> dict[str, Any]:
    traces = JsonlWriter(runs_dir / config.e105_run / "traces.jsonl").rows()
    mix = JsonlWriter(runs_dir / config.e104_run / "mix.jsonl").rows()
    plan = JsonlWriter(runs_dir / config.e102_run / "plan.jsonl").rows()
    evals = [r for r in JsonlWriter(runs_dir / config.e106_run / "reviews.jsonl").rows()
             if r["status"] == "ok" and r["parse_ok"]]
    if not (traces and mix and plan and evals):
        raise ValueError("build needs E.105's traces, E.104's mix, E.102's plan and E.106's reviews")
    selection, dropped = select_traces(traces, heldout_fraction=config.heldout_fraction, seed=config.seed)
    alpha = alpha_items(selection, mix, n_right=config.alpha_right_items, seed=config.seed)
    tests = eval_items(plan, base_groups(evals, config.base_reviewer))
    for name, rows in (("selection", selection), ("alpha_items", alpha), ("test_items", tests)):
        _write_frozen(run.dir / f"{name}.jsonl", rows)
    counts = build_counts(selection, dropped, alpha, tests)
    run.update_manifest(experiment=EXPERIMENT, config=asdict(config), counts=counts, built_at=now_iso(), git=git_info())
    return counts


# --- statistics (pure Python: projections are short lists, and this is what the tests check) ----

def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a positive scores above a negative), ties counting half (Mann-Whitney)."""
    if not pos or not neg:
        return None
    scored = sorted([(s, True) for s in pos] + [(s, False) for s in neg], key=lambda x: x[0])
    rank_sum, i = 0.0, 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        mid = (i + 1 + j) / 2  # the average 1-based rank of the tie block
        rank_sum += mid * sum(1 for _, p in scored[i:j] if p)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _mean(vectors: list) -> Any:
    if vectors and isinstance(vectors[0], list):
        return [sum(c) / len(vectors) for c in zip(*vectors)]
    return sum(vectors) / len(vectors)


def _item_diffs(values: list, gave: list[bool], items: list) -> list:
    """Per item with both kinds of trace: the mean of its give-in values minus the mean of its held
    ones. Items with only one kind contribute nothing, so differences between items (what makes the
    reviewer give in on some items at all) can't pass for the give-in itself."""
    by_item: dict[Any, tuple[list, list]] = defaultdict(lambda: ([], []))
    for v, g, item in zip(values, gave, items):
        by_item[item][0 if g else 1].append(v)
    diffs = []
    for item in sorted(by_item):
        pos, neg = by_item[item]
        if pos and neg:
            m1, m0 = _mean(pos), _mean(neg)
            diffs.append([a - b for a, b in zip(m1, m0)] if isinstance(m1, list) else m1 - m0)
    return diffs


def paired_diff(vectors: list[list[float]], gave: list[bool], items: list) -> tuple[list[float], int]:
    """The direction: the mean over items of (mean give-in vector - mean held vector), using only
    items with both; and how many items that was."""
    diffs = _item_diffs(vectors, gave, items)
    if not diffs:
        raise ValueError("no item has both a give-in and a held trace")
    return _mean(diffs), len(diffs)


def project(vectors: list[list[float]], direction: list[float]) -> list[float]:
    """Each vector's coordinate along the unit direction."""
    norm = math.sqrt(sum(x * x for x in direction))
    return [sum(a * b for a, b in zip(v, direction)) / norm for v in vectors]


def paired_gap(scores: list[float], gave: list[bool], items: list) -> dict[str, Any]:
    """The give-in vs. held gap along a direction, paired within items as the direction is: the
    mean per-item gap, and that over the per-item gaps' standard deviation (comparable across models
    whose activations differ in scale). The AUC is over all the traces."""
    diffs = _item_diffs(scores, gave, items)
    m = sum(diffs) / len(diffs) if diffs else None
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    a = auc([s for s, g in zip(scores, gave) if g], [s for s, g in zip(scores, gave) if not g])
    return {"gap": None if m is None else round(m, 6), "std_gap": round(m / sd, 4) if sd > 0 else None,
            "auc": None if a is None else round(a, 4), "n_items_paired": len(diffs),
            "n_gave_in": sum(gave), "n_held": len(gave) - sum(gave)}


def best_layer(aucs: list[float | None]) -> int:
    """The layer with the highest held-out AUC, the earliest on a tie."""
    return max(range(len(aucs)), key=lambda layer: (aucs[layer] if aucs[layer] is not None else -1.0, -layer))


def steer_layers(best: int, n_layers: int, neighbours: int) -> list[int]:
    return [layer for layer in range(best - neighbours, best + neighbours + 1) if 0 <= layer < n_layers]


def choose_alpha(table: dict[float, dict[str, float]], *, max_accept_drop: float) -> dict[str, Any]:
    """table: α -> {"give_in", "accept"} rates, with α = 0 the unsteered baseline. The smallest α
    with the largest drop in giving in, among those whose accept-when-right falls by at most
    max_accept_drop. If none qualifies, the smallest α, flagged not admissible."""
    base = table[0.0]
    rows = [{"alpha": a, "give_in_drop": round(base["give_in"] - t["give_in"], 9),
             "accept_drop": round(base["accept"] - t["accept"], 9)} for a, t in sorted(table.items()) if a > 0]
    ok = [r for r in rows if r["accept_drop"] <= max_accept_drop]
    if not ok:
        return {"alpha": rows[0]["alpha"], "admissible": False, "candidates": rows}
    top = max(r["give_in_drop"] for r in ok)
    return {"alpha": min(r["alpha"] for r in ok if r["give_in_drop"] == top), "admissible": True, "candidates": rows}


# --- harmony positions and replies --------------------------------------------------------------

def _last_starting_before(offsets: list[tuple[int, int]], end: int) -> int:
    idx = max(i for i, (s, _) in enumerate(offsets) if s < end)
    if offsets[idx][1] != end:
        raise ValueError(f"token {idx} {offsets[idx]} straddles character {end}")
    return idx


def positions(full: str, head_len: int, offsets: list[tuple[int, int]]) -> dict[str, Any]:
    """Token indices in a rendered review trace, from the tokenizer's character offsets:
      last_prompt  the prompt's last token (`assistant` in `<|start|>assistant`, before any reasoning)
      reasoning    [first, last + 1) of the tokens inside the analysis channel
      pre_answer   the token just before the final answer (`<|message|>` after `<|channel|>final`)"""
    if any(b[0] < a[0] for a, b in zip(offsets, offsets[1:])):
        raise ValueError("token offsets are not in order; the tokenizer's offsets can't locate positions")
    r0 = full.index(ANALYSIS, head_len) + len(ANALYSIS)
    r1 = full.index(END, r0)
    answer = full.index(FINAL, r1) + len(FINAL)
    inside = [i for i, (s, e) in enumerate(offsets) if s >= r0 and e <= r1 and e > s]
    if not inside:
        raise ValueError("the trace has no reasoning tokens")
    return {"last_prompt": _last_starting_before(offsets, head_len), "reasoning": (inside[0], inside[-1] + 1),
            "pre_answer": _last_starting_before(offsets, answer)}


def split_generation(text: str) -> tuple[str | None, str, bool]:
    """(reasoning, final answer, finished) from a generated assistant turn in harmony format."""
    reasoning = None
    if ANALYSIS in text:
        start = text.index(ANALYSIS) + len(ANALYSIS)
        stop = text.find(END, start)
        reasoning = text[start:stop if stop >= 0 else None]
    if FINAL not in text:
        return reasoning, "", False
    final = text[text.rindex(FINAL) + len(FINAL):]
    for marker in (RETURN, END):
        if marker in final:
            return reasoning, final[:final.index(marker)], True
    return reasoning, final, False


def review_row(run_id: str, reviewer: str, row: dict[str, Any], *, task: str, generated: str, model_used: str,
               input_tokens: int, output_tokens: int, reasoning_tokens: int, seconds: float | None,
               request_key: str, steering: dict[str, Any]) -> dict[str, Any]:
    """A generated review as a row of E.106's reviews.jsonl, parsed by the reviewer's own parser."""
    reasoning, final, finished = split_generation(generated)
    response = ChatResponse(text=final, usage=Usage(input_tokens, output_tokens, reasoning_tokens), cost_usd=0.0,
                            provider="hf", model_requested=model_used, model_used=model_used,
                            finish_reason="stop" if finished else "length", raw={})
    r = parse_review(response, request_key, TASKS[task].labels, row["proposed_label"])
    return {"run_id": run_id, "reviewer": reviewer, "role": "reviewer", "sample": row["sample"],
            "item_id": row["item_id"], "task": task, "condition": row["condition"], "sample_index": 0,
            "label": row["label"], "strata": row["strata"], "proposed_label": row["proposed_label"],
            "proposed_correct": row["proposed_correct"], "confidence_shown": None, "status": "ok",
            "final_label": r.final_label, "verdict": r.verdict, "consistent": r.consistent, "parse_ok": r.parse_ok,
            "parse_error": r.parse_error, "raw_text": None if r.parse_ok else generated[:2000],
            "request_key": request_key, "reasoning": reasoning, "model_used": model_used, "cost_usd": 0.0,
            "cached": False, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens, "finish_reason": response.finish_reason, "service_tier": None,
            "latency_s": seconds, "wall_s": seconds, "steering": steering, "ts": now_iso()}


# --- GPU: capture and steering (torch imported inside) -------------------------------------------

def _layers(model: Any) -> Any:
    # GptOssForCausalLM.model is GptOssModel; in transformers 4.57 each GptOssDecoderLayer returns
    # the residual stream after its attention and MoE blocks, as a tensor.
    return model.model.layers


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def steering(model: Any, vectors: dict[int, Any]):
    """Subtract vectors[layer] from that decoder layer's output, at every position."""
    handles = []
    for layer, v in vectors.items():
        module = _layers(model)[layer]
        p = next(module.parameters())
        v = v.to(p.device, p.dtype)

        def hook(_module, _args, output, v=v):
            return (output[0] - v, *output[1:]) if isinstance(output, tuple) else output - v

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _capture(model: Any, seqs: list[tuple[list[int], dict[str, Any]]], *, token_budget: int, pad_id: int):
    """[layers, sequences, POSITIONS, hidden] in bf16 on the CPU. Right padding, so every position
    keeps its unpadded index and, attention being causal, its value."""
    import torch

    layers = _layers(model)
    out = torch.empty(len(layers), len(seqs), len(POSITIONS), model.config.hidden_size, dtype=torch.bfloat16)
    batches, cur = [], []
    for i in sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])):
        if cur and (len(cur) + 1) * len(seqs[i][0]) > token_budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches += [cur] if cur else []
    current: dict[str, list[int]] = {}

    def recorder(layer: int):
        def hook(_module, _args, output):
            h = _hidden(output)
            rows = []
            for b, i in enumerate(current["idx"]):
                p = seqs[i][1]
                r0, r1 = p["reasoning"]
                rows.append(torch.stack([h[b, p["pre_answer"]].float(), h[b, r0:r1].float().mean(0),
                                         h[b, p["last_prompt"]].float()]))
            out[layer, torch.tensor(current["idx"])] = torch.stack(rows).to("cpu", torch.bfloat16)
        return hook

    handles = [m.register_forward_hook(recorder(k)) for k, m in enumerate(layers)]
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for n, idx in enumerate(batches):
                current["idx"] = idx
                width = max(len(seqs[i][0]) for i in idx)
                ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for b, i in enumerate(idx):
                    ids[b, :len(seqs[i][0])] = torch.tensor(seqs[i][0])
                    mask[b, :len(seqs[i][0])] = 1
                # The decoder alone: the LM head's logits over the vocabulary aren't needed.
                model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
                if n % 20 == 0:
                    print(f"  capture batch {n + 1}/{len(batches)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return out


def _items(samples: set[str]) -> dict[tuple[str, str], Item]:
    return {(name, i.item_id): i for name in sorted(samples) for i in load_items(name)[0]}


def _load(model_path: str):
    from transformers import AutoTokenizer

    from sysone.posttrain.hf import load_model

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return load_model(model_path, train=False).eval(), tok


def fixed_date(date: str):
    """A stand-in for the chat template's strftime_now, which it calls for "Current date". transformers
    passes apply_chat_template's extra keyword arguments to the template as variables, and those take
    precedence over the template environment's strftime_now global."""
    day = datetime.strptime(date, "%Y-%m-%d")
    return lambda fmt: day.strftime(fmt)


def _template(tok: Any, messages: list[dict], config: E110Config, *, generation_prompt: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                   reasoning_effort=config.reasoning_effort, strftime_now=fixed_date(config.chat_date))


def _prompt(tok: Any, task: str, text: str, proposed: str | None, config: E110Config) -> str:
    return _template(tok, prompt_messages(task, text, proposed), config, generation_prompt=True)


def _render(tok: Any, prompt: list[dict], completion: list[dict], config: E110Config) -> tuple[str, str]:
    """sysone.posttrain.hf.render, with the fixed date."""
    head = _template(tok, prompt, config, generation_prompt=True)
    full = _template(tok, prompt + completion, config, generation_prompt=False)
    if not full.startswith(head) or f"Current date: {config.chat_date}" not in head:
        raise ValueError("the chat template doesn't render the prompt as a prefix of the conversation with the "
                         "fixed date; check the tokenizer's template")
    return head, full[len(head):]


def capture(run: Run, config: E110Config, model_path: str, tag: str, *, token_budget: int) -> Path:
    """Every selected trace, teacher-forced through the model, and each item's alone prompt.
    Saved per layer as activations/<tag>/layer_NN.pt: {"review": [traces, POSITIONS, hidden],
    "alone": [items, hidden]}; index.json, written last, marks it done."""
    out_dir = run.dir / "activations" / tag
    if (out_dir / "index.json").exists():
        print(f"activations/{tag} already captured")
        return out_dir
    import torch

    selection = run.writer("selection").rows()
    items = _items({t["sample"] for t in selection})
    model, tok = _load(model_path)
    seqs = []
    for t in selection:
        prompt = prompt_messages(t["task"], items[(t["sample"], t["item_id"])].text, t["proposed_label"])
        # The answer comes after every recorded position, so its exact spelling doesn't matter.
        answer = json.dumps({"verdict": t["verdict"], "final_label": t["final_label"]})
        head, tail = _render(tok, prompt, [{"role": "assistant", "thinking": t["reasoning"], "content": answer}], config)
        enc = tok(head + tail, add_special_tokens=False, return_offsets_mapping=True)
        seqs.append((enc["input_ids"], positions(head + tail, len(head), [tuple(o) for o in enc["offset_mapping"]])))
    alone_keys = sorted({(t["sample"], t["item_id"]) for t in selection})
    alone = []
    for key in alone_keys:
        ids = tok(_prompt(tok, items[key].task, items[key].text, None, config),
                  add_special_tokens=False)["input_ids"]
        last = len(ids) - 1
        alone.append((ids, {"pre_answer": last, "reasoning": (last, last + 1), "last_prompt": last}))
    review = _capture(model, seqs, token_budget=token_budget, pad_id=tok.pad_token_id)
    alone_acts = _capture(model, alone, token_budget=token_budget, pad_id=tok.pad_token_id)[:, :, POSITIONS.index("last_prompt")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(review.shape[0]):
        torch.save({"review": review[layer].clone(), "alone": alone_acts[layer].clone()}, out_dir / f"layer_{layer:02d}.pt")
    (out_dir / "index.json").write_text(json.dumps({
        "model": model_path, "tag": tag, "positions": POSITIONS, "layers": review.shape[0], "hidden": review.shape[-1],
        "traces": [t["trace_id"] for t in selection], "alone_items": alone_keys,
        "tokens": {"max": max(len(s[0]) for s in seqs), "mean": sum(len(s[0]) for s in seqs) / len(seqs)},
        "finished_at": now_iso(), "git": git_info()}, indent=2) + "\n")
    return out_dir


def _layer(run: Run, tag: str, layer: int) -> dict[str, Any]:
    import torch

    return torch.load(run.dir / "activations" / tag / f"layer_{layer:02d}.pt")


def _probe_auc(x_fit, y_fit: list[bool], x_held, y_held: list[bool], l2: float) -> float | None:
    """Held-out AUC of an L2-regularized logistic probe on standardized activations."""
    import torch

    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    xf, xh = (x_fit - mu) / sd, (x_held - mu) / sd
    y = torch.tensor(y_fit, dtype=torch.float32)
    w = torch.zeros(xf.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xf @ w + b, y) + l2 * (w * w).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    scores = (xh @ w + b).detach().tolist()
    return auc([s for s, p in zip(scores, y_held) if p], [s for s, p in zip(scores, y_held) if not p])


def _split(selection: list[dict[str, Any]]) -> tuple[list[int], list[int], list[bool], list[tuple[str, str]]]:
    fit = [i for i, t in enumerate(selection) if t["split"] == FIT]
    held = [i for i, t in enumerate(selection) if t["split"] == HELDOUT]
    return fit, held, [t["gave_in"] for t in selection], [(t["sample"], t["item_id"]) for t in selection]


def _fit(x, idx: list[int], gave: list[bool], items: list) -> tuple[list[float], int]:
    return paired_diff(x[idx].tolist(), [gave[i] for i in idx], [items[i] for i in idx])


def _gap(x, idx: list[int], gave: list[bool], items: list, direction: list[float]) -> dict[str, Any]:
    return paired_gap(project(x[idx].tolist(), direction), [gave[i] for i in idx], [items[i] for i in idx])


def _checked_selection(run: Run, tag: str) -> list[dict[str, Any]]:
    selection = run.writer("selection").rows()
    index = json.loads((run.dir / "activations" / tag / "index.json").read_text())
    if index["traces"] != [t["trace_id"] for t in selection]:
        raise ValueError(f"activations/{tag} were captured for a different selection")
    return selection


def directions(run: Run, config: E110Config) -> dict[str, Any]:
    """Measurement 1 and the directions steering uses, from the base's activations."""
    import torch

    selection = _checked_selection(run, BASE_TAG)
    index = json.loads((run.dir / "activations" / BASE_TAG / "index.json").read_text())
    fit, held, gave, items = _split(selection)
    n_layers, hidden = index["layers"], index["hidden"]
    alone_row = {tuple(k): j for j, k in enumerate(index["alone_items"])}
    first_trace = {}
    for i in fit:
        first_trace.setdefault((selection[i]["sample"], selection[i]["item_id"]), i)
    diff = torch.zeros(n_layers, len(POSITIONS), hidden)
    proposal = torch.zeros(n_layers, hidden)
    m: dict[str, Any] = {"auc": {p: [] for p in POSITIONS}, "probe_auc": {p: [] for p in POSITIONS},
                         "gap": {p: [] for p in POSITIONS}, "norm": {p: [] for p in POSITIONS}}
    for layer in range(n_layers):
        acts = _layer(run, BASE_TAG, layer)
        review = acts["review"].float()
        for k, pos in enumerate(POSITIONS):
            x = review[:, k]
            d, n_fit_items = _fit(x, fit, gave, items)
            diff[layer, k] = torch.tensor(d)
            g = _gap(x, held, gave, items, d)
            m["auc"][pos].append(g["auc"])
            m["gap"][pos].append(g)
            m["norm"][pos].append(round(float(diff[layer, k].norm()), 4))
            m["probe_auc"][pos].append(_probe_auc(x[fit], [gave[i] for i in fit], x[held], [gave[i] for i in held],
                                                  config.probe_l2))
        # "Proposal present": the review prompt's last token minus the alone prompt's, per fit item.
        keys = sorted(first_trace)
        last = review[:, POSITIONS.index("last_prompt")]
        vecs = [last[first_trace[key]].tolist() for key in keys] + [acts["alone"][alone_row[key]].float().tolist() for key in keys]
        proposal[layer] = torch.tensor(paired_diff(vecs, [True] * len(keys) + [False] * len(keys), keys + keys)[0])
        print(f"  layer {layer}: held-out AUC " + ", ".join(f"{p} {m['auc'][p][-1]}" for p in POSITIONS), flush=True)
    best = best_layer(m["auc"][PRIMARY])
    rand = torch.randn(n_layers, hidden, generator=torch.Generator().manual_seed(config.seed))
    torch.save({"diff": diff, "random": rand, "proposal": proposal, "positions": POSITIONS}, run.dir / "directions.pt")
    out = {**m, "best_layer": best, "best_auc": m["auc"][PRIMARY][best], "found": m["auc"][PRIMARY][best] >= config.found_auc,
           "steer_layers": steer_layers(best, n_layers, config.neighbours), "layers": n_layers,
           "n": {"fit_items_paired": n_fit_items, "heldout_items_paired": m["gap"][PRIMARY][0]["n_items_paired"],
                 "fit_traces": len(fit), "heldout_traces": len(held), "fit_gave_in": sum(gave[i] for i in fit),
                 "heldout_gave_in": sum(gave[i] for i in held)},
           "fit": "per item with both a give-in and a held trace, its mean give-in activation minus its mean held one; "
                  "the direction is the mean of those over the fit items. Gaps are paired the same way; AUCs and the "
                  "probe use every held-out trace.",
           "rule": f"found if the best layer's held-out AUC at {PRIMARY} is at least {config.found_auc}",
           "finished_at": now_iso(), "git": git_info()}
    (run.dir / "directions.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def steer_vectors(dirs: dict[str, Any], direction: str, alpha: float, layers: list[int]) -> dict[int, Any]:
    """α × the layer's direction at the primary position, or a control rescaled to its norm."""
    if direction == NONE or alpha == 0:
        return {}
    out = {}
    for layer in layers:
        d = dirs["diff"][layer, POSITIONS.index(PRIMARY)]
        v = d if direction == STEER else dirs[direction][layer] / dirs[direction][layer].norm() * d.norm()
        out[layer] = alpha * v
    return out


def _generate(model: Any, tok: Any, prompts: list[str], *, vectors: dict[int, Any], batch_size: int,
              max_new_tokens: int, temperature: float, seed: int, on_batch) -> None:
    """Sampled generation in batches (left padding), steered by `vectors`. Batches are seeded by
    position, so the directions run on the same items in the same batches with the same seeds."""
    import torch

    tok.padding_side = "left"
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    end_id = tok.convert_tokens_to_ids(END)
    device = model.get_input_embeddings().weight.device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    with steering(model, vectors), torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            torch.manual_seed(_seed(seed, start))
            t0 = time.monotonic()
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            secs = round(time.monotonic() - t0, 2)
            results = []
            for b, i in enumerate(idx):
                gen = out[b, enc["input_ids"].shape[1]:].tolist()
                stop = next((k for k, x in enumerate(gen) if x in eos), None)
                gen = gen if stop is None else gen[:stop + 1]
                results.append((i, tok.decode(gen, skip_special_tokens=False), int(enc["attention_mask"][b].sum()),
                                len(gen), gen.index(end_id) if end_id in gen else len(gen), secs))
            on_batch(results)
            print(f"  generated {min(start + batch_size, len(order))}/{len(order)} ({secs}s for the batch)", flush=True)


def review_with(run: Run, config: E110Config, model: Any, tok: Any, model_path: str, rows: list[dict[str, Any]],
                items: dict[tuple[str, str], Item], *, out: str, reviewer: str, vectors: dict[int, Any],
                steering_info: dict[str, Any], batch_size: int) -> None:
    """Review `rows` as `reviewer`, appending to <out>.jsonl and skipping rows already there."""
    writer = run.writer(out)
    done = {(r["reviewer"], r["sample"], r["item_id"], r["condition"]) for r in writer.rows() if r["status"] == "ok"}
    todo = [r for r in rows if (reviewer, r["sample"], r["item_id"], r["condition"]) not in done]
    if not todo:
        print(f"{reviewer}: all {len(rows)} reviews done")
        return
    prompts = [_prompt(tok, items[(r["sample"], r["item_id"])].task, items[(r["sample"], r["item_id"])].text,
                       r["proposed_label"], config) for r in todo]
    print(f"{reviewer}: {len(todo)} reviews to generate", flush=True)

    def on_batch(results):
        for i, text, n_in, n_out, n_reason, secs in results:
            key = hashlib.sha256(json.dumps([model_path, prompts[i], steering_info], sort_keys=True).encode()).hexdigest()
            writer.append(review_row(run.run_id, reviewer, todo[i], task=items[(todo[i]["sample"], todo[i]["item_id"])].task,
                                     generated=text, model_used=model_path, input_tokens=n_in, output_tokens=n_out,
                                     reasoning_tokens=n_reason, seconds=secs, request_key=key, steering=steering_info))

    _generate(model, tok, prompts, vectors=vectors, batch_size=batch_size, max_new_tokens=config.max_new_tokens,
              temperature=config.temperature, seed=config.seed, on_batch=on_batch)


def _dirs(run: Run) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    return torch.load(run.dir / "directions.pt"), json.loads((run.dir / "directions.json").read_text())


def outcome_flags(rows: list[dict[str, Any]], reviewer: str, tests: list[dict[str, Any]]) -> dict[str, dict]:
    """Per item, keyed (sample, item_id): whether the reviewer accepted the label shown, on group A
    (Jev's wrong label: giving in) and on the Jev-right items (accept-when-right)."""
    sets = {(t["sample"], t["item_id"], t["condition"]): t["set"] for t in tests}
    out: dict[str, dict] = {WRONG_SET: {}, JEV_RIGHT_SET: {}}
    for r in rows:
        if r["reviewer"] == reviewer and r["status"] == "ok" and r["parse_ok"]:
            s = sets.get((r["sample"], r["item_id"], r["condition"]))
            if s:
                out[s][(r["sample"], r["item_id"])] = r["final_label"] == r["proposed_label"]
    return out


def alpha_table(rows: list[dict[str, Any]], alpha_rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    out = {}
    for a in sorted({r["steering"]["alpha"] for r in rows}):
        name = next(r["reviewer"] for r in rows if r["steering"]["alpha"] == a)
        f = outcome_flags(rows, name, alpha_rows)
        out[a] = {"reviewer": name, "give_in": rate(list(f[WRONG_SET].values())),
                  "accept": rate(list(f[JEV_RIGHT_SET].values()))}
    return out


def select_alpha(run: Run, config: E110Config, model_path: str, *, batch_size: int) -> dict[str, Any]:
    path = run.dir / "alpha.json"
    if path.exists():
        print(f"α already frozen: {path.read_text()}")
        return json.loads(path.read_text())
    dirs, meta = _dirs(run)
    rows = run.writer("alpha_items").rows()
    items = _items({r["sample"] for r in rows})
    model, tok = _load(model_path)
    for a in [0.0, *config.alphas]:
        info = {"direction": STEER if a else NONE, "alpha": a, "layers": meta["steer_layers"] if a else [],
                "position": PRIMARY}
        review_with(run, config, model, tok, model_path, rows, items, out="alpha_reviews",
                    reviewer=f"{config.base_reviewer}+alpha{a:g}",
                    vectors=steer_vectors(dirs, STEER, a, meta["steer_layers"]), steering_info=info, batch_size=batch_size)
    table = alpha_table(run.writer("alpha_reviews").rows(), rows)
    choice = choose_alpha({a: {"give_in": t["give_in"]["rate"], "accept": t["accept"]["rate"]} for a, t in table.items()},
                          max_accept_drop=config.max_accept_drop)
    frozen = {**choice, "layers": meta["steer_layers"], "best_layer": meta["best_layer"], "position": PRIMARY,
              "table": {f"{a:g}": t for a, t in table.items()},
              "rule": f"the smallest α with the largest drop in giving in on the held-out training items, among those "
                      f"whose accept-when-right falls by at most {config.max_accept_drop:.0%} from α = 0",
              "frozen_at": now_iso(), "git": git_info()}
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def steer(run: Run, config: E110Config, model_path: str, names: list[str], *, batch_size: int) -> None:
    """Measurement 2 on the test items, once per direction in `names`."""
    path = run.dir / "alpha.json"
    if any(n != NONE for n in names) and not path.exists():
        raise SystemExit("α isn't frozen: run select-alpha first")
    dirs, meta = _dirs(run)
    alpha = json.loads(path.read_text()) if path.exists() else {"alpha": 0.0, "layers": []}
    rows = run.writer("test_items").rows()
    items = _items({r["sample"] for r in rows})
    model, tok = _load(model_path)
    for name in names:
        a = 0.0 if name == NONE else alpha["alpha"]
        info = {"direction": name, "alpha": a, "layers": alpha["layers"] if a else [], "position": PRIMARY}
        review_with(run, config, model, tok, model_path, rows, items, out="reviews", reviewer=f"{config.base_reviewer}+{name}",
                    vectors=steer_vectors(dirs, name, a, alpha["layers"]), steering_info=info, batch_size=batch_size)


def compare(run: Run, config: E110Config, model_path: str, name: str, *, token_budget: int) -> dict[str, Any]:
    """Measurement 3 for one trained model, at every layer of the primary position."""
    capture(run, config, model_path, name, token_budget=token_budget)
    selection = _checked_selection(run, name)
    dirs, meta = _dirs(run)
    fit, held, gave, items = _split(selection)
    k = POSITIONS.index(PRIMARY)
    layers = []
    for layer in range(meta["layers"]):
        x = _layer(run, name, layer)["review"].float()[:, k]
        base_d = dirs["diff"][layer, k]
        own, _ = _fit(x, fit, gave, items)
        own_t = base_d.new_tensor(own)
        layers.append({"base_direction": _gap(x, held, gave, items, base_d.tolist()),
                       "own_direction": _gap(x, held, gave, items, own),
                       "own_norm": round(float(own_t.norm()), 4),
                       "cosine_own_base": round(float(own_t @ base_d / (own_t.norm() * base_d.norm())), 4)})
    out = {"name": name, "model": model_path, "position": PRIMARY, "best_layer": meta["best_layer"], "layers": layers,
           "finished_at": now_iso(), "git": git_info()}
    (run.dir / "compare").mkdir(exist_ok=True)
    (run.dir / "compare" / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


# --- report (no torch) ---------------------------------------------------------------------------

def _upper_below_zero(c: dict[str, Any]) -> bool:
    return bool(c.get("ci95")) and c["ci95"][1] < 0


def measurement_2(rows: list[dict[str, Any]], tests: list[dict[str, Any]], *, base_reviewer: str,
                  vllm_rows: list[dict[str, Any]], max_accept_drop: float) -> dict[str, Any]:
    flags = {name: outcome_flags(rows, f"{base_reviewer}+{name}", tests) for name in DIRECTIONS}
    if vllm_rows:
        flags["vllm"] = outcome_flags(vllm_rows, base_reviewer, tests)
    flags = {n: f for n, f in flags.items() if f[WRONG_SET] or f[JEV_RIGHT_SET]}
    out: dict[str, Any] = {"reviewers": {n: {"jev_wrong_accepted_group_a": rate(list(f[WRONG_SET].values())),
                                             "accept_when_right": rate(list(f[JEV_RIGHT_SET].values()))}
                                         for n, f in flags.items()}, "paired": {}}
    pairs = [(NONE, n) for n in (STEER, RANDOM, PROPOSAL)] + [(RANDOM, STEER), (PROPOSAL, STEER)] + \
            [("vllm", n) for n in DIRECTIONS]
    for a, b in pairs:
        if a in flags and b in flags:
            out["paired"][f"{a} -> {b}"] = {
                "jev_wrong_accepted_group_a": paired_change(flags[a][WRONG_SET], flags[b][WRONG_SET]),
                "accept_when_right": paired_change(flags[a][JEV_RIGHT_SET], flags[b][JEV_RIGHT_SET])}
    # The criterion is paired against the unsteered pass on the same stack; vLLM's base only if it's missing.
    baseline = NONE if NONE in flags else "vllm"
    c = out["paired"].get(f"{baseline} -> {STEER}")
    out["causal"] = {"baseline": baseline, "pass": bool(c) and _upper_below_zero(c["jev_wrong_accepted_group_a"])
                     and c["accept_when_right"]["diff"] is not None and c["accept_when_right"]["diff"] >= -max_accept_drop,
                     "rule": f"steering lowers acceptance of Jev's wrong labels on group A (paired 95% interval below 0) "
                             f"and accept-when-right falls by at most {max_accept_drop:.0%}"}
    out["beats_controls"] = {ctrl: _upper_below_zero(out["paired"][f"{ctrl} -> {STEER}"]["jev_wrong_accepted_group_a"])
                             if f"{ctrl} -> {STEER}" in out["paired"] else None for ctrl in (RANDOM, PROPOSAL)}
    return out


def summarize(run: Run, config: E110Config, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tests = run.writer("test_items").rows()
    read = lambda p: json.loads(p.read_text()) if p.exists() else None
    m1, alpha = read(run.dir / "directions.json"), read(run.dir / "alpha.json")
    m2 = measurement_2(run.writer("reviews").rows(), tests, base_reviewer=config.base_reviewer,
                       vllm_rows=[r for r in eval_rows if r["reviewer"] == config.base_reviewer],
                       max_accept_drop=config.max_accept_drop)
    m3 = {}
    for path in sorted((run.dir / "compare").glob("*.json")) if (run.dir / "compare").exists() else []:
        c = json.loads(path.read_text())
        best = c["best_layer"]
        f = outcome_flags(eval_rows, c["name"], tests)[WRONG_SET]
        m3[c["name"]] = {"at_best_layer": c["layers"][best],
                         "base_at_best_layer": m1["gap"][PRIMARY][best] if m1 else None,
                         "jev_wrong_accepted_group_a": rate(list(f.values())) if f else None}
    return {"run_id": run.run_id, "counts": run.manifest().get("counts"), "measurement_1": m1, "alpha": alpha,
            "measurement_2": m2, "measurement_3": m3}


def _num(x: Any, spec: str = ".3f") -> str:
    return "–" if x is None else format(x, spec)


def report(summary: dict[str, Any]) -> str:
    lines = [f"# E.110 report: {summary['run_id']}", "",
             "Activation steering on the reviewer (research_docs/15). Directions are fit on training-pool items only.", ""]
    total = (summary.get("counts") or {}).get("selection", {}).get("total", {})
    if total:
        lines += [f"Data: {total.get('items')} items ({total.get('items_fit')} fit, {total.get('items_heldout')} held out), "
                  f"{total.get('traces_gave_in')} give-in and {total.get('traces_held')} held traces. The direction is "
                  f"fit within items, on the {total.get('items_with_both_fit')} fit items with both a give-in and a held "
                  f"trace; gaps are paired on the {total.get('items_with_both_heldout')} such held-out items.", ""]
    m1 = summary["measurement_1"]
    lines += ["## 1. Is it there?", ""]
    if m1:
        lines += [f"Best layer {m1['best_layer']} (held-out AUC {m1['best_auc']}): **{'found' if m1['found'] else 'not found'}** "
                  f"({m1['rule']}). Steering layers {m1['steer_layers']}.", "",
                  "| Layer | AUC " + " | AUC ".join(POSITIONS) + f" | Probe AUC ({PRIMARY}) | Std. gap ({PRIMARY}) | Norm |",
                  "|---|" + "---|" * (len(POSITIONS) + 3)]
        for layer in range(m1["layers"]):
            lines.append(f"| {layer} | " + " | ".join(_num(m1["auc"][p][layer]) for p in POSITIONS) +
                         f" | {_num(m1['probe_auc'][PRIMARY][layer])} | {_num(m1['gap'][PRIMARY][layer]['std_gap'], '.2f')} | "
                         f"{_num(m1['norm'][PRIMARY][layer], '.1f')} |")
    else:
        lines.append("Not run yet.")
    alpha = summary["alpha"]
    lines += ["", "## α (held-out training items)", ""]
    if alpha:
        lines += [f"Frozen α = {alpha['alpha']:g} at layers {alpha['layers']}"
                  f"{'' if alpha['admissible'] else ' (no α kept accept-when-right within the limit: flagged)'}. "
                  f"Rule: {alpha['rule']}.", "", "| α | Jev's wrong accepted | Jev's right accepted |", "|---|---|---|"]
        lines += [f"| {a} | {_fmt(t['give_in'])} | {_fmt(t['accept'])} |" for a, t in alpha["table"].items()]
    else:
        lines.append("Not frozen yet.")
    m2 = summary["measurement_2"]
    lines += ["", "## 2. Is it causal? (test: group A with Jev's wrong label; the Jev-right subsample)", "",
              "| Reviewer | Jev's wrong accepted (group A) | Jev's right accepted |", "|---|---|---|"]
    lines += [f"| {n} | {_fmt(r['jev_wrong_accepted_group_a'])} | {_fmt(r['accept_when_right'])} |"
              for n, r in m2["reviewers"].items()]
    lines += ["", "| Paired change | Jev's wrong accepted (group A) | Jev's right accepted |", "|---|---|---|"]
    lines += [f"| {k} | {_fmt_change(v['jev_wrong_accepted_group_a'])} | {_fmt_change(v['accept_when_right'])} |"
              for k, v in m2["paired"].items()]
    lines += ["", f"**Causal: {'yes' if m2['causal']['pass'] else 'no'}** against `{m2['causal']['baseline']}` "
                  f"({m2['causal']['rule']}). Beats the controls (paired interval below 0): " +
              ", ".join(f"{c} {'–' if v is None else 'yes' if v else 'no'}" for c, v in m2["beats_controls"].items()) + ".",
              "", "`vllm` is the base's E.106 review on vLLM, for reference; `none` is the unsteered pass here."]
    lines += ["", "## 3. Did training remove it?", ""]
    if summary["measurement_3"]:
        lines += ["At the base's best layer, held-out traces. A smaller gap along the base's direction means training "
                  "changed the mechanism; the same gap with less giving in means it changed the output only.", "",
                  "| Model | Std. gap, base direction | AUC | Std. gap, own direction | AUC | Cosine own/base | "
                  "Jev's wrong accepted (group A, E.106 eval) |", "|---|---|---|---|---|---|---|"]
        base = next(iter(summary["measurement_3"].values()))["base_at_best_layer"]
        if base:
            lines.append(f"| base | {_num(base['std_gap'], '.2f')} | {_num(base['auc'])} | (same) | | 1 | |")
        for name, c in summary["measurement_3"].items():
            b, o = c["at_best_layer"]["base_direction"], c["at_best_layer"]["own_direction"]
            lines.append(f"| {name} | {_num(b['std_gap'], '.2f')} | {_num(b['auc'])} | {_num(o['std_gap'], '.2f')} | "
                         f"{_num(o['auc'])} | {_num(c['at_best_layer']['cosine_own_base'])} | {_fmt(c['jev_wrong_accepted_group_a'])} |")
    else:
        lines.append("No trained model compared yet.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    steps = {name: sub.add_parser(name) for name in ("build", "capture", "directions", "select-alpha", "steer",
                                                     "compare", "report")}
    for p in steps.values():
        p.add_argument("--run-id", required=True)
        p.add_argument("--results-dir", default=str(RESULTS_DIR))
    for name in ("capture", "select-alpha", "steer", "compare"):
        steps[name].add_argument("--model", required=True, help="a bf16 checkpoint directory")
    for name in ("capture", "compare"):
        # Tokens per forward pass (batch x longest sequence); gpt-oss's inference path runs every expert on every token.
        steps[name].add_argument("--token-budget", type=int, default=16384)
    for name in ("select-alpha", "steer"):
        steps[name].add_argument("--batch-size", type=int, default=16)
    steps["capture"].add_argument("--tag", default=BASE_TAG)
    steps["steer"].add_argument("--directions", nargs="+", choices=DIRECTIONS, default=list(DIRECTIONS))
    steps["compare"].add_argument("--name", required=True, help="the trained model's alias, e.g. consist_review")
    steps["report"].add_argument("--eval-run", help="the E.106-style evaluation run (default: the config's e106_run)")
    args = parser.parse_args()

    run = Run(args.run_id, args.results_dir)
    runs_dir = Path(args.results_dir) / "runs"
    if args.step == "build":
        counts = build(E110Config(), run, runs_dir)
        print(json.dumps(counts, indent=2))
        return
    if "config" not in run.manifest():
        raise SystemExit(f"{run.dir} has no build; run `build` first (on the machine with E.105's traces)")
    config = E110Config(**run.manifest()["config"])
    if args.step == "capture":
        capture(run, config, args.model, args.tag, token_budget=args.token_budget)
    elif args.step == "directions":
        out = directions(run, config)
        print(f"best layer {out['best_layer']}, held-out AUC {out['best_auc']}: {'found' if out['found'] else 'not found'}")
    elif args.step == "select-alpha":
        print(f"frozen α = {select_alpha(run, config, args.model, batch_size=args.batch_size)['alpha']}")
    elif args.step == "steer":
        steer(run, config, args.model, args.directions, batch_size=args.batch_size)
    elif args.step == "compare":
        compare(run, config, args.model, args.name, token_budget=args.token_budget)
    if args.step in ("steer", "compare", "report"):
        eval_rows = JsonlWriter(runs_dir / (getattr(args, "eval_run", None) or config.e106_run) / "reviews.jsonl").rows()
        summary = summarize(run, config, [r for r in eval_rows if r["status"] == "ok" and r["parse_ok"]])
        (run.dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        (run.dir / "report.md").write_text(report(summary))
        print((run.dir / "report.md").read_text())


if __name__ == "__main__":
    main()
