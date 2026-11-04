"""Training records for Part 2 (E.105's export), in TRL's conversational formats.

Every prompt is built by sysone.cascade.reviewer.Reviewer, the same code that builds the
evaluation prompts, so a trained reviewer sees at test time exactly the prompt it was trained on.

SFT records are prompt-completion pairs, so the loss falls on the reviewer's turn only:
  {"prompt": [{"role": "user", "content": ...}],
   "completion": [{"role": "assistant", "thinking": <reasoning>, "content": <JSON answer>}],
   "meta": {...}}
gpt-oss's chat template renders "thinking" into the analysis channel and "content" into the
final channel. The answer is the canonical JSON for the gold label (and the matching verdict),
whatever spacing the sampled reply used.

DPO records pair a correct trace with a wrong one for the same prompt, both from the reviewer
itself (the policy being trained): {"prompt", "chosen", "rejected", "meta"}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sysone.cascade.reviewer import ACCEPT, Reviewer
from sysone.cascade.tasks import TASKS

REVIEWER_SOURCE, TEACHER_SOURCE = "reviewer", "teacher"
_CUT_OFF = {"max_tokens", "length"}


@dataclass(frozen=True)
class _Spec:
    provider: str = "bedrock"  # no JSON schema in the request, as on Bedrock and the local server


@dataclass(frozen=True)
class _Client:
    spec: _Spec = _Spec()


def prompt_messages(task: str, text: str, proposed_label: str | None) -> list[dict[str, str]]:
    """The reviewer's prompt for an item, alone (no proposal) or reviewing a proposed label."""
    request = Reviewer(_Client(), TASKS[task]).build_request(text, proposed_label)  # type: ignore[arg-type]
    return [{"role": m.role, "content": m.content} for m in request.messages]


def answer_json(target_label: str, proposed_label: str | None) -> str:
    """The reply the reviewer should give: the gold label, and in review the verdict it implies."""
    if proposed_label is None:
        return json.dumps({"final_label": target_label})
    verdict = ACCEPT if proposed_label == target_label else "override"
    return json.dumps({"verdict": verdict, "final_label": target_label})


def correct(row: dict[str, Any]) -> bool:
    """A trace that reaches the gold label (with a verdict that agrees, in review)."""
    return (row["status"] == "ok" and row["parse_ok"] and row["final_label"] == row["label"]
            and row.get("consistent") is not False)


def usable(row: dict[str, Any]) -> bool:
    """A trace that can be trained on: parsed, with its reasoning saved, and not cut off."""
    return (row["status"] == "ok" and row["parse_ok"] and bool(row.get("reasoning"))
            and str(row.get("finish_reason") or "").lower() not in _CUT_OFF)


def choose(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The trace to train on: correct and usable, the reviewer's own before the teacher's, earliest first."""
    ok = [t for t in traces if correct(t) and usable(t)]
    return min(ok, key=lambda t: (t["source"] != REVIEWER_SOURCE, t["sample_index"]), default=None)


def rejected(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A wrong trace from the reviewer itself, for a DPO pair. In review, one that accepted a wrong
    proposal (deference) is preferred."""
    bad = [t for t in traces if t["source"] == REVIEWER_SOURCE and usable(t) and not correct(t)]
    return min(bad, key=lambda t: (t.get("final_label") != t.get("proposed_label"), t["sample_index"]), default=None)


def _assistant(trace: dict[str, Any], content: str) -> dict[str, str]:
    return {"role": "assistant", "thinking": trace["reasoning"], "content": content}


def sft_record(mix_row: dict[str, Any], text: str, proposed_label: str | None, trace: dict[str, Any],
               *, arm: str) -> dict[str, Any]:
    return {
        "prompt": prompt_messages(mix_row["task"], text, proposed_label),
        "completion": [_assistant(trace, answer_json(mix_row["target_label"], proposed_label))],
        "meta": {"arm": arm, "sample": mix_row["sample"], "item_id": mix_row["item_id"], "group": mix_row["group"],
                 "format": "alone" if proposed_label is None else "review", "proposed_label": proposed_label,
                 "label": mix_row["target_label"], "trace_source": trace["source"],
                 "trace_sample_index": trace["sample_index"], "split": mix_row["split"]},
    }


def dpo_record(mix_row: dict[str, Any], text: str, good: dict[str, Any], bad: dict[str, Any]) -> dict[str, Any]:
    proposed = mix_row["proposed_label"]
    bad_answer = json.dumps({"verdict": bad.get("verdict"), "final_label": bad["final_label"]}) if proposed is not None \
        else json.dumps({"final_label": bad["final_label"]})
    return {
        "prompt": prompt_messages(mix_row["task"], text, proposed),
        "chosen": [_assistant(good, answer_json(mix_row["target_label"], proposed))],
        "rejected": [_assistant(bad, bad_answer)],
        "meta": {"arm": "review", "sample": mix_row["sample"], "item_id": mix_row["item_id"], "group": mix_row["group"],
                 "proposed_label": proposed, "label": mix_row["target_label"], "split": mix_row["split"],
                 "rejected_deferred": proposed is not None and bad["final_label"] == proposed},
    }
