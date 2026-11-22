"""E.113 step 1: build the blind review package for a Claude teacher (no API calls).

The same 300 training-pool items as E.112's screen. Each becomes one file holding exactly the review
prompt gpt-oss-20b and the API teachers saw (policy, item, Jev's label, E.112's written-reasoning
instruction), under a random ID. What never goes into the blind folder:
  - the gold label, the original item ID, the pool, the stratum, whether Jev was right;
  - anything else from the repo.
The key (blind ID -> pool, item ID) is written to the run folder, which no reviewer is pointed at.
The package is checked before it's used: no file may contain a source ID, and every field is accounted for.

    uv run python research_analyses/E113_claude_teacher/build_blind.py --blind-dir <dir> --run-id e113-v1-<date>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from pathlib import Path

from sysone.cascade.items import load_items
from sysone.cascade.reviewer import Reviewer
from sysone.cascade.tasks import TASKS
from sysone.clients.base import GuardedClient, ModelSpec
from sysone.budget.pricing import Price
from sysone.experiments.e112 import WRITTEN_PREFIX, E112Config, select
from sysone.registry import CONFIG_DIR, RESULTS_DIR
from sysone.runs import JsonlWriter


def render(request) -> str:
    assert len(request.messages) == 1 and request.messages[0].role == "user"
    return request.messages[0].content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-dir", required=True, help="outside the repo; the only thing reviewers see")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    config = E112Config.load(CONFIG_DIR / "experiments" / "e112.yaml")
    runs = RESULTS_DIR / "runs"
    mix = JsonlWriter(runs / config.source_run / "mix.jsonl").rows()
    traces = [t for t in JsonlWriter(runs / config.traces_run / "traces.jsonl").rows() if t["status"] == "ok"]
    picked = select(mix, traces, config)
    items = {(s, i.item_id): i for s in config.samples for i in load_items(s)[0]}

    salt = secrets.token_hex(16)
    blind_dir = Path(args.blind_dir)
    item_dir = blind_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=False)
    run_dir = runs / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # A placeholder client: only build_request is used, nothing is sent.
    dummy = GuardedClient(ModelSpec("blind", "bedrock", "none", Price(0.0, 0.0)), None, None, cache=None)
    key, source_text = [], {}
    for p in picked:
        item = items[(p["sample"], p["item_id"])]
        bid = "r" + hashlib.sha256(f"{salt}:{p['sample']}:{p['item_id']}".encode()).hexdigest()[:10]
        reviewer = Reviewer(dummy, TASKS[item.task])
        text = render(reviewer.build_request(item.text, p["proposed_label"], task_prefix=WRITTEN_PREFIX))
        (item_dir / f"{bid}.md").write_text(text)
        source_text[bid] = item.text.lower()
        key.append({"blind_id": bid, "sample": p["sample"], "item_id": p["item_id"], "proposed_label": p["proposed_label"],
                    "labels": list(TASKS[item.task].labels), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})

    # Checks before any reviewer sees the package.
    files = sorted(item_dir.iterdir())
    assert len(files) == len(picked) == len({k["blind_id"] for k in key}), "blind IDs must be unique, one file per item"
    assert [f.name for f in blind_dir.iterdir()] == ["items"], "the blind folder holds only the item files"
    id_pattern = re.compile(r"\b(cc-\d+|wgtrain-[pr]-\d+)\b")
    for f in files:
        body = f.read_text()
        assert not id_pattern.search(body), f"{f.name} contains a source item ID"
        for word in ("gold", "jev_correct", "stratum", "unsolved", "civil_toxicity_train", "wildguard_prompt_train"):
            # Allowed only where the item's own text contains it (e.g. "golden"), never from metadata.
            assert word not in body.lower() or word in source_text[f.stem], f"{f.name} contains {word!r}"
    key.sort(key=lambda k: k["blind_id"])
    (run_dir / "key.json").write_text(json.dumps({"salt": salt, "blind_dir": str(blind_dir), "items": key}, indent=1) + "\n")
    print(f"{len(files)} blind item files in {item_dir}; key in {run_dir / 'key.json'}")
    print(json.dumps(sorted(k["blind_id"] for k in key)))


if __name__ == "__main__":
    main()
