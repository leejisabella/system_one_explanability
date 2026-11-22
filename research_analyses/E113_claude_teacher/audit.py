"""E.113 leakage audit: every tool call every blind reviewer made, from the workflow's transcripts.

Passes only if each reviewer agent:
  - was given exactly one task file (from its first prompt) inside the blind folder;
  - made no tool call other than Read on that one file (and the StructuredOutput reply);
  - ended with a structured answer.
Any other Read, any Bash/Grep/Glob/Write/WebFetch, or a path outside the blind folder fails the audit.

    uv run python research_analyses/E113_claude_teacher/audit.py <transcript dir> [<transcript dir> ...] --blind-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TASK = re.compile(r"Your task file: (\S+\.md)")
ALLOWED_REPLY = {"StructuredOutput"}


def messages(path: Path):
    for line in path.read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def content_blocks(row: dict):
    msg = row.get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        for b in c:
            if isinstance(b, dict):
                yield msg.get("role") or row.get("type"), b
    elif isinstance(c, str):
        yield msg.get("role") or row.get("type"), {"type": "text", "text": c}


def audit_agent(path: Path, blind_dir: str) -> dict:
    task, calls, problems = None, [], []
    for row in messages(path):
        for role, b in content_blocks(row):
            if role == "user" and b.get("type") == "text" and task is None:
                m = TASK.search(b.get("text", ""))
                if m:
                    task = m.group(1)
            if b.get("type") == "tool_use":
                calls.append((b.get("name"), b.get("input") or {}))
    if task is None:
        return {"agent": path.name, "task": None, "calls": calls, "problems": ["no task file in the first prompt"]}
    if not task.startswith(blind_dir.rstrip("/") + "/"):
        problems.append(f"task file outside the blind folder: {task}")
    reads = 0
    for name, inp in calls:
        if name == "Read":
            reads += 1
            if inp.get("file_path") != task:
                problems.append(f"read another file: {inp.get('file_path')}")
        elif name not in ALLOWED_REPLY:
            problems.append(f"used tool {name}: {json.dumps(inp)[:200]}")
    if not any(n in ALLOWED_REPLY for n, _ in calls):
        problems.append("no structured answer")
    if reads == 0:
        problems.append("never read its task file")
    return {"agent": path.name, "task": Path(task).stem, "calls": [n for n, _ in calls], "problems": problems}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+")
    ap.add_argument("--blind-dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for d in args.transcripts:
        for path in sorted(Path(d).rglob("agent-*.jsonl")):
            results.append(audit_agent(path, args.blind_dir))
    tasks = Counter(r["task"] for r in results if r["task"])
    bad = [r for r in results if r["problems"]]
    summary = {"agents": len(results), "distinct_tasks": len(tasks),
               "tasks_reviewed_more_than_once": {t: n for t, n in tasks.items() if n > 1},
               "tool_calls": dict(Counter(c for r in results for c in r["calls"])),
               "agents_with_problems": len(bad), "problems": bad[:50],
               "passed": not bad}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "agents": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
