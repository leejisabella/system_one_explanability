"""Per-run results directory: manifest plus append-only JSONL outputs.

results/runs/<run_id>/
  manifest.json   what was run, with which code, configs, models and benchmark
  ledger.jsonl    every outside API attempt and its cost (written by the cost guard)
  <name>.jsonl    outputs, appended as they arrive so a crash loses nothing
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Hashable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sysone.registry import REPO_ROOT


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(status)}


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def done_keys(self, key: Callable[[dict[str, Any]], Hashable]) -> set[Hashable]:
        return {key(row) for row in self.rows()}


class Run:
    def __init__(self, run_id: str, results_dir: str | Path):
        self.run_id = run_id
        self.dir = Path(results_dir) / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        manifest = {**self.manifest(), "run_id": self.run_id, **fields}
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return manifest

    def writer(self, name: str) -> JsonlWriter:
        return JsonlWriter(self.dir / f"{name}.jsonl")
