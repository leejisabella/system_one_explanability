"""Append-only JSONL log of every outside API request and what it cost."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Ledger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()

    @classmethod
    def for_run(cls, results_dir: str | Path, run_id: str) -> Ledger:
        return cls(Path(results_dir) / "runs" / run_id / "ledger.jsonl", run_id)

    def append(self, **entry: Any) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id, **entry}
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(line + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def total_spent(self) -> float:
        """Spend already recorded for this run, so a resumed run keeps its budget."""
        return sum(e.get("cost_usd", 0.0) for e in self.entries())
