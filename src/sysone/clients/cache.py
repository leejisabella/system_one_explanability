"""On-disk response cache, so re-running an experiment does not pay twice."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from sysone.types import ChatRequest, ChatResponse, Usage


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self._conn.commit()

    @staticmethod
    def key(provider: str, model: str, request: ChatRequest) -> str:
        fields = dataclasses.asdict(request)
        fields.pop("tags")
        payload = json.dumps({"provider": provider, "model": model, "request": fields}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> ChatResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        body = json.loads(row[0])
        body["usage"] = Usage(**body["usage"])
        # A cache hit costs nothing; the original cost is already in the ledger.
        return ChatResponse(**{**body, "cost_usd": 0.0, "cached": True})

    def put(self, key: str, response: ChatResponse) -> None:
        body = json.dumps(dataclasses.asdict(response))
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO responses (key, body) VALUES (?, ?)", (key, body))
            self._conn.commit()
