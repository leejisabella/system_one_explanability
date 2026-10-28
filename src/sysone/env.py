"""Load secrets from the repo's .env file (gitignored) into the environment.

Lines are KEY=value; blank lines and # comments are skipped. Variables already set in
the environment win, and blank values are ignored, so an unfilled field never hides a
key set in the shell or leaves an empty credential for boto3 to trip on.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value
