"""Freezing and loading versioned benchmarks (benchmarks/<name>/cases.jsonl + meta.json)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sysone.benchmark.cases import Case


def freeze_benchmark(cases: list[Case], out_dir: str | Path, meta: dict[str, Any], *, force: bool = False) -> Path:
    out_dir = Path(out_dir)
    cases_path = out_dir / "cases.jsonl"
    if cases_path.exists() and not force:
        raise FileExistsError(f"{cases_path} exists; frozen benchmarks are not overwritten (use a new name)")
    out_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in sorted(cases, key=lambda c: c.case_id))
    cases_path.write_text(body)
    full_meta = {
        **meta,
        "name": out_dir.name,
        "n_cases": len(cases),
        "n_variants": sum(len(c.variants) for c in cases),
        "case_type_counts": dict(sorted(Counter(str(c.case_type) for c in cases).items())),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(full_meta, indent=2, sort_keys=True) + "\n")
    return out_dir


def load_benchmark(path: str | Path) -> tuple[list[Case], dict[str, Any]]:
    """Load a frozen benchmark, refusing it if the cases file no longer matches its recorded hash.

    Benchmarks whose meta names a domain other than refund load as DomainCase."""
    path = Path(path)
    body = (path / "cases.jsonl").read_text()
    meta = json.loads((path / "meta.json").read_text())
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: cases.jsonl hash {digest} does not match meta.json {meta['sha256']}")
    case_cls = Case
    if meta.get("domain", "refund") != "refund":
        from sysone.benchmark.domains.base import DomainCase

        case_cls = DomainCase
    return [case_cls.from_dict(json.loads(line)) for line in body.splitlines() if line.strip()], meta
