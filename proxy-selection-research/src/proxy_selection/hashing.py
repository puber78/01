"""Hashing utilities: canonical JSON, file digests, config fingerprints.

All hashes in this project are lowercase hex SHA-256 unless explicitly stated.
Canonical JSON uses sorted keys, compact separators and UTF-8 encoding so that
the same logical record always yields the same digest across platforms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

CHUNK = 1024 * 1024


def canonical_json(obj: Any) -> str:
    """Return a deterministic JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    """Stream a file through SHA-256. Raises FileNotFoundError if missing."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_many(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash several files, keyed by the path as given."""
    return {str(p): sha256_file(p) for p in paths}


def combine_hashes(hashes: Iterable[str]) -> str:
    """Order-sensitive combination; callers must pass a canonical order."""
    h = hashlib.sha256()
    for item in hashes:
        h.update(item.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def hash_mapping(mapping: Mapping[str, Any]) -> str:
    return sha256_json({str(k): v for k, v in mapping.items()})
