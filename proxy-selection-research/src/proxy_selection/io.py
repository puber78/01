"""Guarded filesystem writes, append-only artifact ledger and manifest I/O.

Two invariants are enforced here and counted for the EXAM01 acceptance metrics:

* ``original_overwrite_count`` -- how many times an EXISTING file was replaced
  without first archiving it.  The only allowed replacement path is
  :func:`replace_with_archive`, which copies the previous bytes into
  ``ops/archive/<relpath>/<sha256>.<ts>.bak`` first.  All other writes go through
  :func:`write_new`, which refuses to touch an existing path.
* ``artifact_ledger`` -- every produced file is recorded with its SHA-256, size,
  producing run id and kind, so a later audit can re-verify the whole tree.

Nothing in this module deletes files.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .hashing import sha256_bytes, sha256_file, sha256_json
from .contracts.schema import REQUIRED_COLUMNS, validate_rows


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ImmutableFileError(RuntimeError):
    """Raised when a write would overwrite an existing artifact."""


class Ledger:
    """Append-only CSV ledgers under ops/."""

    def __init__(self, ops_dir: str | Path):
        self.ops = Path(ops_dir)
        self.artifacts_csv = self.ops / "manifests" / "artifacts.csv"
        self.violations_csv = self.ops / "manifests" / "guard_violations.csv"
        for p in (self.artifacts_csv, self.violations_csv):
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                with p.open("w", newline="", encoding="utf-8") as fh:
                    if p is self.artifacts_csv:
                        csv.writer(fh).writerow(
                            ["recorded_at", "path", "sha256", "bytes", "kind", "run_id", "note"]
                        )
                    else:
                        csv.writer(fh).writerow(
                            ["recorded_at", "path", "attempted_bytes", "reason", "run_id"]
                        )

    def record_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        run_id: str,
        note: str = "",
        sha256: str | None = None,
    ) -> str:
        p = Path(path)
        digest = sha256 or sha256_file(p)
        with self.artifacts_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [utcnow(), str(p), digest, p.stat().st_size, kind, run_id, note]
            )
        return digest

    def record_violation(self, path: str | Path, *, reason: str, run_id: str) -> None:
        with self.violations_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([utcnow(), str(path), "", reason, run_id])

    def count_violations(self) -> int:
        with self.violations_csv.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        return max(0, len(rows) - 1)

    def count_artifacts(self) -> int:
        with self.artifacts_csv.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        return max(0, len(rows) - 1)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_new(
    path: str | Path,
    data: bytes | str,
    *,
    ledger: Ledger | None = None,
    run_id: str = "unlogged",
    kind: str = "artifact",
    note: str = "",
) -> str:
    """Create a NEW file. Never overwrites. Returns the sha256 of the bytes."""
    p = Path(path)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    if p.exists():
        if ledger is not None:
            ledger.record_violation(p, reason="overwrite_refused", run_id=run_id)
        raise ImmutableFileError(f"refusing to overwrite existing artifact: {p}")
    _atomic_write_bytes(p, payload)
    digest = sha256_bytes(payload)
    if ledger is not None:
        ledger.record_artifact(p, kind=kind, run_id=run_id, note=note, sha256=digest)
    return digest


def replace_with_archive(
    path: str | Path,
    data: bytes | str,
    *,
    archive_root: str | Path,
    ledger: Ledger | None = None,
    run_id: str = "unlogged",
    kind: str = "artifact",
    note: str = "",
) -> str:
    """Replace a file, archiving the previous bytes first (never destructive)."""
    p = Path(path)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    if p.exists():
        old = p.read_bytes()
        old_sha = sha256_bytes(old)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = Path(archive_root) / p.parent.name / f"{p.name}.{old_sha[:12]}.{ts}.bak"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(old)
        if ledger is not None:
            ledger.record_artifact(dest, kind="archive", run_id=run_id, note=f"pre-replace of {p}")
    _atomic_write_bytes(p, payload)
    digest = sha256_bytes(payload)
    if ledger is not None:
        ledger.record_artifact(p, kind=kind, run_id=run_id, note=note, sha256=digest)
    return digest


def write_json(
    path: str | Path,
    obj: Any,
    *,
    ledger: Ledger | None = None,
    run_id: str = "unlogged",
    kind: str = "json",
    note: str = "",
    replace: bool = False,
    archive_root: str | Path | None = None,
) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if replace and Path(path).exists():
        if archive_root is None:
            raise ValueError("replace=True requires archive_root")
        return replace_with_archive(
            path, text, archive_root=archive_root, ledger=ledger, run_id=run_id, kind=kind, note=note
        )
    return write_new(path, text, ledger=ledger, run_id=run_id, kind=kind, note=note)


def write_parquet(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    table: str,
    *,
    ledger: Ledger | None = None,
    run_id: str = "unlogged",
    note: str = "",
    strict: bool = True,
) -> str:
    """Validate rows against the table schema, then write a parquet file."""
    data = [dict(r) for r in rows]
    if strict:
        errs = validate_rows(table, data)
        if errs:
            raise ValueError(f"schema validation failed for {table}: {errs[:5]}")
    df = pd.DataFrame(data, columns=list(REQUIRED_COLUMNS[table].keys()))
    p = Path(path)
    if p.exists():
        if ledger is not None:
            ledger.record_violation(p, reason="overwrite_refused_parquet", run_id=run_id)
        raise ImmutableFileError(f"refusing to overwrite existing artifact: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    digest = sha256_file(p)
    if ledger is not None:
        ledger.record_artifact(
            p, kind=f"parquet:{table}", run_id=run_id, note=f"{len(df)} rows; {note}", sha256=digest
        )
    return digest


def read_parquet(path: str | Path, table: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    expected = list(REQUIRED_COLUMNS[table].keys())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def snapshot_hashes(root: str | Path, pattern: str = "**/*", exclude_dirs: Iterable[str] = (".venv", "tmp", "ops/archive")) -> dict[str, str]:
    """Hash a tree (relative path -> sha256), skipping volatile directories."""
    root = Path(root)
    excl = [root / d for d in exclude_dirs]
    out: dict[str, str] = {}
    for p in sorted(root.glob(pattern)):
        if not p.is_file():
            continue
        if any(str(p).startswith(str(e)) for e in excl):
            continue
        out[str(p.relative_to(root)).replace("\\", "/")] = sha256_file(p)
    return out
