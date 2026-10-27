"""Structured run logging (JSONL) and the append-only command ledger.

Every script in this project opens a :class:`RunLogger` with a fresh run id and
records: environment facts, input hashes, each artifact written, every failure,
and a closing summary.  ``command_ledger.csv`` mirrors the shell commands that
produced the artifacts (command text, cwd, start/end, exit status, log path).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts.ids import new_run_id


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def environment_facts() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "env_HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@dataclass
class RunLogger:
    """JSONL logger with one file per run."""

    root: Path
    run_id: str
    events: list[dict] = field(default_factory=list)
    _path: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.log_dir = self.root / "ops" / "logs" / "runs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / f"{self.run_id}.jsonl"

    @classmethod
    def start(cls, root: str | Path, *, note: str = "") -> "RunLogger":
        logger = cls(Path(root), new_run_id())
        logger.event("run_start", note=note, environment=environment_facts())
        return logger

    def event(self, event: str, **fields: Any) -> None:
        rec = {"ts": utcnow(), "run_id": self.run_id, "event": event, **fields}
        self.events.append(rec)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def input_hashes(self, mapping: Mapping[str, str]) -> None:
        self.event("input_hashes", hashes=dict(mapping))

    def artifact(self, path: str | Path, sha256: str, kind: str = "artifact", note: str = "") -> None:
        self.event("artifact", path=str(path), sha256=sha256, kind=kind, note=note)

    def failure(self, where: str, error: str, **fields: Any) -> None:
        self.event("failure", where=where, error=error, **fields)

    def finish(self, status: str = "PASS", **fields: Any) -> None:
        self.event("run_end", status=status, **fields)

    @property
    def path(self) -> Path:
        assert self._path is not None
        return self._path


class CommandLedger:
    """Append-only record of executed shell commands and their exit status."""

    COLUMNS = [
        "recorded_at",
        "requested_at",
        "duration_s",
        "cwd",
        "command",
        "exit_code",
        "log_path",
        "note",
    ]

    def __init__(self, root: str | Path):
        self.path = Path(root) / "ops" / "logs" / "command_ledger.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.COLUMNS)

    def append(
        self,
        command: str,
        *,
        exit_code: int,
        cwd: str = "",
        log_path: str = "",
        duration_s: float | None = None,
        note: str = "",
        requested_at: str | None = None,
    ) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [
                    utcnow(),
                    requested_at or utcnow(),
                    f"{duration_s:.3f}" if duration_s is not None else "",
                    cwd or os.getcwd(),
                    command.strip(),
                    exit_code,
                    log_path,
                    note,
                ]
            )


class timed:  # noqa: N801 - used as a context manager
    """Small helper: ``with timed() as t: ...`` then ``t.elapsed``."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def __enter__(self) -> "timed":
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
