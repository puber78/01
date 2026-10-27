"""Identifier contracts for sources, lineage, display assets, captures, samples,
models, rules and runs.

Design rules
------------
1. Every identifier carries a type prefix so a stray id cannot be silently used
   in the wrong relation.
2. Content-derived ids (source, split group, display asset, sample, model, rule)
   are deterministic UUIDv5 values over a canonical description: re-ingesting the
   same record yields the same id, so accidental duplication is detectable.
3. Event ids (captures, runs) are time/uuid4 based because a physical capture or
   an execution is a new event even when metadata repeats.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

NAMESPACE = uuid.UUID("6f1c2d54-0c1e-5b7a-9a3e-2d5f8c4b1a70")

PREFIXES = {
    "source": "src",
    "split_group": "grp",
    "display_asset": "dsp",
    "capture": "cap",
    "sample": "smp",
    "model": "mdl",
    "rule": "rul",
    "run": "run",
}

_ID_RE = re.compile(r"^(src|grp|dsp|cap|smp|mdl|rul|run)_[0-9a-zA-Z_]{6,48}$")


class IdError(ValueError):
    """Raised when an identifier does not satisfy the declared contract."""


def _mint(kind: str, payload: Mapping[str, Any]) -> str:
    if kind not in PREFIXES:
        raise IdError(f"unknown id kind: {kind}")
    canon = _canonical_payload(payload)
    digest = uuid.uuid5(NAMESPACE, f"{kind}:{canon}").hex[:24]
    return f"{PREFIXES[kind]}_{digest}"


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    from ..hashing import canonical_json

    return canonical_json({str(k): v for k, v in payload.items()})


def mint_source_id(*, lineage_key: str, label: int, provenance_uri: str) -> str:
    return _mint("source", {"lineage_key": lineage_key, "label": int(label), "provenance_uri": provenance_uri})


def mint_split_group_id(*, lineage_key: str, generation_family: str) -> str:
    return _mint("split_group", {"lineage_key": lineage_key, "generation_family": generation_family})


def mint_display_asset_id(*, source_id: str, display_spec_hash: str) -> str:
    require_id(source_id, "source")
    return _mint("display_asset", {"source_id": source_id, "display_spec_hash": display_spec_hash})


def mint_sample_id(*, branch: str, transform_id: str, parent_key: str) -> str:
    """branch: 'digital' | 'physical'. parent_key is display_asset_id (digital)
    or capture_id (physical)."""
    if branch not in {"digital", "physical"}:
        raise IdError(f"unknown branch: {branch}")
    require_id(parent_key, "display_asset" if branch == "digital" else "capture")
    return _mint("sample", {"branch": branch, "transform_id": transform_id, "parent": parent_key})


def mint_model_id(*, repo: str, revision: str, weight_sha256: str) -> str:
    return _mint("model", {"repo": repo, "revision": revision, "weight_sha256": weight_sha256})


def mint_rule_id(*, rule_name: str, spec: Mapping[str, Any]) -> str:
    return _mint("rule", {"rule_name": rule_name, "spec": dict(spec)})


def new_capture_id() -> str:
    return f"cap_{uuid.uuid4().hex[:24]}"


def new_run_id(now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def require_id(value: str, kind: str) -> str:
    """Validate an identifier and its expected kind."""
    if kind not in PREFIXES:
        raise IdError(f"unknown id kind: {kind}")
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise IdError(f"malformed id: {value!r}")
    if not value.startswith(PREFIXES[kind] + "_"):
        raise IdError(f"expected {kind} id ({PREFIXES[kind]}_*), got {value!r}")
    return value


@dataclass(frozen=True)
class EntityRefs:
    """A validated bundle of references used to assemble one sample row."""

    source_id: str
    split_group_id: str
    display_asset_id: str
    model_id: str
    run_id: str

    def __post_init__(self) -> None:
        require_id(self.source_id, "source")
        require_id(self.split_group_id, "split_group")
        require_id(self.display_asset_id, "display_asset")
        require_id(self.model_id, "model")
        require_id(self.run_id, "run")
