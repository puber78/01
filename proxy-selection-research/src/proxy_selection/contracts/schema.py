"""Row schemas and validation for the research manifests.

The minimal per-image scoring row follows the execution outline §7.2 exactly:

    run_id, sample_id, source_id/private_join_id, split_role, environment_id,
    model_id, image_branch, transform_id, raw_score, score_orientation,
    checkpoint_hash, preprocess_hash, status, failure_code, is_fixture

Every table has required columns and per-column validators.  Validation is
strict by default: a missing required field is an error, never ``None`` filler,
and a missing score is represented by ``status=failed`` plus a ``failure_code``
plus ``raw_score=None`` -- never by 0 or 0.5.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

STATUSES = {"ok", "failed", "skipped"}
ORIENTATIONS = {"higher_is_ai", "higher_is_real"}
BRANCHES = {"digital", "physical"}
FAILURE_CODES = {
    None,
    "FILE_MISSING",
    "DECODE_ERROR",
    "EXIF_ERROR",
    "PREPROCESS_ERROR",
    "MODEL_ERROR",
    "OOM",
    "UNSUPPORTED_FORMAT",
    "ORIENTATION_UNDETERMINED",
    "NON_FINITE_SCORE",
    "TIMEOUT",
    "OTHER",
}


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and len(v) > 0


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_float(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_label(v: Any) -> bool:
    return v in (0, 1) and not isinstance(v, bool)


VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "str": _is_str,
    "int": _is_int,
    "float": _is_float,
    "bool": _is_bool,
    "label": _is_label,
}


REQUIRED_COLUMNS: dict[str, dict[str, str]] = {
    "sources": {
        "source_id": "str",
        "split_group_id": "str",
        "lineage_root_id": "str",
        "label": "label",
        "content_category": "str",
        "generator_id": "str",
        "provenance_uri": "str",
        "provenance_sha256": "str",
        "license_id": "str",
        "license_evidence_sha256": "str",
        "label_evidence_sha256": "str",
        "original_file_sha256": "str",
        "ingested_at": "str",
        "ingest_run_id": "str",
        "notes": "str",
    },
    "splits": {
        "split_group_id": "str",
        "source_id": "str",
        "split_role": "str",
        "assigned_at": "str",
        "assignment_hash": "str",
        "assignment_run_id": "str",
    },
    "display_assets": {
        "display_asset_id": "str",
        "source_id": "str",
        "display_file_sha256": "str",
        "width": "int",
        "height": "int",
        "color_space": "str",
        "display_spec_hash": "str",
        "created_at": "str",
        "run_id": "str",
    },
    "captures": {
        "capture_id": "str",
        "display_asset_id": "str",
        "session_id": "str",
        "attempt_index": "int",
        "is_planned_repeat": "bool",
        "capture_file_sha256": "str",
        "device_id": "str",
        "qc_status": "str",
        "failure_code": "str",
        "captured_at": "str",
        "run_id": "str",
    },
    "scores": {
        "run_id": "str",
        "sample_id": "str",
        "source_id": "str",
        "split_role": "str",
        "environment_id": "str",
        "model_id": "str",
        "image_branch": "str",
        "transform_id": "str",
        "raw_score": "float_or_none",
        "score_orientation": "str",
        "checkpoint_hash": "str",
        "preprocess_hash": "str",
        "status": "str",
        "failure_code": "str_or_none",
        "is_fixture": "bool",
    },
    "selections": {
        "rule_id": "str",
        "budget": "int",
        "replay_id": "str",
        "calibration_ids_hash": "str",
        "digital_ids_hash": "str",
        "fitted_parameters_hash": "str",
        "selected_model_id": "str",
        "frozen_at": "str",
    },
}


def _validate_score_row(row: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    if row.get("status") not in STATUSES:
        errs.append(f"bad status: {row.get('status')!r}")
    if row.get("image_branch") not in BRANCHES:
        errs.append(f"bad image_branch: {row.get('image_branch')!r}")
    if row.get("score_orientation") not in ORIENTATIONS:
        errs.append(f"bad score_orientation: {row.get('score_orientation')!r}")
    if row.get("failure_code") not in FAILURE_CODES:
        errs.append(f"bad failure_code: {row.get('failure_code')!r}")
    raw = row.get("raw_score")
    if row.get("status") == "ok":
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(float(raw)):
            errs.append("status=ok requires a finite numeric raw_score")
        if not (0.0 <= float(raw) <= 1.0):
            errs.append("raw_score must be a probability in [0,1]")
        if row.get("failure_code") is not None:
            errs.append("status=ok must have failure_code=None")
    else:
        if raw is not None:
            errs.append("failed/skipped rows must have raw_score=None (never 0 or 0.5)")
        if row.get("failure_code") is None:
            errs.append("failed/skipped rows require a failure_code")
    if not row.get("is_fixture") in (True, False):
        errs.append("is_fixture must be boolean")
    return errs


def validate_row(table: str, row: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    if table not in REQUIRED_COLUMNS:
        raise KeyError(f"unknown table: {table}")
    errs: list[str] = []
    spec = REQUIRED_COLUMNS[table]
    for col, typ in spec.items():
        if col not in row:
            errs.append(f"missing column {col}")
            continue
        val = row[col]
        if typ == "float_or_none":
            if val is not None and not _is_float(val):
                errs.append(f"{col} must be a finite float or None")
            continue
        if typ == "str_or_none":
            if val is not None and not _is_str(val):
                errs.append(f"{col} must be a non-empty str or None")
            continue
        if not VALIDATORS[typ](val):
            errs.append(f"{col} fails {typ} validator: {val!r}")
    if table == "scores":
        errs.extend(_validate_score_row(row))
    return errs


def validate_rows(table: str, rows: list[Mapping[str, Any]]) -> list[str]:
    errs: list[str] = []
    for i, row in enumerate(rows):
        for e in validate_row(table, row):
            errs.append(f"row {i}: {e}")
    return errs
