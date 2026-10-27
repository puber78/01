"""Scoring layer: joins images + frozen labels to adapter predictions.

The adapter sees only pixels.  Source ids, labels, data roles and the
environment come from manifests, are joined AFTER inference, and the join is
recorded with hashes.  This separation is what lets the development role run
inference on unlabeled images without ever touching a label.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .adapters.base import DetectorAdapter, Prediction
from .contracts.ids import require_id
from .metrics.auc import roc_auc

MAX_PIXELS = 64_000_000  # decompression-bomb guard, ~8000x8000


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    sha256: str
    image: Image.Image | None
    failure_code: str | None
    detail: str = ""


def load_rgb(path: str | Path) -> LoadedImage:
    """Read an image file as single-frame RGB; failures are explicit codes."""
    p = Path(path)
    if not p.exists():
        return LoadedImage(p, "", None, "FILE_MISSING")
    try:
        data = p.read_bytes()
    except OSError as exc:
        return LoadedImage(p, "", None, "DECODE_ERROR", str(exc))
    digest = hashlib.sha256(data).hexdigest()
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            return LoadedImage(p, digest, None, "UNSUPPORTED_FORMAT", "pixel bomb guard")
        img = img.convert("RGB")
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return LoadedImage(p, digest, None, "DECODE_ERROR", str(exc))
    return LoadedImage(p, digest, img, None)


def score_rows(
    adapter: DetectorAdapter,
    samples: Sequence[dict[str, Any]],
    *,
    run_id: str,
    transform_id: str,
    is_fixture: bool = False,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run unlabeled inference over samples and return schema-valid rows.

    Each sample dict needs: sample_id, source_id, split_role, environment_id,
    file_path.  ``image_branch`` is taken as 'digital' unless stated.
    """
    require_id(run_id, "run")
    rows: list[dict[str, Any]] = []
    loaded: list[LoadedImage] = []
    pending: list[dict[str, Any]] = []

    for s in samples:
        require_id(s["sample_id"], "sample")
        require_id(s["source_id"], "source")
        li = load_rgb(s["file_path"])
        if li.image is None:
            rows.append(_row(s, adapter, None, "failed", li.failure_code, run_id, transform_id, is_fixture))
        else:
            loaded.append(li)
            pending.append(s)

    bs = getattr(adapter, "batch_size", 16)
    preds: list[Prediction] = []
    for start in range(0, len(loaded), bs):
        batch = [li.image for li in loaded[start : start + bs]]  # type: ignore[list-item]
        preds.extend(adapter.predict_batch(batch))
        if progress:
            print(f"  scored {min(start + bs, len(loaded))}/{len(loaded)}", flush=True)

    for s, li, pred in zip(pending, loaded, preds):
        rows.append(
            _row(s, adapter, pred.raw_score, pred.status, pred.failure_code, run_id, transform_id, is_fixture)
        )
    return rows


def _row(
    s: dict[str, Any],
    adapter: DetectorAdapter,
    raw_score: float | None,
    status: str,
    failure_code: str | None,
    run_id: str,
    transform_id: str,
    is_fixture: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": s["sample_id"],
        "source_id": s["source_id"],
        "split_role": s["split_role"],
        "environment_id": s["environment_id"],
        "model_id": adapter.model_id,
        "image_branch": s.get("image_branch", "digital"),
        "transform_id": transform_id,
        "raw_score": float(raw_score) if raw_score is not None else None,
        "score_orientation": adapter.score_direction,
        "checkpoint_hash": adapter.checkpoint_hash,
        "preprocess_hash": adapter.preprocess_hash,
        "status": status,
        "failure_code": failure_code,
        "is_fixture": is_fixture,
    }


def join_labels(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach labels by source_id. Rows without a label are dropped explicitly."""
    lab = labels[["source_id", "label", "split_group_id"]].drop_duplicates("source_id")
    out = scores.merge(lab, on="source_id", how="left", validate="many_to_one")
    return out


def compute_auc_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (model, transform, branch) AUC over rows with status == ok."""
    ok = scored[(scored["status"] == "ok") & (~scored["is_fixture"])].copy()
    rows = []
    for (model_id, branch, transform_id), g in ok.groupby(
        ["model_id", "image_branch", "transform_id"], dropna=False
    ):
        if g["label"].nunique() < 2:
            rows.append(
                {
                    "model_id": model_id,
                    "image_branch": branch,
                    "transform_id": transform_id,
                    "n": int(len(g)),
                    "n_ai": int((g["label"] == 1).sum()),
                    "n_real": int((g["label"] == 0).sum()),
                    "auc": float("nan"),
                    "status": "insufficient_classes",
                }
            )
            continue
        rows.append(
            {
                "model_id": model_id,
                "image_branch": branch,
                "transform_id": transform_id,
                "n": int(len(g)),
                "n_ai": int((g["label"] == 1).sum()),
                "n_real": int((g["label"] == 0).sum()),
                "auc": roc_auc(g["label"].to_numpy(), g["raw_score"].to_numpy()),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).sort_values(["image_branch", "transform_id", "model_id"]).reset_index(drop=True)


def coverage_report(scored: pd.DataFrame, expected_units: int | None = None) -> dict[str, Any]:
    total = len(scored)
    ok = int((scored["status"] == "ok").sum())
    failed = int((scored["status"] == "failed").sum())
    return {
        "rows": total,
        "ok": ok,
        "failed": failed,
        "failure_codes": scored.loc[scored["status"] != "ok", "failure_code"].value_counts().to_dict(),
        "coverage": (ok / total) if total else None,
        "expected_units": expected_units,
        "complete": (total == expected_units) if expected_units is not None else None,
    }
