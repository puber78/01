"""Read a remote parquet dataset through HTTP range requests (no full download).

Why this exists: the `datasets` library writes long lock-file names that exceed
the Windows path limit inside this (Chinese-named) workspace.  Reading the
parquet shards directly with fsspec + pyarrow needs only the footer and the row
groups actually consumed, which is also cheaper than downloading 500 MB shards.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from typing import Any, Iterator

import fsspec
import pyarrow.parquet as pq
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 research-agent/1.0"}


def resolve_url(repo: str, filename: str, endpoint: str = "https://hf-mirror.com") -> str:
    return f"{endpoint}/datasets/{repo}/resolve/main/{filename}"


def list_dataset_files(repo: str, endpoint: str = "https://hf-mirror.com") -> list[dict[str, Any]]:
    api = f"{endpoint}/api/datasets/{repo}/tree/main"
    req = urllib.request.Request(api, headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def read_rows(
    url: str,
    *,
    columns: list[str] | None = None,
    limit: int = 50,
    block_size: int = 5 * 1024 * 1024,
    retries: int = 3,
    backoff_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Read up to ``limit`` rows, retrying with backoff on transient HTTP errors.

    The mirror occasionally answers 429; retrying the whole shard read is safe
    because the row set for a fixed limit is deterministic.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            fs = fsspec.filesystem("http", headers=UA)
            with fs.open(url, "rb", block_size=block_size) as fh:
                pf = pq.ParquetFile(fh)
                out: list[dict[str, Any]] = []
                for batch in pf.iter_batches(batch_size=min(limit, 100), columns=columns):
                    out.extend(batch.to_pylist())
                    if len(out) >= limit:
                        return out[:limit]
                return out
        except Exception as exc:  # noqa: BLE001 - retried below
            last_exc = exc
            time.sleep(backoff_s * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def iter_rows(url: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    for row in read_rows(url, **kwargs):
        yield row


def decode_image_cell(cell: Any) -> Image.Image:
    """Decode HF `Image` cell ({'bytes': ..., 'path': ...}) into PIL RGB."""
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None:
            raise ValueError("image cell has no bytes (path-only cell unsupported)")
    elif isinstance(cell, (bytes, bytearray)):
        raw = bytes(cell)
    else:
        raise TypeError(f"unsupported image cell type: {type(cell)}")
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("RGB")

