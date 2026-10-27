import json
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

from .hfstream import iter_rows as _hf_iter_rows
from .hfstream import resolve_url


def read_local_rows(
    path: Any,
    *,
    columns: list[str] | None = None,
    limit: int = 500,
    skip: int = 0,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Read rows from a local parquet file (fast, no network)."""
    pf = pq.ParquetFile(str(path))
    out: list[dict[str, Any]] = []
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            if seen >= skip:
                out.append(row)
                if len(out) >= limit:
                    return out
            seen += 1
    return out


def stream_rows(
    *,
    local_paths: Sequence[Any] | Any | None,
    repo: str,
    remote_files: list[str],
    columns: list[str] | None = None,
    limit_per_shard: int = 200,
    total_limit: int | None = None,
    sleep_s: float = 1.5,
) -> Iterator[dict[str, Any]]:
    """Yield rows from local parquet shards if present, otherwise from remote shards."""
    produced = 0
    if local_paths is not None:
        paths = (
            [local_paths]
            if isinstance(local_paths, (str, Path))
            else list(local_paths)
        )
        paths = [Path(p) for p in paths if Path(p).exists()]
        if paths:
            for p in paths:
                pf = pq.ParquetFile(str(p))
                for batch in pf.iter_batches(batch_size=32, columns=columns):
                    for row in batch.to_pylist():
                        yield row
                        produced += 1
                        if total_limit and produced >= total_limit:
                            return
            return
    for name in remote_files:
        for row in _hf_iter_rows(resolve_url(repo, name), limit=limit_per_shard, columns=columns):
            yield row
            produced += 1
            if total_limit and produced >= total_limit:
                return
        time.sleep(sleep_s)
