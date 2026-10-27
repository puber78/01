"""Download a large wheel with visible progress and an integrity check.

Used because `pip install` stalled on the multi-GB torch wheel; the same URL
serves fine over urllib with range requests.  The downloaded file is verified to
be a readable zip archive before being handed to pip.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 research-agent/1.0"}


def download(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"exists: {dest}")
        return
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as fh:
            while True:
                block = r.read(chunk)
                if not block:
                    break
                fh.write(block)
                got += len(block)
                if total:
                    pct = 100.0 * got / total
                    print(
                        f"\r{got/1e6:8.1f}/{total/1e6:8.1f} MB {pct:5.1f}% "
                        f"{got/1e6/max(time.time()-t0,1e-6):5.2f} MB/s",
                        end="",
                        flush=True,
                    )
    print()
    if total and got != total:
        raise RuntimeError(f"short download: {got} != {total}")
    tmp.replace(dest)
    if dest.suffix == ".whl":
        with zipfile.ZipFile(dest) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"corrupt wheel member: {bad}")
    print(f"OK {dest} {dest.stat().st_size/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    args = ap.parse_args()
    download(args.url, Path(args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
