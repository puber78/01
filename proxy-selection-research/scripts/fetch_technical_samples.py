"""Fetch the EXAM01 TECHNICAL sample from public, provenance-documented sources.

This is NOT research data:
  * role = TECH (engineering verification only),
  * never used for selection, calibration or confirmation,
  * every image keeps its dataset, record id, generator model / prompt, licence
    and content hash in a provenance manifest.

Sources
-------
real (label 0): `detection-datasets/coco` (real photographs, COCO).
ai   (label 1): `elsaEU/ELSA1M_track1` (diffusion images; the generating model
                is recorded per row; licence CC-BY-4.0).  Shards are sampled
                across the dataset so several generators appear.

Rows are read with HTTP range requests (footer + first row groups only), which
avoids downloading 500 MB shards and avoids the over-long cache paths that the
`datasets` library creates on this Windows workspace.

Outputs (all created with the no-overwrite guard):
  data/raw/TECH/real/*.png
  data/raw/TECH/ai/*.png
  data/raw/TECH/technical_sources.csv
  data/raw/TECH/provenance.jsonl
  data/raw/TECH/fetch_report.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "hf_home" / "hub"))
sys.path.insert(0, str(ROOT / "src"))

from proxy_selection.hashing import sha256_bytes  # noqa: E402
from proxy_selection.hfstream import UA, decode_image_cell, iter_rows, resolve_url  # noqa: E402
from proxy_selection.io import Ledger, write_new  # noqa: E402
from proxy_selection.runlog import RunLogger  # noqa: E402


def _tree(repo: str, sub: str = "") -> list[dict]:
    url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main"
    if sub:
        url += f"/{sub}"
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())


def _save_png(img, path: Path) -> tuple[str, int, int]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    write_new(path, data, kind="tech_image")
    return sha256_bytes(data), img.width, img.height


def _row_real(image_id: int, digest: str, w: int, h: int, rel: Path) -> dict:
    return {
        "tech_id": f"tech_real_{image_id:012d}",
        "label": 0,
        "split_role": "TECH",
        "source_dataset": "detection-datasets/coco",
        "source_split": "train",
        "record_id": str(image_id),
        "generator_model": "",
        "prompt": "",
        "license_id": "coco_terms_of_use",
        "license_url": "https://cocodataset.org/#termsofuse",
        "file_relpath": str(rel).replace("\\", "/"),
        "sha256": digest,
        "width": w,
        "height": h,
    }


def load_existing_real(out_dir: Path, ledger: Ledger, log: RunLogger) -> tuple[list[dict], list[dict]]:
    """Rebuild manifest rows for real images already on disk (crash resume)."""
    from PIL import Image

    from proxy_selection.hashing import sha256_file

    rows: list[dict] = []
    prov: list[dict] = []
    for p in sorted((out_dir / "real").glob("coco_*.png")):
        try:
            image_id = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        with Image.open(p) as im:
            w, h = im.size
        digest = sha256_file(p)
        rel = p.relative_to(ROOT)
        rows.append(_row_real(image_id, digest, w, h, rel))
        prov.append(
            {
                "tech_id": f"tech_real_{image_id:012d}",
                "dataset": "detection-datasets/coco",
                "record": {"image_id": image_id, "width": w, "height": h},
                "license": "COCO terms of use; source images originate from Flickr",
                "note": "reconstructed from a previous partial run (file already on disk)",
            }
        )
    if rows:
        log.event("real_resume", n=len(rows))
    return rows, prov


def fetch_real(
    out_dir: Path, n: int, scan_limit: int, ledger: Ledger, log: RunLogger,
    existing_rows: list[dict], existing_prov: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    rows = list(existing_rows)
    prov = list(existing_prov)
    seen = {r["record_id"] for r in rows}
    stats = {"scanned": 0, "kept": len(rows), "decode_failures": 0, "too_small": 0, "network": False}
    if len(rows) >= n:
        return rows, prov, stats

    files = [f["path"] for f in _tree("detection-datasets/coco", "data") if f["path"].endswith(".parquet")]
    url = resolve_url("detection-datasets/coco", files[0])
    stats["network"] = True
    stats["file"] = files[0]
    for rec in iter_rows(url, limit=scan_limit, columns=["image_id", "image", "width", "height"]):
        stats["scanned"] += 1
        if len(rows) >= n:
            break
        image_id = int(rec["image_id"])
        if str(image_id) in seen:
            continue
        try:
            img = decode_image_cell(rec["image"])
        except Exception:  # noqa: BLE001
            stats["decode_failures"] += 1
            continue
        if min(img.width, img.height) < 480:
            stats["too_small"] += 1
            continue
        rel = Path("data") / "raw" / "TECH" / "real" / f"coco_{image_id:012d}.png"
        digest, w, h = _save_png(img, ROOT / rel)
        ledger.record_artifact(ROOT / rel, kind="tech_image", run_id=log.run_id)
        rows.append(_row_real(image_id, digest, w, h, rel))
        prov.append(
            {
                "tech_id": f"tech_real_{image_id:012d}",
                "dataset": "detection-datasets/coco",
                "record": {"image_id": image_id, "width": rec.get("width"), "height": rec.get("height")},
                "license": "COCO terms of use; source images originate from Flickr",
                "parquet": files[0],
            }
        )
        seen.add(str(image_id))
        stats["kept"] += 1
    return rows, prov, stats


def fetch_ai(out_dir: Path, n: int, ledger: Ledger, log: RunLogger) -> tuple[list[dict], list[dict], dict]:
    import time

    files = [f["path"] for f in _tree("elsaEU/ELSA1M_track1", "data") if f["path"].endswith(".parquet")]
    files.sort()
    # Sample a handful of shards spread across the dataset so several generators appear.
    n_shards = 4
    step = max(1, len(files) // n_shards)
    picks = files[::step][:n_shards]
    per_shard = max(1, (n + len(picks) - 1) // len(picks))
    rows: list[dict] = []
    prov: list[dict] = []
    stats = {"shards": picks, "scanned": 0, "kept": 0, "decode_failures": 0, "too_small": 0, "models": {}}
    for shard in picks:
        if len(rows) >= n:
            break
        url = resolve_url("elsaEU/ELSA1M_track1", shard)
        taken = 0
        for rec in iter_rows(
            url,
            limit=per_shard * 3,
            columns=["image", "id", "model", "positive_prompt", "original_prompt", "filepath", "aspect_ratio", "nsfw"],
        ):
            if len(rows) >= n or taken >= per_shard:
                break
            stats["scanned"] += 1
            try:
                img = decode_image_cell(rec["image"])
            except Exception:  # noqa: BLE001
                stats["decode_failures"] += 1
                continue
            if min(img.width, img.height) < 384:
                stats["too_small"] += 1
                continue
            rid = str(rec["id"])
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rid)[:80] or "unknown"
            rel = Path("data") / "raw" / "TECH" / "ai" / f"elsa_{safe}.png"
            if (ROOT / rel).exists():
                continue
            digest, w, h = _save_png(img, ROOT / rel)
            ledger.record_artifact(ROOT / rel, kind="tech_image", run_id=log.run_id)
            model = str(rec.get("model") or "")
            prompt = str(rec.get("positive_prompt") or rec.get("original_prompt") or "")[:500]
            rows.append(
                {
                    "tech_id": f"tech_ai_{rid}",
                    "label": 1,
                    "split_role": "TECH",
                    "source_dataset": "elsaEU/ELSA1M_track1",
                    "source_split": "train",
                    "record_id": rid,
                    "generator_model": model,
                    "prompt": prompt,
                    "license_id": "cc-by-4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "file_relpath": str(rel).replace("\\", "/"),
                    "sha256": digest,
                    "width": w,
                    "height": h,
                }
            )
            prov.append(
                {
                    "tech_id": f"tech_ai_{rid}",
                    "dataset": "elsaEU/ELSA1M_track1",
                    "record": {
                        "id": rid,
                        "model": model,
                        "prompt": prompt,
                        "filepath": rec.get("filepath"),
                        "aspect_ratio": rec.get("aspect_ratio"),
                        "nsfw": rec.get("nsfw"),
                    },
                    "license": "CC-BY-4.0",
                    "parquet": shard,
                }
            )
            taken += 1
            stats["kept"] += 1
        time.sleep(4.0)  # be polite to the mirror between shards
    stats["models"] = dict(Counter(r["generator_model"] for r in rows))
    return rows, prov, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=40)
    ap.add_argument("--n-ai", type=int, default=40)
    ap.add_argument("--scan-limit", type=int, default=90)
    args = ap.parse_args()

    log = RunLogger.start(ROOT, note="fetch technical samples (real=COCO, ai=ELSA1M)")
    ledger = Ledger(ROOT / "ops")
    out_dir = ROOT / "data" / "raw" / "TECH"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows, real_prov = load_existing_real(out_dir, ledger, log)
    real_rows, real_prov, real_stats = fetch_real(
        out_dir, args.n_real, args.scan_limit, ledger, log, real_rows, real_prov
    )
    log.event("real_fetch", **real_stats)
    ai_rows, ai_prov, ai_stats = fetch_ai(out_dir, args.n_ai, ledger, log)
    log.event("ai_fetch", **ai_stats)

    rows = real_rows + ai_rows
    if not rows:
        log.finish("FAIL", reason="no technical image fetched")
        return 1
    assert {r["label"] for r in rows} == {0, 1}, "technical sample must contain both classes"

    csv_path = out_dir / "technical_sources.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    write_new(csv_path, buf.getvalue(), ledger=ledger, run_id=log.run_id, kind="tech_manifest")

    payload = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in real_prov + ai_prov)
    write_new(out_dir / "provenance.jsonl", payload, ledger=ledger, run_id=log.run_id, kind="tech_provenance")

    report = {
        "run_id": log.run_id,
        "real": real_stats,
        "ai": ai_stats,
        "n_total": len(rows),
        "csv_sha256": sha256_bytes(buf.getvalue().encode("utf-8")),
        "note": "TECH role only; not research data; not usable as confirmation.",
    }
    write_new(
        out_dir / "fetch_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
        ledger=ledger,
        run_id=log.run_id,
        kind="tech_report",
    )
    log.artifact(csv_path, report["csv_sha256"])
    log.finish("PASS", n_total=len(rows))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
