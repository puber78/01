"""EXAM02: build the source pools (C_pool, T_pilot, D_sel) and display assets.

Source population (recorded, not assumed):
  * photographic (label 0): COCO real photographs (detection-datasets/coco)
  * AI-generated (label 1): ELSA1M diffusion images (elsaEU/ELSA1M_track1,
    generator recorded per row)

Roles (budget examples from the outline, not a power guarantee):
  C_pool  40 photo + 40 ai   (physically captured; calibration / selection)
  T_pilot 80 photo + 80 ai   (physically captured; pilot evaluation)
  D_sel   150 photo + 150 ai (DIGITAL ONLY; selection scoring)

Technically the D_sel size is a new decision: 300 groups is ~3.75x C_pool and
keeps digital AUC standard error near 0.025; it is registered as a PROPOSAL to
be approved, not as a statistically justified fixed number.

Display canvas: 2560x1440 sRGB, source fitted (LANCZOS) with black letterbox;
the exact content rectangle is stored so the physical crop can be recovered.

All outputs are new files; every image is hashed and logged.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from proxy_selection.config import DATA_RAW, MANIFESTS, MODELS  # noqa: E402
from proxy_selection.contracts.ids import (  # noqa: E402
    mint_display_asset_id,
    mint_sample_id,
    mint_source_id,
    mint_split_group_id,
)
from proxy_selection.hashing import sha256_bytes, sha256_json  # noqa: E402
from proxy_selection.hfstream import UA, decode_image_cell  # noqa: E402
from proxy_selection.localdata import stream_rows  # noqa: E402
from proxy_selection.io import Ledger, write_parquet  # noqa: E402
from proxy_selection.runlog import RunLogger  # noqa: E402

CANVAS_W, CANVAS_H = 2560, 1440
SEED = 20261026


def _tree(repo: str, sub: str = "") -> list[dict]:
    url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main"
    if sub:
        url += f"/{sub}"
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())


def _save_png(img: Image.Image, path: Path, ledger: Ledger, run_id: str) -> str:
    """Write a PNG once. If it already exists, verify and reuse (idempotent resume)."""
    from proxy_selection.hashing import sha256_file

    if path.exists():
        return sha256_file(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = sha256_bytes(data)
    ledger.record_artifact(path, kind="source_image", run_id=run_id, sha256=digest)
    return digest


def _used_coco_ids() -> set[str]:
    p = ROOT / "data" / "raw" / "TECH" / "technical_sources.csv"
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return set(df.loc[df["source_dataset"] == "detection-datasets/coco", "record_id"])


def fetch_coco(n: int, used: set[str], ledger: Ledger, run_id: str) -> list[dict]:
    files = [f["path"] for f in _tree("detection-datasets/coco", "data") if f["path"].endswith(".parquet")]
    out: list[dict] = []
    # resume: reuse files already written by an interrupted run
    from proxy_selection.hashing import sha256_file

    for p in sorted((DATA_RAW / "sources" / "pool_photo").glob("coco_*.png")):
        try:
            iid = str(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
        with Image.open(p) as im:
            w, h = im.size
        out.append(
            {
                "dataset": "detection-datasets/coco",
                "split": "train",
                "record_id": iid,
                "label": 0,
                "generator": "",
                "content": "photograph",
                "prompt": "",
                "file_relpath": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(p),
                "width": w,
                "height": h,
                "license_id": "coco_terms_of_use",
                "license_url": "https://cocodataset.org/#termsofuse",
                "parquet": "local:resume",
            }
        )
        used.add(iid)
    seen_ids = {r["record_id"] for r in out}
    cols = ["image_id", "image", "width", "height"]
    stream = stream_rows(
        local_paths=[MODELS / "datasets" / f"coco_shard{i}.parquet" for i in range(3)],
        repo="detection-datasets/coco",
        remote_files=files[1:4],
        columns=cols,
        limit_per_shard=220,
        total_limit=n + 3000,
    )
    for rec in stream:
        if len(out) >= n:
            break
        iid = str(int(rec["image_id"]))
        if iid in seen_ids or iid in used:
            continue
        try:
            img = decode_image_cell(rec["image"])
        except Exception:  # noqa: BLE001
            continue
        if min(img.size) < 480:
            continue
        d = DATA_RAW / "sources" / "pool_photo" / f"coco_{int(iid):012d}.png"
        digest = _save_png(img, d, ledger, run_id)
        out.append(
            {
                "dataset": "detection-datasets/coco",
                "split": "train",
                "record_id": iid,
                "label": 0,
                "generator": "",
                "content": "photograph",
                "prompt": "",
                "file_relpath": str(d.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "width": img.width,
                "height": img.height,
                "license_id": "coco_terms_of_use",
                "license_url": "https://cocodataset.org/#termsofuse",
                "parquet": "local_or_remote",
            }
        )
        seen_ids.add(iid)
        used.add(iid)
    return out[:n]


def fetch_elsa(n: int, used: set[str], ledger: Ledger, run_id: str) -> list[dict]:
    files = sorted(f["path"] for f in _tree("elsaEU/ELSA1M_track1", "data") if f["path"].endswith(".parquet"))
    shards = files[:: max(1, len(files) // 8)][:8]
    out: list[dict] = []
    cols = ["image", "id", "model", "positive_prompt", "original_prompt", "filepath", "aspect_ratio"]
    stream = stream_rows(
        local_paths=[MODELS / "datasets" / f"elsa_shard{i}.parquet" for i in range(2)],
        repo="elsaEU/ELSA1M_track1",
        remote_files=shards,
        columns=cols,
        limit_per_shard=80,
        total_limit=n + 3000,
    )
    for rec in stream:
        if len(out) >= n:
            break
        rid = str(rec["id"])
        if rid in used:
            continue
        try:
            img = decode_image_cell(rec["image"])
        except Exception:  # noqa: BLE001
            continue
        if min(img.size) < 384:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rid)[:80] or "unknown"
        d = DATA_RAW / "sources" / "pool_ai" / f"elsa_{safe}.png"
        digest = _save_png(img, d, ledger, run_id)
        out.append(
            {
                "dataset": "elsaEU/ELSA1M_track1",
                "split": "train",
                "record_id": rid,
                "label": 1,
                "generator": str(rec.get("model") or ""),
                "content": "diffusion_generated",
                "prompt": str(rec.get("positive_prompt") or rec.get("original_prompt") or "")[:500],
                "file_relpath": str(d.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "width": img.width,
                "height": img.height,
                "license_id": "cc-by-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "parquet": "local_or_remote",
            }
        )
        used.add(rid)
    return out


def make_display(src_path: Path, out_path: Path, ledger: Ledger, run_id: str) -> tuple[str, dict]:
    img = Image.open(src_path).convert("RGB")
    scale = min(CANVAS_W / img.width, CANVAS_H / img.height)
    w, h = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    resized = img.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    x, y = (CANVAS_W - w) // 2, (CANVAS_H - h) // 2
    canvas.paste(resized, (x, y))
    if out_path.exists():
        from proxy_selection.hashing import sha256_file
        digest = sha256_file(out_path)
        return digest, {"content_x": x, "content_y": y, "content_w": w, "content_h": h,
                        "canvas_w": CANVAS_W, "canvas_h": CANVAS_H}
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    digest = sha256_bytes(data)
    ledger.record_artifact(out_path, kind="display_asset", run_id=run_id, sha256=digest)
    return digest, {"content_x": x, "content_y": y, "content_w": w, "content_h": h, "canvas_w": CANVAS_W, "canvas_h": CANVAS_H}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpool", type=int, default=100)
    ap.add_argument("--tpilot", type=int, default=350)
    ap.add_argument("--dsel", type=int, default=250)
    args = ap.parse_args()

    log = RunLogger.start(ROOT, note="EXAM02 source pools + display assets")
    ledger = Ledger(ROOT / "ops")
    rng = np.random.default_rng(SEED)

    used_coco = _used_coco_ids()
    n_photo = args.cpool + args.tpilot + args.dsel
    photo = fetch_coco(n_photo, used_coco, ledger, log.run_id)
    log.event("coco_fetch", requested=n_photo, got=len(photo))
    used_elsa: set[str] = set()
    ai = fetch_elsa(n_photo, used_elsa, ledger, log.run_id)
    log.event("elsa_fetch", requested=n_photo, got=len(ai))
    if len(photo) < n_photo or len(ai) < n_photo:
        log.finish("FAIL", reason="source quota not met", photo=len(photo), ai=len(ai))
        print(f"FAIL: photo={len(photo)} ai={len(ai)} need {n_photo}")
        return 1

    # deterministic role assignment per class
    def split_role(items: list[dict], sizes: dict[str, int], seed_offset: int) -> dict[str, list[dict]]:
        order = np.random.default_rng(SEED + seed_offset).permutation(len(items))
        shuffled = [items[i] for i in order]
        out, cur = {}, 0
        for role, k in sizes.items():
            out[role] = shuffled[cur : cur + k]
            cur += k
        return out

    sizes = {"C_pool": args.cpool, "T_pilot": args.tpilot, "D_sel": args.dsel}
    photo_role = split_role(photo, sizes, 1)
    ai_role = split_role(ai, sizes, 2)

    sources_rows, splits_rows, display_rows, sample_rows, prov_rows = [], [], [], [], []
    now = log.run_id  # placeholder for timestamps; ISO time recorded below
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assignment_hash = None

    for role in ("C_pool", "T_pilot", "D_sel"):
        for rec in photo_role[role] + ai_role[role]:
            label = int(rec["label"])
            lineage = f"{rec['dataset']}:{rec['record_id']}"
            src_id = mint_source_id(
                lineage_key=lineage, label=label, provenance_uri=f"{rec['dataset']}#{rec['record_id']}"
            )
            grp_id = mint_split_group_id(
                lineage_key=lineage, generation_family=(rec["generator"] or "camera_original")
            )
            sources_rows.append(
                {
                    "source_id": src_id,
                    "split_group_id": grp_id,
                    "lineage_root_id": lineage,
                    "label": label,
                    "content_category": rec["content"],
                    "generator_id": rec["generator"] or "camera_original",
                    "provenance_uri": f"{rec['dataset']}#{rec['record_id']}",
                    "provenance_sha256": sha256_json(
                        {"dataset": rec["dataset"], "record_id": rec["record_id"], "prompt": rec["prompt"]}
                    ),
                    "license_id": rec["license_id"],
                    "license_evidence_sha256": sha256_json(
                        {"license_id": rec["license_id"], "license_url": rec["license_url"]}
                    ),
                    "label_evidence_sha256": sha256_json({"dataset": rec["dataset"], "label": label}),
                    "original_file_sha256": rec["sha256"],
                    "ingested_at": ts,
                    "ingest_run_id": log.run_id,
                    "notes": f"role={role}",
                }
            )
            splits_rows.append(
                {
                    "split_group_id": grp_id,
                    "source_id": src_id,
                    "split_role": role,
                    "assigned_at": ts,
                    "assignment_hash": sha256_json({"role": role, "source_id": src_id, "seed": SEED}),
                    "assignment_run_id": log.run_id,
                }
            )
            prov_rows.append(
                {
                    "source_id": src_id,
                    "role": role,
                    "dataset": rec["dataset"],
                    "record_id": rec["record_id"],
                    "generator": rec["generator"],
                    "prompt": rec["prompt"],
                    "parquet": rec["parquet"],
                    "license": rec["license_id"],
                }
            )
            if role in ("C_pool", "T_pilot"):
                spec_hash = sha256_json(
                    {"canvas": [CANVAS_W, CANVAS_H], "fit": "lanczos_letterbox_black", "role": role}
                )
                dsp_id = mint_display_asset_id(source_id=src_id, display_spec_hash=spec_hash)
                out_path = ROOT / "data" / "display" / role / f"{dsp_id}.png"
                d_digest, geom = make_display(ROOT / rec["file_relpath"], out_path, ledger, log.run_id)
                display_rows.append(
                    {
                        "display_asset_id": dsp_id,
                        "source_id": src_id,
                        "display_file_sha256": d_digest,
                        "width": geom["canvas_w"],
                        "height": geom["canvas_h"],
                        "color_space": "sRGB_unmanaged",
                        "display_spec_hash": spec_hash,
                        "created_at": ts,
                        "run_id": log.run_id,
                    }
                )
                # digital clean sample from the display asset (same pixels as shown)
                smp_id = mint_sample_id(branch="digital", transform_id="clean", parent_key=dsp_id)
                sample_rows.append(
                    {
                        "sample_id": smp_id,
                        "source_id": src_id,
                        "split_group_id": grp_id,
                        "display_asset_id": dsp_id,
                        "split_role": role,
                        "environment_id": "display_2560x1440",
                        "file_relpath": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                    }
                )
                display_rows[-1]["_geom"] = geom
            else:
                # D_sel: digital-only, clean sample uses the source image itself
                spec_hash = sha256_json({"display_spec": "dsel_source_identity"})
                dsp_id = mint_display_asset_id(source_id=src_id, display_spec_hash=spec_hash)
                display_rows.append(
                    {
                        "display_asset_id": dsp_id,
                        "source_id": src_id,
                        "display_file_sha256": rec["sha256"],
                        "width": int(rec["width"]),
                        "height": int(rec["height"]),
                        "color_space": "sRGB_unmanaged",
                        "display_spec_hash": spec_hash,
                        "created_at": ts,
                        "run_id": log.run_id,
                        "_geom": {
                            "content_x": 0,
                            "content_y": 0,
                            "content_w": int(rec["width"]),
                            "content_h": int(rec["height"]),
                            "canvas_w": int(rec["width"]),
                            "canvas_h": int(rec["height"]),
                            "identity_digital_only": True,
                        },
                    }
                )
                smp_id = mint_sample_id(branch="digital", transform_id="clean", parent_key=dsp_id)
                sample_rows.append(
                    {
                        "sample_id": smp_id,
                        "source_id": src_id,
                        "split_group_id": grp_id,
                        "display_asset_id": dsp_id,
                        "split_role": role,
                        "environment_id": "no_display_digital_only",
                        "file_relpath": rec["file_relpath"],
                    }
                )

    # drop helper key before schema validation, keep geometry separately
    geom = {r["display_asset_id"]: r.pop("_geom") for r in display_rows}
    assignment_hash = sha256_json(sorted(r["source_id"] for r in splits_rows))
    for r in splits_rows:
        r["assignment_hash"] = assignment_hash

    outdir = MANIFESTS / "EXAM02" / log.run_id
    write_parquet(outdir / "sources.parquet", sources_rows, "sources", ledger=ledger, run_id=log.run_id)
    write_parquet(outdir / "splits.parquet", splits_rows, "splits", ledger=ledger, run_id=log.run_id)
    write_parquet(outdir / "display_assets.parquet", display_rows, "display_assets", ledger=ledger, run_id=log.run_id)
    (outdir / "display_geometry.json").write_text(json.dumps(geom, indent=2), encoding="utf-8")
    ledger.record_artifact(outdir / "display_geometry.json", kind="display_geometry", run_id=log.run_id)
    (outdir / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample_rows), encoding="utf-8"
    )
    ledger.record_artifact(outdir / "samples.jsonl", kind="samples", run_id=log.run_id)
    (outdir / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prov_rows), encoding="utf-8"
    )
    ledger.record_artifact(outdir / "provenance.jsonl", kind="provenance", run_id=log.run_id)

    summary = {
        "run_id": log.run_id,
        "roles": {
            role: {
                "photo": int(len(photo_role[role])),
                "ai": int(len(ai_role[role])),
                "display_assets": sum(1 for d in display_rows if d["source_id"] in {s["source_id"] for s in sources_rows if s["notes"] == f"role={role}"}),
            }
            for role in sizes
        },
        "n_sources": len(sources_rows),
        "n_display_assets": len(display_rows),
        "canvas": [CANVAS_W, CANVAS_H],
        "assignment_hash": assignment_hash,
        "generator_mix": {str(k): int(v) for k, v in pd.Series([r["generator"] for r in ai]).value_counts().items()},
        "note": "D_sel size is a proposal, not a power calculation.",
    }
    (outdir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.record_artifact(outdir / "build_summary.json", kind="summary", run_id=log.run_id)
    log.finish("PASS", n_sources=len(sources_rows), n_display=len(display_rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

