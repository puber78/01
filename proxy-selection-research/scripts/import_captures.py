"""Import, register and QC the physical captures of one session.

Mapping strategy
----------------
The photographer shoots strictly in queue order; extra files are retakes.  Each
capture is matched against its *expected* display asset with ORB features and a
small look-ahead (1-2 extra files), so a retake cannot shift the whole session.
Every assignment is verified by a homography with RANSAC inlier counting; a low
inlier ratio is recorded as an explicit QC failure rather than silently kept.

Registration
------------
The homography maps the 2560x1440 display canvas onto the photo; the photo is
warped to the canvas, then cropped to the stored content rectangle.  This is the
frozen physical branch input; the un-warped photo is kept as raw evidence.

Outputs
-------
  data/raw/captures/<session>/<capture_id>.jpg       original bytes (copied)
  data/physical/<session>/<capture_id>.png           rectified content crop
  manifests/EXAM02_captures/<run>/captures.parquet   schema-valid capture rows
  manifests/EXAM02_captures/<run>/qc.json            per-capture QC metrics
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from proxy_selection.contracts.ids import new_capture_id  # noqa: E402
from proxy_selection.hashing import sha256_bytes, sha256_file  # noqa: E402
from proxy_selection.io import Ledger, write_new, write_parquet  # noqa: E402
from proxy_selection.runlog import RunLogger  # noqa: E402

MIN_MATCHES = 18
MIN_INLIER_RATIO = 0.25
BLUR_MIN = 60.0
MEAN_MIN, MEAN_MAX = 25.0, 235.0


def orb_features(gray: np.ndarray, n: int = 5000) -> tuple[list, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=n, fastThreshold=12)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_count(des_a: np.ndarray | None, des_b: np.ndarray | None) -> int:
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(des_a, des_b, k=2)
    good = 0
    for p in pairs:
        if len(p) == 2 and p[0].distance < 0.75 * p[1].distance:
            good += 1
    return good


def register(display: np.ndarray, photo: np.ndarray) -> dict:
    g1 = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    kp1, d1 = orb_features(g1)
    kp2, d2 = orb_features(g2)
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return {"ok": False, "reason": "too_few_features", "good": 0, "inliers": 0}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_MATCHES:
        return {"ok": False, "reason": "too_few_matches", "good": len(good), "inliers": 0}
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return {"ok": False, "reason": "homography_failed", "good": len(good), "inliers": 0}
    inliers = int(mask.sum())
    ratio = inliers / max(1, len(good))
    return {
        "ok": inliers >= MIN_MATCHES and ratio >= MIN_INLIER_RATIO,
        "reason": "" if inliers >= MIN_MATCHES and ratio >= MIN_INLIER_RATIO else "low_inlier_ratio",
        "good": len(good),
        "inliers": inliers,
        "inlier_ratio": ratio,
        "H": H,
    }


def qc_metrics(rect: np.ndarray, coverage: float) -> dict:
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    clipped_hi = float((gray >= 250).mean())
    clipped_lo = float((gray <= 5).mean())
    return {
        "blur_laplacian_var": blur,
        "mean_luma": mean,
        "clip_high_fraction": clipped_hi,
        "clip_low_fraction": clipped_lo,
        "screen_coverage": coverage,
        "sharp": blur >= BLUR_MIN,
        "exposure_ok": MEAN_MIN <= mean <= MEAN_MAX and clipped_hi < 0.05,
    }


def poly_area(poly: np.ndarray) -> float:
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--queue", default=None)
    ap.add_argument("--device-id", default="redmi_k80")
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = RunLogger.start(ROOT, note=f"import captures {args.session}")
    ledger = Ledger(ROOT / "ops")
    queue_path = Path(args.queue) if args.queue else ROOT / "captures" / "kit" / f"queue_{args.session}.json"
    queue = json.loads(queue_path.read_text("utf-8"))
    items = [it for it in queue["items"]]
    progress_path = queue_path.with_name(queue_path.stem + "_progress.json")
    progress = json.loads(progress_path.read_text("utf-8")) if progress_path.exists() else {"items": {}}
    status = {int(k): v.get("status") for k, v in progress.get("items", {}).items()}
    expected = [it for it in items if status.get(it["seq"]) == "shot"]
    skipped = [it["seq"] for it in items if status.get(it["seq"]) == "skipped"]
    if not expected:
        print("no items marked as shot in the progress file")
        return 2

    incoming = Path(args.incoming)
    files = sorted(
        [p for p in incoming.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".heic", ".png"}],
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    log.event("incoming", n_files=len(files), n_expected=len(expected), skipped=skipped)
    if len(files) < len(expected):
        print(f"FAIL: {len(files)} files for {len(expected)} expected shots")
        log.finish("FAIL", reason="too_few_files")
        return 3

    display_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    capture_rows, qc_rows = [], []
    j = 0
    for it in expected:
        disp = cv2.imread(str(ROOT / it["file_relpath"]))
        if disp is None:
            raise FileNotFoundError(it["file_relpath"])
        if it["display_asset_id"] not in display_cache:
            g = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
            _, des = orb_features(g)
            display_cache[it["display_asset_id"]] = (disp, des)
        _, d_disp = display_cache[it["display_asset_id"]]

        best = None
        for k in range(j, min(j + args.lookahead + 1, len(files))):
            photo = cv2.imread(str(files[k]))
            if photo is None:
                continue
            g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
            _, des = orb_features(g)
            score = match_count(d_disp, des)
            if best is None or score > best[0]:
                best = (score, k, photo)
        if best is None:
            break
        _, k, photo = best
        j = k + 1
        reg = register(disp, photo)
        capture_id = new_capture_id()
        src_file = files[k]
        raw_digest = sha256_file(src_file)

        qc_status, failure_code = "pending", None
        rect_path = None
        qc = {"good_matches": reg.get("good"), "inliers": reg.get("inliers"),
              "inlier_ratio": reg.get("inlier_ratio")}
        if not reg["ok"]:
            qc_status, failure_code = "failed", "REGISTRATION_FAILED"
        else:
            # H maps display coordinates -> photo coordinates; invert to rectify
            H_inv = np.linalg.inv(reg["H"])
            warped = cv2.warpPerspective(photo, H_inv, (disp.shape[1], disp.shape[0]))
            # geometry shipped with the queue item (content rect inside the canvas)
            geom = it.get("geometry") or {
                "content_x": 0, "content_y": 0, "content_w": disp.shape[1], "content_h": disp.shape[0]
            }
            x, y, w, h = geom["content_x"], geom["content_y"], geom["content_w"], geom["content_h"]
            rect = warped[y : y + h, x : x + w]
            qc.update(qc_metrics(rect, coverage=float(w * h) / (disp.shape[0] * disp.shape[1])))
            if not qc["sharp"]:
                qc_status, failure_code = "failed", "BLUR"
            elif not qc["exposure_ok"]:
                qc_status, failure_code = "failed", "EXPOSURE"
            else:
                qc_status = "passed"
            if not args.dry_run:
                out = ROOT / "data" / "physical" / args.session / f"{capture_id}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), rect)
                rect_path = str(out.relative_to(ROOT)).replace("\\", "/")
                ledger.record_artifact(out, kind="physical_rectified", run_id=log.run_id)

        if not args.dry_run:
            raw_dir = ROOT / "data" / "raw" / "captures" / args.session
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_dest = raw_dir / f"{capture_id}{src_file.suffix.lower()}"
            shutil.copy2(src_file, raw_dest)
            ledger.record_artifact(raw_dest, kind="capture_original", run_id=log.run_id)

        capture_rows.append(
            {
                "capture_id": capture_id,
                "display_asset_id": it["display_asset_id"],
                "session_id": args.session,
                "attempt_index": int(it["seq"]),
                "is_planned_repeat": bool(it.get("is_repeat", False)),
                "capture_file_sha256": raw_digest,
                "device_id": args.device_id,
                "qc_status": qc_status,
                "failure_code": failure_code,
                "captured_at": datetime.fromtimestamp(src_file.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "run_id": log.run_id,
            }
        )
        qc_rows.append(
            {
                "seq": it["seq"],
                "capture_id": capture_id,
                "source_id": it["source_id"],
                "role": it["role"],
                "label": it["label"],
                "is_repeat": bool(it.get("is_repeat", False)),
                "file": str(files[k]),
                "rectified": rect_path,
                "qc_status": qc_status,
                "failure_code": failure_code,
                **{k2: v for k2, v in qc.items() if not isinstance(v, np.ndarray)},
            }
        )

    outdir = ROOT / "manifests" / "EXAM02_captures" / log.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    if capture_rows and not args.dry_run:
        write_parquet(outdir / "captures.parquet", capture_rows, "captures", ledger=ledger, run_id=log.run_id)
    write_new(outdir / "qc.json", json.dumps(qc_rows, ensure_ascii=False, indent=2), kind="qc")
    summary = {
        "session": args.session,
        "expected": len(expected),
        "files": len(files),
        "imported": len(capture_rows),
        "passed": sum(1 for r in qc_rows if r["qc_status"] == "passed"),
        "failed": sum(1 for r in qc_rows if r["qc_status"] == "failed"),
        "failure_codes": pd.Series([r["failure_code"] for r in qc_rows if r["failure_code"]]).value_counts().to_dict(),
        "skipped": skipped,
        "dry_run": args.dry_run,
    }
    write_new(outdir / "import_summary.json", json.dumps(summary, ensure_ascii=False, indent=2), kind="import_summary")
    log.finish("PASS" if summary["failed"] == 0 else "PARTIAL", **{k: v for k, v in summary.items() if k != "skipped"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["imported"] == len(expected) else 4


if __name__ == "__main__":
    raise SystemExit(main())
