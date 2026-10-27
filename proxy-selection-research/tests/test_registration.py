"""Registration/QC unit tests on synthetic images (never research data)."""
import cv2
import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_captures import match_count, qc_metrics, register  # noqa: E402


def _textured(h=600, w=800, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(8, 40))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1)
    for _ in range(40):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (255, 255, 255), 1)
    return img


def test_registration_recovers_perspective():
    display = _textured(600, 800, seed=1)
    src = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    dst = np.float32([[40, 30], [760, 60], [780, 570], [30, 540]])
    H_true = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(display, H_true, (900, 700), borderValue=(20, 20, 20))
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    res = register(display, photo)
    assert res["ok"], res
    assert res["inliers"] >= 30
    # warping the photo back must roughly reproduce the display
    aligned = cv2.warpPerspective(photo, np.linalg.inv(res["H"]), (800, 600))
    diff = cv2.absdiff(aligned, display).mean()
    assert diff < 12.0


def test_registration_rejects_unrelated_images():
    a = _textured(600, 800, seed=2)
    b = _textured(600, 800, seed=3)
    res = register(a, b)
    assert not res["ok"]


def test_match_count_self_is_high():
    img = _textured(400, 500, seed=4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
    _, des = orb.detectAndCompute(gray, None)
    assert match_count(des, des) > 100


def test_qc_flags_blur():
    sharp = _textured(600, 800, seed=5)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    q_sharp = qc_metrics(sharp, coverage=0.8)
    q_blur = qc_metrics(blurred, coverage=0.8)
    assert q_sharp["blur_laplacian_var"] > q_blur["blur_laplacian_var"]
    assert q_sharp["sharp"] and not q_blur["sharp"]


def test_qc_flags_overexposure():
    bright = np.full((300, 400, 3), 255, np.uint8)
    q = qc_metrics(bright, coverage=0.9)
    assert not q["exposure_ok"]
