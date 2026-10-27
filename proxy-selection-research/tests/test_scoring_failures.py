import io

import numpy as np
import pytest
from PIL import Image

from proxy_selection.adapters.base import DetectorAdapter, Prediction
from proxy_selection.contracts.schema import validate_row
from proxy_selection.scoring import coverage_report, load_rgb, score_rows


class DummyAdapter(DetectorAdapter):
    """Test-only adapter: constant score, no torch, clearly marked as fixture."""

    name = "dummy"
    batch_size = 2

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def preprocess(self, images):
        return list(images)

    def predict_batch(self, images):
        return [Prediction(0.5, "ok", None) for _ in images]

    @property
    def score_direction(self) -> str:
        return "higher_is_ai"

    @property
    def checkpoint_hash(self) -> str:
        return "c" * 64

    @property
    def preprocess_hash(self) -> str:
        return "p" * 64

    @property
    def model_id(self) -> str:
        return "mdl_0123456789abcdef01234567"


def _png_bytes(color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="PNG")
    return buf.getvalue()


def test_load_rgb_missing_file(tmp_path):
    got = load_rgb(tmp_path / "nope.png")
    assert got.image is None
    assert got.failure_code == "FILE_MISSING"


def test_load_rgb_decode_error(tmp_path):
    p = tmp_path / "notimage.png"
    p.write_text("not an image", encoding="utf-8")
    got = load_rgb(p)
    assert got.image is None
    assert got.failure_code == "DECODE_ERROR"


def test_load_rgb_converts_to_rgb(tmp_path):
    p = tmp_path / "gray.png"
    Image.new("L", (8, 8), 100).save(p)
    got = load_rgb(p)
    assert got.image is not None
    assert got.image.mode == "RGB"


def test_score_rows_emits_explicit_failure_without_imputation(tmp_path):
    good = tmp_path / "good.png"
    good.write_bytes(_png_bytes())
    samples = [
        {
            "sample_id": "smp_0123456789abcdef01234567",
            "source_id": "src_0123456789abcdef01234567",
            "split_role": "TECH",
            "environment_id": "env0",
            "file_path": str(good),
        },
        {
            "sample_id": "smp_1123456789abcdef01234567",
            "source_id": "src_1123456789abcdef01234567",
            "split_role": "TECH",
            "environment_id": "env0",
            "file_path": str(tmp_path / "missing.png"),
        },
    ]
    rows = score_rows(
        DummyAdapter(),
        samples,
        run_id="run_aaaaaaaabbbbbbbbcccccccc",
        transform_id="clean",
        is_fixture=True,
    )
    assert len(rows) == 2
    for r in rows:
        assert validate_row("scores", r) == []
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["raw_score"] is None
    assert failed[0]["failure_code"] == "FILE_MISSING"
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok[0]["raw_score"] == 0.5
    rep = coverage_report(
        __import__("pandas").DataFrame(rows), expected_units=2
    )
    assert rep["coverage"] == pytest.approx(0.5)
