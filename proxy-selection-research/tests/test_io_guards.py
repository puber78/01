import csv
from pathlib import Path

import pytest

from proxy_selection.io import (
    ImmutableFileError,
    Ledger,
    read_parquet,
    write_json,
    write_new,
    write_parquet,
)

RUN = "run_aaaaaaaabbbbbbbbcccccccc"


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ops")


def test_write_new_creates_and_records(tmp_path):
    led = _ledger(tmp_path)
    p = tmp_path / "out" / "a.txt"
    digest = write_new(p, "hello", ledger=led, run_id=RUN, kind="test")
    assert p.read_text(encoding="utf-8") == "hello"
    assert len(digest) == 64
    assert led.count_artifacts() == 1
    assert led.count_violations() == 0


def test_write_new_refuses_overwrite_and_counts_violation(tmp_path):
    led = _ledger(tmp_path)
    p = tmp_path / "a.txt"
    write_new(p, "hello", ledger=led, run_id=RUN)
    with pytest.raises(ImmutableFileError):
        write_new(p, "hello again", ledger=led, run_id=RUN)
    assert p.read_text(encoding="utf-8") == "hello"
    assert led.count_violations() == 1


def test_write_new_never_deletes_originals(tmp_path):
    original = tmp_path / "original.bin"
    original.write_bytes(b"immutable input")
    with pytest.raises(ImmutableFileError):
        write_new(original, b"new")
    assert original.read_bytes() == b"immutable input"


def test_parquet_roundtrip_and_schema(tmp_path):
    led = _ledger(tmp_path)
    row = {
        "run_id": RUN,
        "sample_id": "smp_0123456789abcdef01234567",
        "source_id": "src_0123456789abcdef01234567",
        "split_role": "TECH",
        "environment_id": "env0",
        "model_id": "mdl_0123456789abcdef01234567",
        "image_branch": "digital",
        "transform_id": "clean",
        "raw_score": 0.5,
        "score_orientation": "higher_is_ai",
        "checkpoint_hash": "c" * 64,
        "preprocess_hash": "p" * 64,
        "status": "ok",
        "failure_code": None,
        "is_fixture": False,
    }
    p = tmp_path / "scores.parquet"
    write_parquet(p, [row], "scores", ledger=led, run_id=RUN)
    df = read_parquet(p, "scores")
    assert len(df) == 1
    assert df.loc[0, "raw_score"] == pytest.approx(0.5)


def test_parquet_rejects_invalid_row(tmp_path):
    bad = {"run_id": RUN}
    with pytest.raises(ValueError):
        write_parquet(tmp_path / "bad.parquet", [bad], "scores")


def test_write_json_and_archive_on_replace(tmp_path):
    led = _ledger(tmp_path)
    p = tmp_path / "r.json"
    write_json(p, {"v": 1}, ledger=led, run_id=RUN)
    assert p.exists()
    write_json(
        p,
        {"v": 2},
        ledger=led,
        run_id=RUN,
        replace=True,
        archive_root=tmp_path / "ops" / "archive",
    )
    assert p.read_text(encoding="utf-8").strip().endswith("}")
    backups = list((tmp_path / "ops" / "archive").rglob("*.bak"))
    assert len(backups) == 1
    assert '"v": 1' in backups[0].read_text(encoding="utf-8")
