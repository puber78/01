import pytest

from proxy_selection.contracts.ids import (
    IdError,
    mint_display_asset_id,
    mint_model_id,
    mint_sample_id,
    mint_source_id,
    new_capture_id,
    new_run_id,
    require_id,
)
from proxy_selection.contracts.roles import AccessPolicy, AccessViolation, roles_are_disjoint
from proxy_selection.contracts.schema import validate_row


def test_source_id_is_deterministic_and_typed():
    a = mint_source_id(lineage_key="L001", label=0, provenance_uri="file:///x.jpg")
    b = mint_source_id(lineage_key="L001", label=0, provenance_uri="file:///x.jpg")
    assert a == b
    assert a.startswith("src_")
    require_id(a, "source")


def test_source_id_changes_with_label_and_provenance():
    a = mint_source_id(lineage_key="L001", label=0, provenance_uri="file:///x.jpg")
    b = mint_source_id(lineage_key="L001", label=1, provenance_uri="file:///x.jpg")
    c = mint_source_id(lineage_key="L001", label=0, provenance_uri="file:///y.jpg")
    assert len({a, b, c}) == 3


def test_sample_branch_must_match_parent_type():
    src = mint_source_id(lineage_key="L1", label=0, provenance_uri="u")
    dsp = mint_display_asset_id(source_id=src, display_spec_hash="d" * 64)
    cap = new_capture_id()
    mint_sample_id(branch="digital", transform_id="clean", parent_key=dsp)
    mint_sample_id(branch="physical", transform_id="raw", parent_key=cap)
    with pytest.raises(IdError):
        mint_sample_id(branch="digital", transform_id="clean", parent_key=cap)


def test_require_id_rejects_wrong_prefix():
    with pytest.raises(IdError):
        require_id("src_0123456789abcdef", "model")
    with pytest.raises(IdError):
        require_id("not-an-id", "source")


def test_model_id_depends_on_weight_hash():
    a = mint_model_id(repo="r", revision="v1", weight_sha256="a" * 64)
    b = mint_model_id(repo="r", revision="v1", weight_sha256="b" * 64)
    assert a != b


def test_run_and_capture_ids_are_unique_events():
    assert new_run_id() != new_run_id()
    assert new_capture_id() != new_capture_id()


def test_development_cannot_read_held_out_role():
    policy = AccessPolicy("development")
    with pytest.raises(AccessViolation):
        policy.check_read("T_pilot")
    assert policy.violation_count == 1
    policy.check_read("C_pool")
    assert policy.violation_count == 1


def test_evaluation_can_read_all_roles():
    policy = AccessPolicy("evaluation")
    for role in ("C_pool", "D_sel", "T_pilot", "T_confirm", "TECH", "EXTERNAL"):
        policy.check_read(role)
    assert policy.violation_count == 0


def test_role_disjointness():
    assert roles_are_disjoint([{"g1", "g2"}, {"g3"}])
    assert not roles_are_disjoint([{"g1", "g2"}, {"g2"}])


def _score_row(**over):
    base = dict(
        run_id="run_0123456789abcdef01234567",
        sample_id="smp_0123456789abcdef01234567",
        source_id="src_0123456789abcdef01234567",
        split_role="TECH",
        environment_id="env0",
        model_id="mdl_0123456789abcdef01234567",
        image_branch="digital",
        transform_id="clean",
        raw_score=0.9,
        score_orientation="higher_is_ai",
        checkpoint_hash="c" * 64,
        preprocess_hash="p" * 64,
        status="ok",
        failure_code=None,
        is_fixture=False,
    )
    base.update(over)
    return base


def test_valid_score_row_passes():
    assert validate_row("scores", _score_row()) == []


def test_failed_row_requires_null_score_and_code():
    assert validate_row("scores", _score_row(status="failed", raw_score=None, failure_code="DECODE_ERROR")) == []
    errs = validate_row("scores", _score_row(status="failed", raw_score=0.0, failure_code="DECODE_ERROR"))
    assert any("must have raw_score=None" in e for e in errs)


def test_missing_failure_code_rejected():
    errs = validate_row("scores", _score_row(status="failed", raw_score=None, failure_code=None))
    assert errs


def test_probability_bounds_enforced():
    errs = validate_row("scores", _score_row(raw_score=1.5))
    assert any("probability" in e for e in errs)


def test_unknown_status_rejected():
    errs = validate_row("scores", _score_row(status="maybe"))
    assert errs
