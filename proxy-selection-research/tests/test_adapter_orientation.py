import pytest

from proxy_selection.adapters.hf_classifier import OrientationUndetermined, resolve_ai_index


def test_resolves_artificial_vs_human():
    assert resolve_ai_index({0: "artificial", 1: "human"}) == (0, "artificial")


def test_resolves_ai_vs_real():
    assert resolve_ai_index({0: "real", 1: "AI"}) == (1, "AI")


def test_resolves_fake_vs_real():
    assert resolve_ai_index({0: "fake", 1: "real"}) == (0, "fake")


def test_rejects_generic_labels():
    with pytest.raises(OrientationUndetermined):
        resolve_ai_index({0: "LABEL_0", 1: "LABEL_1"})


def test_override_label_is_honoured():
    assert resolve_ai_index({0: "LABEL_0", 1: "LABEL_1"}, override_label="LABEL_1") == (1, "LABEL_1")


def test_override_must_exist():
    with pytest.raises(OrientationUndetermined):
        resolve_ai_index({0: "LABEL_0", 1: "LABEL_1"}, override_label="AI")


def test_ambiguous_multiclass_rejected():
    with pytest.raises(OrientationUndetermined):
        resolve_ai_index({0: "generated", 1: "generated_hq", 2: "real"})
