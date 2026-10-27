import numpy as np
import pytest

from proxy_selection.metrics.auc import (
    orient,
    paired_model_delta,
    rankdata_average,
    roc_auc,
    roc_auc_ci_normal,
)


def test_auc_perfect_ordering():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y, s) == pytest.approx(1.0)


def test_auc_reversed_scores_give_zero():
    y = [0, 0, 1, 1]
    s = [0.9, 0.8, 0.2, 0.1]
    assert roc_auc(y, s) == pytest.approx(0.0)


def test_auc_with_ties_is_half():
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_auc_known_value_with_partial_overlap():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    # pairs: (0.35>0.1), (0.35<0.4), (0.8>0.1), (0.8>0.4) -> 3/4
    assert roc_auc(y, s) == pytest.approx(0.75)


def test_rankdata_average_ties():
    ranks = rankdata_average(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_orientation_inversion_restores_direction():
    y = [0, 0, 1, 1]
    raw = [0.9, 0.8, 0.2, 0.1]  # "higher means real"
    inverted = [orient(v, "higher_is_real") for v in raw]
    assert roc_auc(y, inverted) == pytest.approx(1.0)


def test_orientation_rejects_unknown():
    with pytest.raises(ValueError):
        orient(0.3, "sideways")


def test_same_model_delta_is_exactly_zero(synthetic_two_model_scores):
    d = synthetic_two_model_scores
    delta = paired_model_delta(d["y"], {"a": d["s_a"], "b": d["s_a"].copy()}, "a", "b")
    assert delta.delta == 0.0  # exact, no tolerance


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_auc_rejects_nonfinite():
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, float("nan")])


def test_normal_ci_contains_point_estimate(synthetic_two_model_scores):
    d = synthetic_two_model_scores
    lo, hi = roc_auc_ci_normal(d["y"], d["s_a"])
    point = roc_auc(d["y"], d["s_a"])
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0
