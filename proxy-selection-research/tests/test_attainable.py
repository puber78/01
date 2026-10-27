import numpy as np
import pytest

from proxy_selection.metrics.attainable import (
    AttainableError,
    attainable_set,
    component_dominates,
    dominated_pairs,
    finite_grid_attainable,
    unattainable_candidates,
    weighted_scores,
)


def test_component_dominance_detected(synthetic_dominant_matrix):
    m = synthetic_dominant_matrix
    assert component_dominates(m["A"], m["B"])
    assert not component_dominates(m["B"], m["A"])
    assert ("A", "B") in dominated_pairs(m)


def test_dominance_is_strict():
    a = np.array([0.9, 0.8])
    assert not component_dominates(a, a.copy())


def test_review_example_c_is_not_attainable(synthetic_digital_matrix):
    m = synthetic_digital_matrix
    att = set(attainable_set(m))
    assert att == {"A", "B"}
    assert unattainable_candidates(m) == ["C"]


def test_c_weighted_score_never_exceeds_best(synthetic_digital_matrix):
    m = synthetic_digital_matrix
    for w in [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2), (1.0, 0.0), (0.0, 1.0)]:
        s = weighted_scores(m, np.array(w))
        assert max(s["A"], s["B"]) >= s["C"] - 1e-12


def test_dominated_candidate_unattainable(synthetic_dominant_matrix):
    att = set(attainable_set(synthetic_dominant_matrix))
    assert att == {"A"}


def test_finite_grid_may_miss_simplex_attainable_models(synthetic_digital_matrix):
    # The single 50/50 weight ties A and B (both 0.75) and never reaches C (0.70);
    # a finite grid must not be described as covering the whole simplex.
    grid = np.array([[0.5, 0.5]])
    winners, counts = finite_grid_attainable(synthetic_digital_matrix, grid)
    assert winners == {"A", "B"}
    assert "C" not in winners
    assert winners <= set(attainable_set(synthetic_digital_matrix))


def test_finite_grid_counts_are_consistent(synthetic_digital_matrix):
    grid = np.array([[1.0, 0.0], [0.0, 1.0]])
    winners, counts = finite_grid_attainable(synthetic_digital_matrix, grid)
    assert winners == {"A", "B"}
    assert counts["A"] == 1 and counts["B"] == 1 and counts["C"] == 0


def test_weight_validation_rejects_offsimplex(synthetic_digital_matrix):
    with pytest.raises(AttainableError):
        weighted_scores(synthetic_digital_matrix, np.array([0.5, 0.6]))
    with pytest.raises(AttainableError):
        weighted_scores(synthetic_digital_matrix, np.array([1.5, -0.5]))
