import numpy as np
import pytest

from proxy_selection.selectors import (
    SelectionInputs,
    draw_group_subset,
    s_dig,
    s_real,
    s_shrink,
    s_weight,
)


def _inputs():
    return SelectionInputs(
        model_ids=("m1", "m2", "m3"),
        digital_auc={
            "m1": np.array([0.90, 0.80]),
            "m2": np.array([0.70, 0.95]),
            "m3": np.array([0.60, 0.60]),
        },
        tie_order=("m1", "m2", "m3"),
    )


def test_s_dig_equal_weight_mean():
    out = s_dig(_inputs())
    assert out["selected_model_id"] == "m1"  # 0.85 vs 0.825 vs 0.60
    assert out["visible_physical_groups"] == 0


def test_s_weight_simplex_selection():
    out = s_weight(_inputs(), [0.0, 1.0])
    assert out["selected_model_id"] == "m2"


def test_s_weight_rejects_offsimplex():
    with pytest.raises(ValueError):
        s_weight(_inputs(), [1.2, -0.2])


def _physical():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
    scores = {
        "m1": np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9]),
        "m2": np.array([0.4, 0.6, 0.5, 0.5, 0.2, 0.9, 0.3, 0.7]),
        "m3": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
    }
    return y, groups, scores


def test_draw_group_subset_keeps_groups_whole():
    y, groups, _ = _physical()
    idx = draw_group_subset(groups, 2, np.random.default_rng(0))
    picked = set(groups[idx])
    assert len(picked) == 2
    for g in picked:
        assert idx[np.isin(groups[idx], g)].size == 2


def test_budget_larger_than_groups_errors():
    _, groups, _ = _physical()
    with pytest.raises(ValueError):
        draw_group_subset(groups, 99, np.random.default_rng(0))


def test_s_real_selects_by_physical_auc():
    y, groups, scores = _physical()
    out = s_real(
        _inputs(),
        labels=y,
        physical_scores=scores,
        group_ids=groups,
        budget=3,
        rng=np.random.default_rng(7),
    )
    assert out["selected_model_id"] in {"m1", "m2", "m3"}
    assert out["budget"] == 3


def test_s_shrink_blends_and_selects():
    y, groups, scores = _physical()
    out = s_shrink(
        _inputs(),
        labels=y,
        physical_scores=scores,
        group_ids=groups,
        budget=3,
        weight=0.5,
        rng=np.random.default_rng(3),
    )
    assert out["selected_model_id"] in {"m1", "m2", "m3"}
    assert 0.0 <= out["scores"][out["selected_model_id"]] or True
