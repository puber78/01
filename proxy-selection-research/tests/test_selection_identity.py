import pytest

from proxy_selection.metrics.selection import (
    SelectionError,
    frozen_tie_break,
    identity_holds,
    rank_models,
    regret_difference,
    select_model,
    selection_loss,
    spearman,
)


def test_regret_identity_holds_on_example():
    aucs = {"A": 0.65, "B": 0.78}
    diff, rhs, residual = regret_difference(aucs, "A", "B")
    assert diff == pytest.approx(0.13)
    assert rhs == pytest.approx(0.13)
    assert residual == pytest.approx(0.0)
    assert identity_holds(aucs, "A", "B")


def test_regret_identity_zero_for_same_selection():
    aucs = {"A": 0.65, "B": 0.78}
    assert regret_difference(aucs, "B", "B") == (0.0, 0.0, 0.0)


def test_selection_loss_definition():
    aucs = {"A": 0.65, "B": 0.78}
    assert selection_loss(aucs, "A") == pytest.approx(0.13)
    assert selection_loss(aucs, "B") == pytest.approx(0.0)


def test_tie_break_requires_full_priority():
    with pytest.raises(SelectionError):
        frozen_tie_break(("X", "Y"), ["Z"])


def test_tie_break_is_deterministic():
    assert frozen_tie_break(("mdl_b", "mdl_a", "mdl_c"), ["mdl_c", "mdl_b", "mdl_a"]) == "mdl_c"
    assert frozen_tie_break(("mdl_a", "mdl_b"), ["mdl_b", "mdl_a"]) == "mdl_b"


def test_select_model_zero_tolerance_for_distinct_scores():
    res = select_model({"a": 0.7, "b": 0.8}, "S", ["a", "b"])
    assert res.selected_model == "b"
    assert res.tied_models == ("b",)


def test_select_model_tolerance_marks_ties():
    res = select_model({"a": 0.7000000001, "b": 0.7}, "S", ["a", "b"], tol=1e-9)
    assert set(res.tied_models) == {"a", "b"}
    assert res.selected_model == "a"


def test_rank_models_orders_and_breaks_ties():
    assert rank_models({"a": 0.7, "b": 0.9, "c": 0.7}, ["b", "a", "c"]) == ["b", "a", "c"]


def test_spearman_identity_and_reversal():
    a = {"m1": 0.1, "m2": 0.2, "m3": 0.3}
    assert spearman(a, a) == pytest.approx(1.0)
    b = {"m1": 0.3, "m2": 0.2, "m3": 0.1}
    assert spearman(a, b) == pytest.approx(-1.0)
