"""Selection rules operating on frozen digital/physical evidence.

All rules obey the same information-rights table (``contracts.roles``) and the
same frozen tie rule.  They are plain, auditable functions -- no learned network,
no per-image routing.

Rules
-----
S_dig        equal-weight mean of per-family digital AUC; reads NO physical data
S_real(b)    physical AUC on a size-b calibration subset sampled by group
S_shrink(b)  fixed 0.5 blend of z-scored digital and physical selection scores
S_weight(b)  simplex weights over family AUCs (experimental; see attainable set)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .metrics.auc import roc_auc
from .metrics.selection import select_model


@dataclass(frozen=True)
class SelectionInputs:
    """Everything a rule is allowed to see.

    ``digital_auc``: model -> per-family AUC vector (no physical information)
    ``physical``:    optional per-row data for a calibration subset
    """

    model_ids: tuple[str, ...]
    digital_auc: Mapping[str, np.ndarray]
    tie_order: tuple[str, ...]


def _check_models(model_ids: Sequence[str], tie_order: Sequence[str]) -> None:
    missing = [m for m in model_ids if m not in tie_order]
    if missing:
        raise ValueError(f"tie order does not cover models: {missing}")


def s_dig(inputs: SelectionInputs, rule_id: str = "S_dig") -> dict:
    _check_models(inputs.model_ids, inputs.tie_order)
    scores = {m: float(np.mean(np.asarray(inputs.digital_auc[m], dtype=float))) for m in inputs.model_ids}
    res = select_model(scores, rule_id, inputs.tie_order)
    return {
        "rule_id": rule_id,
        "selected_model_id": res.selected_model,
        "scores": scores,
        "tied_models": list(res.tied_models),
        "budget": 0,
        "visible_physical_groups": 0,
    }


def _physical_auc_on_subset(
    subset: np.ndarray, labels: np.ndarray, scores: Mapping[str, np.ndarray]
) -> dict[str, float]:
    y = labels[subset]
    if y.min() == y.max():
        raise ValueError("calibration subset contains a single class; invalid draw")
    return {m: roc_auc(y, np.asarray(scores[m])[subset]) for m in scores}


def s_real(
    inputs: SelectionInputs,
    *,
    labels: np.ndarray,
    physical_scores: Mapping[str, np.ndarray],
    group_ids: np.ndarray,
    budget: int,
    rng: np.random.Generator | None = None,
    rule_id: str = "S_real",
) -> dict:
    """Direct physical selection on a size-``budget`` subset of source groups."""
    _check_models(inputs.model_ids, inputs.tie_order)
    rng = rng or np.random.default_rng(0)
    subset = draw_group_subset(group_ids, budget, rng)
    aucs = _physical_auc_on_subset(subset, labels, physical_scores)
    res = select_model(aucs, rule_id, inputs.tie_order)
    return {
        "rule_id": rule_id,
        "selected_model_id": res.selected_model,
        "scores": aucs,
        "tied_models": list(res.tied_models),
        "budget": int(budget),
        "visible_physical_groups": int(len(np.unique(group_ids[subset]))),
        "subset_hash": None,
    }


def draw_group_subset(group_ids: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Return row indices for ``budget`` distinct source groups (all their rows)."""
    groups = np.unique(group_ids)
    if budget > len(groups):
        raise ValueError(f"budget {budget} exceeds available groups {len(groups)}")
    chosen = rng.choice(groups, size=budget, replace=False)
    mask = np.isin(group_ids, chosen)
    return np.flatnonzero(mask)


def _zscore(x: Mapping[str, float]) -> dict[str, float]:
    vals = np.array(list(x.values()), dtype=float)
    sd = vals.std(ddof=0)
    mu = vals.mean()
    if sd == 0:
        return {m: 0.0 for m in x}
    return {m: float((v - mu) / sd) for m, v in x.items()}


def s_shrink(
    inputs: SelectionInputs,
    *,
    labels: np.ndarray,
    physical_scores: Mapping[str, np.ndarray],
    group_ids: np.ndarray,
    budget: int,
    weight: float = 0.5,
    rng: np.random.Generator | None = None,
    rule_id: str = "S_shrink",
) -> dict:
    """0.5 * z(digital mean) + 0.5 * z(physical AUC on C_b)."""
    rng = rng or np.random.default_rng(0)
    subset = draw_group_subset(group_ids, budget, rng)
    phys = _physical_auc_on_subset(subset, labels, physical_scores)
    dig = {m: float(np.mean(np.asarray(inputs.digital_auc[m], dtype=float))) for m in inputs.model_ids}
    zd, zp = _zscore(dig), _zscore(phys)
    blended = {m: (1 - weight) * zd[m] + weight * zp[m] for m in inputs.model_ids}
    res = select_model(blended, rule_id, inputs.tie_order)
    return {
        "rule_id": rule_id,
        "selected_model_id": res.selected_model,
        "scores": blended,
        "tied_models": list(res.tied_models),
        "budget": int(budget),
        "visible_physical_groups": int(len(np.unique(group_ids[subset]))),
    }


def s_weight(
    inputs: SelectionInputs,
    weights: Sequence[float],
    *,
    rule_id: str = "S_weight",
) -> dict:
    _check_models(inputs.model_ids, inputs.tie_order)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(next(iter(inputs.digital_auc.values()))):
        raise ValueError("weight length must equal number of digital families")
    if (w < -1e-12).any() or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must lie on the simplex")
    scores = {m: float(np.asarray(inputs.digital_auc[m], dtype=float) @ w) for m in inputs.model_ids}
    res = select_model(scores, rule_id, inputs.tie_order)
    return {
        "rule_id": rule_id,
        "selected_model_id": res.selected_model,
        "scores": scores,
        "tied_models": list(res.tied_models),
        "budget": 0,
        "visible_physical_groups": 0,
        "weights": [float(x) for x in w],
    }
