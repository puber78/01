"""Attainable-set diagnostics for the non-negative weighting family.

Given a candidate x digital-family AUC matrix ``A`` (models x families), the
weighting rule is ``S_w(m) = sum_j w_j a_{m,j}`` with ``w`` in the simplex.
A model is *attainable* if it maximises ``S_w`` for at least one simplex weight:

    M_D = { m : exists w in Delta_J with m in argmax_n S_w(n) }

The membership test is a linear program (per candidate):

    max_{w,t}  t   s.t.  (a_m - a_n) . w >= t  for every n,  sum_j w_j = 1, w >= 0

``m`` is attainable iff the optimum ``t* >= -tol``.  A strictly positive
optimum means ``m`` is the unique maximiser on some face of the simplex; a zero
optimum means it ties with another model.

This is a *diagnostic* of the stated family. It says nothing about the physical
optimum and nothing about families outside the simplex (e.g. non-linear rules).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linprog


class AttainableError(ValueError):
    pass


@dataclass(frozen=True)
class AttainabilityResult:
    model: str
    optimal_margin: float
    attainable: bool
    weights: tuple[float, ...]
    lp_status: int
    tolerance: float


def _as_matrix(aucs: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    models = list(aucs.keys())
    if not models:
        raise AttainableError("empty candidate set")
    mat = np.vstack([np.asarray(aucs[m], dtype=float) for m in models])
    if mat.ndim != 2 or mat.shape[0] != len(models):
        raise AttainableError("AUC matrix must be models x families")
    if not np.isfinite(mat).all():
        raise AttainableError("AUC matrix must be finite")
    return models, mat


def component_dominates(a_m: np.ndarray, a_n: np.ndarray, strict: bool = True) -> bool:
    """True if m is >= n on every family (and > on at least one when strict)."""
    if strict:
        return bool(np.all(a_m >= a_n) and np.any(a_m > a_n))
    return bool(np.all(a_m >= a_n))


def dominated_pairs(aucs: Mapping[str, np.ndarray]) -> list[tuple[str, str]]:
    models, mat = _as_matrix(aucs)
    out = []
    for i, m in enumerate(models):
        for j, n in enumerate(models):
            if i != j and component_dominates(mat[i], mat[j]):
                out.append((m, n))
    return out


def weighted_scores(aucs: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
    models, mat = _as_matrix(aucs)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != mat.shape[1]:
        raise AttainableError(f"weights must have length {mat.shape[1]}")
    if (w < -1e-12).any() or abs(w.sum() - 1.0) > 1e-9:
        raise AttainableError("weights must lie on the simplex")
    return {m: float(mat[i] @ w) for i, m in enumerate(models)}


def attainability_lp(
    aucs: Mapping[str, np.ndarray],
    model: str,
    *,
    tolerance: float = 1e-9,
) -> AttainabilityResult:
    models, mat = _as_matrix(aucs)
    if model not in models:
        raise AttainableError(f"unknown model {model}")
    a_m = mat[models.index(model)]
    others = [mat[i] for i, n in enumerate(models) if n != model]
    n_fam = mat.shape[1]
    # variables: [w_1..w_J, t]; minimise -t
    c = np.zeros(n_fam + 1)
    c[-1] = -1.0
    # (a_m - a_n).w - t >= 0  ->  -(a_m-a_n).w + t <= 0
    a_ub = []
    b_ub = []
    for a_n in others:
        row = np.zeros(n_fam + 1)
        row[:n_fam] = -(a_m - a_n)
        row[-1] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)
    a_eq = np.zeros((1, n_fam + 1))
    a_eq[0, :n_fam] = 1.0
    bounds = [(0.0, None)] * n_fam + [(None, None)]
    res = linprog(
        c,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise AttainableError(f"LP failed for {model}: {res.message}")
    t_star = float(res.x[-1])
    w = np.clip(res.x[:n_fam], 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()
    return AttainabilityResult(
        model=model,
        optimal_margin=t_star,
        attainable=bool(t_star >= -tolerance),
        weights=tuple(float(x) for x in w),
        lp_status=int(res.status),
        tolerance=tolerance,
    )


def attainable_set(aucs: Mapping[str, np.ndarray], *, tolerance: float = 1e-9) -> list[str]:
    out = []
    for m in aucs:
        if attainability_lp(aucs, m, tolerance=tolerance).attainable:
            out.append(m)
    return out


def unattainable_candidates(
    aucs: Mapping[str, np.ndarray], *, tolerance: float = 1e-9
) -> list[str]:
    """Candidates that cannot be selected by ANY simplex weight."""
    att = set(attainable_set(aucs, tolerance=tolerance))
    return [m for m in aucs if m not in att]


def finite_grid_attainable(
    aucs: Mapping[str, np.ndarray], grid: np.ndarray, *, tolerance: float = 1e-12
) -> tuple[set[str], dict[str, int]]:
    """Attainable set when only a FINITE weight grid is allowed.

    Returns (attainable models, selection counts per model over the grid).  This
    is reported separately from the full-simplex result because a finite grid
    must never be described as covering the whole simplex.
    """
    models, mat = _as_matrix(aucs)
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.shape[1] != mat.shape[1]:
        raise AttainableError("grid must be n_points x n_families")
    counts = {m: 0 for m in models}
    winners: set[str] = set()
    for w in g:
        s = mat @ w
        best = s.max()
        for i, m in enumerate(models):
            if s[i] >= best - tolerance:
                winners.add(m)
        counts[models[int(np.argmax(s))]] += 1
    return winners, counts
