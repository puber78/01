"""Rank-based AUC with correct tie handling, paired differences and grouped
bootstrap utilities.

Conventions
-----------
* ``y=1`` means AI-generated (the "positive" class); ``y=0`` means original
  photographic source.
* a score is a probability-like value oriented so that HIGHER means MORE
  AI-generated (``higher_is_ai``).  Adapters that emit the opposite orientation
  are inverted *before* reaching these functions; :func:`orient` does it once.
* AUC is the Mann-Whitney statistic with mid-ranks, so ties contribute 1/2 and
  the value lies in [0, 1] with 0.5 as chance.

No AUC here is used as a scientific pass/fail gate anywhere in the project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def orient(score: float, orientation: str) -> float:
    if orientation == "higher_is_ai":
        return float(score)
    if orientation == "higher_is_real":
        return 1.0 - float(score)
    raise ValueError(f"unknown orientation: {orientation}")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks with average ranks for ties (no scipy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # mean of 1-based ranks i+1 .. j
        ranks[order[i:j]] = avg
        i = j
    return ranks


def roc_auc(y: Sequence[int], s: Sequence[float]) -> float:
    """AUC with mid-rank ties. Raises if a class is missing."""
    y_arr = np.asarray(y, dtype=np.int64)
    s_arr = np.asarray(s, dtype=np.float64)
    if y_arr.shape != s_arr.shape:
        raise ValueError("y and s must have the same length")
    if not np.isfinite(s_arr).all():
        raise ValueError("scores must be finite")
    n1 = int((y_arr == 1).sum())
    n0 = int((y_arr == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC requires both classes present")
    ranks = rankdata_average(s_arr)
    auc = (ranks[y_arr == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def roc_auc_ci_normal(y: Sequence[int], s: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Hanley-McNeil normal approximation; used only for quick sanity output.

    Formal intervals in this project come from grouped bootstrap, not this.
    """
    from math import sqrt
    from statistics import NormalDist

    y_arr = np.asarray(y)
    auc = roc_auc(y_arr, s)
    n1 = int((y_arr == 1).sum())
    n0 = int((y_arr == 0).sum())
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    half = z * sqrt(max(var, 0.0))
    return (max(0.0, auc - half), min(1.0, auc + half))


@dataclass(frozen=True)
class PairedDelta:
    model_a: str
    model_b: str
    auc_a: float
    auc_b: float

    @property
    def delta(self) -> float:
        return self.auc_a - self.auc_b


def paired_model_delta(
    y: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    model_a: str,
    model_b: str,
) -> PairedDelta:
    """Delta = AUC(model_a) - AUC(model_b) on the same rows.

    When ``model_a == model_b`` the function returns exactly 0.0 because both
    sides are computed by the same deterministic expression on the same data.
    """
    a = roc_auc(y, scores[model_a])
    b = roc_auc(y, scores[model_b])
    return PairedDelta(model_a, model_b, a, b)


def group_bootstrap_indices(
    group_ids: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Resample whole source groups (never individual images) with replacement.

    Returns index arrays into the original rows, so paired models stay paired and
    multi-image lineages are never split.
    """
    groups = np.asarray(list(group_ids))
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    out: list[np.ndarray] = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in pick])
        out.append(idx)
    return out


def paired_bootstrap_delta(
    y: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    model_a: str,
    model_b: str,
    group_ids: Sequence[str],
    n_boot: int = 2000,
    seed: int = 20261026,
) -> dict:
    """Grouped bootstrap interval for AUC(a)-AUC(b), preserving pairing."""
    y_arr = np.asarray(y)
    idxs = group_bootstrap_indices(group_ids, n_boot, np.random.default_rng(seed))
    deltas = np.empty(n_boot, dtype=np.float64)
    skipped = 0
    for i, idx in enumerate(idxs):
        yy = y_arr[idx]
        if yy.min() == yy.max():
            deltas[i] = np.nan
            skipped += 1
            continue
        sa = np.asarray(scores[model_a], dtype=float)[idx]
        sb = np.asarray(scores[model_b], dtype=float)[idx]
        deltas[i] = roc_auc(yy, sa) - roc_auc(yy, sb)
    valid = deltas[np.isfinite(deltas)]
    if valid.size == 0:
        return {"delta": float("nan"), "ci_low": None, "ci_high": None, "n_valid": 0, "n_skipped": skipped}
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "delta": float(roc_auc(y_arr, scores[model_a]) - roc_auc(y_arr, scores[model_b])),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_valid": int(valid.size),
        "n_skipped": skipped,
        "n_groups": int(len(set(group_ids))),
    }
