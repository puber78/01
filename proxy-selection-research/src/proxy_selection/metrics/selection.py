"""Selection rules, regret and the regret-difference identity.

Definitions (per the execution outline §3.1, §5):

    m_hat_S  = argmax_{m in M} S(m)          (frozen tie rule)
    R(S)     = max_m A(m) - A(m_hat_S)       (candidate-set selection loss)

For two rules on the same candidate set and target:

    R(S1) - R(S2) = A(m_hat_S2) - A(m_hat_S1)

which is what the project analyses: the paired physical AUC difference between
the models the two rules select.  If both rules select the same model the
difference is exactly 0.0, by construction and not by tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    selected_model: str
    scores: Mapping[str, float]
    max_score: float
    tied_models: tuple[str, ...]


def frozen_tie_break(candidates: Sequence[str], tie_order: Sequence[str]) -> str:
    """Pick the first candidate in the frozen priority order.

    ``tie_order`` must cover every candidate; a candidate missing from it is an
    error, because an unspecified tie would be an implicit decision.
    """
    order = list(tie_order)
    for c in order:
        if c in candidates:
            return c
    raise SelectionError(f"no candidate from {list(candidates)} appears in tie order")


def select_model(
    scores: Mapping[str, float],
    rule_id: str,
    tie_order: Sequence[str],
    *,
    tol: float = 0.0,
) -> RuleResult:
    if not scores:
        raise SelectionError("empty candidate set")
    best = max(scores.values())
    tied = tuple(sorted(m for m, s in scores.items() if s >= best - tol))
    chosen = frozen_tie_break(tied, tie_order) if len(tied) > 1 else tied[0]
    return RuleResult(rule_id, chosen, dict(scores), float(best), tied)


def selection_loss(aucs: Mapping[str, float], selected: str) -> float:
    return float(max(aucs.values()) - aucs[selected])


def regret_difference(
    aucs: Mapping[str, float],
    selected_s1: str,
    selected_s2: str,
) -> tuple[float, float, float]:
    """Return (R1-R2, A(m2)-A(m1), residual); the residual must be 0 up to fp error."""
    r1 = selection_loss(aucs, selected_s1)
    r2 = selection_loss(aucs, selected_s2)
    rhs = aucs[selected_s2] - aucs[selected_s1]
    return float(r1 - r2), float(rhs), float((r1 - r2) - rhs)


def identity_holds(
    aucs: Mapping[str, float],
    selected_s1: str,
    selected_s2: str,
    atol: float = 1e-12,
) -> bool:
    diff, rhs, residual = regret_difference(aucs, selected_s1, selected_s2)
    return abs(residual) <= atol


def rank_models(aucs: Mapping[str, float], tie_order: Sequence[str]) -> list[str]:
    ordered = sorted(aucs.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        tied = [m for m, _ in ordered[i:j]]
        while tied:
            pick = frozen_tie_break(tied, tie_order)
            out.append(pick)
            tied.remove(pick)
        i = j
    return out


def top_k_overlap(aucs: Mapping[str, float], digital_scores: Mapping[str, float], k: int = 1) -> int:
    by_phys = [m for m, _ in sorted(aucs.items(), key=lambda kv: (-kv[1], kv[0]))][:k]
    by_dig = [m for m, _ in sorted(digital_scores.items(), key=lambda kv: (-kv[1], kv[0]))][:k]
    return len(set(by_phys) & set(by_dig))


def spearman(values_a: Mapping[str, float], values_b: Mapping[str, float]) -> float:
    keys = sorted(set(values_a) & set(values_b))
    if len(keys) < 2:
        raise ValueError("need at least two common models")
    a = np.array([values_a[k] for k in keys], dtype=float)
    b = np.array([values_b[k] for k in keys], dtype=float)
    ra = _average_rank(a)
    rb = _average_rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _average_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks
