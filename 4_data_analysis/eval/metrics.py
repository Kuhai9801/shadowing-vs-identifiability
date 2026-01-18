"""Evaluation metrics and symbolic labeling.

This module implements the metrics specified in the plan:

  * Imputation error (historical truth):
        e^(j) = || E( z_hat_{q-1}^{(j)} ) - E( z_{q-1}^{true} ) ||.

  * Defect / dynamical validity:
        δ_E^(j) = max_n || E(z_hat_{n+1}) - E(A z_hat_n) ||
    (embedding-space). Additionally, for rigorous shadowing certification we compute
        δ_T^(j) = max_n d_{T^2}( frac(z_hat_{n+1}), frac( A frac(z_hat_n) ) ).

  * Certificate radius:
        ε_cert^(j) = C(A) δ_T^(j).

  * Symbolic itinerary and normalized Hamming distance for coloring.

All computations are deterministic given fixed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from systems.torus_map import (
    E,
    E_x,
    chordal_dist,
    dT2,
    frac,
    splitting_data,
)


@dataclass(frozen=True)
class HypothesisMetrics:
    """Metrics associated with one hypothesis."""

    e_impute: float
    e_impute_x: float
    defect_E_max: float
    defect_T_max: float
    eps_cert: float
    d_H: float


def embedding_defect_max(A: np.ndarray, z_hat: np.ndarray) -> float:
    """Compute δ_E = max_n ||E(z_{n+1}) - E(A z_n)|| for a candidate segment.

    Args:
        A: 2x2 integer matrix.
        z_hat: array of shape (q+1,2) in R^2 (not modded).

    Returns:
        Maximum embedding residual norm.
    """
    A = np.asarray(A, dtype=float)
    z_hat = np.asarray(z_hat, dtype=float)
    if z_hat.ndim != 2 or z_hat.shape[1] != 2:
        raise ValueError("z_hat must have shape (q+1,2).")
    if z_hat.shape[0] < 2:
        raise ValueError("z_hat must contain at least two time steps.")

    zA = z_hat[:-1] @ A.T
    res = E(z_hat[1:]) - E(zA)
    norms = np.linalg.norm(res, axis=-1)
    return float(np.max(norms))


def torus_defect_max(A: np.ndarray, z_hat: np.ndarray) -> float:
    """Compute δ_T = max_n d_{T^2}( frac(z_{n+1}), f_A(frac(z_n)) )."""
    A = np.asarray(A, dtype=float)
    z_hat = np.asarray(z_hat, dtype=float)
    if z_hat.ndim != 2 or z_hat.shape[1] != 2:
        raise ValueError("z_hat must have shape (q+1,2).")
    z_mod = frac(z_hat)
    if z_mod.shape[0] < 2:
        raise ValueError("z_hat must contain at least two time steps.")
    # Vectorized forward map on the torus: f_A(z_n) = frac(A z_n).
    # Use row-vector convention: (q,2) @ A.T produces (q,2).
    f_mod = frac(z_mod[:-1] @ A.T)
    d = dT2(z_mod[1:], f_mod)
    return float(np.max(d))

def imputation_error(A: np.ndarray, z_hat: np.ndarray, traj_true: np.ndarray, *, t_index: int) -> float:
    """Compute embedding imputation error at a specified time index."""
    z_hat = np.asarray(z_hat, dtype=float)
    traj_true = np.asarray(traj_true, dtype=float)
    if not (0 <= t_index < z_hat.shape[0]):
        raise ValueError("t_index out of range.")
    if traj_true.shape[0] <= t_index:
        raise ValueError("traj_true shorter than required t_index.")
    return float(chordal_dist(E(z_hat[t_index]), E(traj_true[t_index]), squared=False))


def imputation_error_x(z_hat: np.ndarray, traj_true: np.ndarray, *, t_index: int) -> float:
    """Compute x-only embedding imputation error at a specified time index.

    This is the natural quantity for the branch-counting and minimax arguments,
    which are stated on T^1 for the observed coordinate x. We report the chordal
    distance on the unit circle:

        ||E_x(x_hat) - E_x(x_true)||_2.
    """
    z_hat = np.asarray(z_hat, dtype=float)
    traj_true = np.asarray(traj_true, dtype=float)
    if not (0 <= t_index < z_hat.shape[0]):
        raise ValueError("t_index out of range.")
    if traj_true.shape[0] <= t_index:
        raise ValueError("traj_true shorter than required t_index.")
    return float(chordal_dist(E_x(z_hat[t_index, 0]), E_x(traj_true[t_index, 0]), squared=False))


def symbolic_itinerary(A: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
    """Compute symbolic itinerary m_n = floor(A frac(z_n)) \in Z^2 for n=0,...,q-1.

    Args:
        A: 2x2 integer matrix.
        z_seq: shape (q+1,2) trajectory; will be reduced mod 1 for the computation.

    Returns:
        Array m of shape (q,2) with integer entries.
    """
    A = np.asarray(A, dtype=float)
    z_mod = frac(np.asarray(z_seq, dtype=float))
    if z_mod.ndim != 2 or z_mod.shape[1] != 2:
        raise ValueError("z_seq must have shape (q+1,2).")
    if z_mod.shape[0] < 2:
        return np.empty((0, 2), dtype=int)
    # Vectorized: m_n = floor(A z_n) with z_n in [0,1)^2.
    Az = z_mod[:-1] @ A.T
    m = np.floor(Az).astype(int)
    return m

def normalized_hamming(m1: np.ndarray, m2: np.ndarray) -> float:
    """Normalized Hamming distance between two integer-symbol sequences."""
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    if m1.shape != m2.shape:
        raise ValueError("Symbol sequences must have identical shapes.")
    q = m1.shape[0]
    if q == 0:
        return 0.0
    mism = np.any(m1 != m2, axis=-1)
    return float(np.mean(mism.astype(float)))


def compute_metrics(
    *,
    A: np.ndarray,
    z_hat: np.ndarray,
    traj_true: np.ndarray,
    t_impute: int,
    m_true: Optional[np.ndarray] = None,
    C: Optional[float] = None,
) -> HypothesisMetrics:
    """Compute all metrics for a hypothesis."""
    # Allow callers (the experiment runners) to precompute invariants that are reused
    # many times per instance. This substantially reduces wall time without changing
    # the definition of any metric.
    if C is None:
        C = float(splitting_data(A).C)
    if m_true is None:
        m_true = symbolic_itinerary(A, traj_true)

    e = imputation_error(A, z_hat, traj_true, t_index=t_impute)
    ex = imputation_error_x(z_hat, traj_true, t_index=t_impute)
    dE = embedding_defect_max(A, z_hat)
    dT = torus_defect_max(A, z_hat)
    eps_cert = float(float(C) * dT)

    m_hat = symbolic_itinerary(A, z_hat)
    d_H = normalized_hamming(m_hat, m_true)

    return HypothesisMetrics(
        e_impute=float(e),
        e_impute_x=float(ex),
        defect_E_max=float(dE),
        defect_T_max=float(dT),
        eps_cert=float(eps_cert),
        d_H=float(d_H),
    )


def best_of_k_success(
    errors: Sequence[float],
    scores: Sequence[float],
    eps: float,
    K_list: Sequence[int],
) -> Dict[int, int]:
    """Compute best-of-K success indicators using solver-available scores.

    The procedure is:
      1) Sort hypotheses by increasing solver score (e.g., total loss or residual).
      2) For each K, declare success iff min_{j \le K} error_j < eps.

    Args:
        errors: imputation errors e^(j).
        scores: solver scores used for ordering.
        eps: global success threshold.
        K_list: iterable of K values.

    Returns:
        Dict K -> {0,1}.
    """
    if len(errors) != len(scores):
        raise ValueError("errors and scores must have the same length.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    order = np.argsort(np.asarray(scores, dtype=float))
    errs_sorted = np.asarray(errors, dtype=float)[order]

    out: Dict[int, int] = {}
    for K in K_list:
        if K < 1:
            raise ValueError("K must be >= 1.")
        K_eff = min(int(K), len(errs_sorted))
        succ = 1 if float(np.min(errs_sorted[:K_eff])) < eps else 0
        out[int(K)] = int(succ)
    return out



def best_of_k_success_symbolic(
    d_H: Sequence[float],
    scores: Sequence[float],
    K_list: Sequence[int],
    *,
    tol: float = 0.0,
) -> Dict[int, int]:
    """Compute best-of-K success indicators using symbolic itinerary agreement.

    This success definition is the correct one for the *branch counting* /
    information-theoretic barrier plots: in the noiseless endpoint regime,
    there is exactly one true history among |b_q| admissible branches, and
    d_H=0 is (almost surely) equivalent to selecting that branch.

    Procedure:
      1) Sort hypotheses by increasing solver score (loss/residual).
      2) For each K, declare success iff min_{j \le K} d_H^{(j)} \le tol.

    Args:
        d_H: normalized Hamming distances (0 means exact symbolic match).
        scores: solver scores used for ordering.
        K_list: iterable of K values.
        tol: tolerance on d_H for declaring a match. Default 0.0 (exact match).

    Returns:
        Dict K -> {0,1}.
    """
    if len(d_H) != len(scores):
        raise ValueError("d_H and scores must have the same length.")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative.")

    order = np.argsort(np.asarray(scores, dtype=float))
    dh_sorted = np.asarray(d_H, dtype=float)[order]

    out: Dict[int, int] = {}
    for K in K_list:
        if K < 1:
            raise ValueError("K must be >= 1.")
        K_eff = min(int(K), len(dh_sorted))
        succ = 1 if float(np.min(dh_sorted[:K_eff])) <= float(tol) else 0
        out[int(K)] = int(succ)
    return out
