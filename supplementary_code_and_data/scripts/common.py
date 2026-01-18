"""Shared utilities for experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import math
import numpy as np

from systems.torus_map import (
    bq,
    check_sl2z_hyperbolic,
    embedding_circle_distance_from_torus_distance,
    interior_x_separation,
)


def parse_int_list(spec: str) -> List[int]:
    """Parse a comma-separated list of integers.

    Examples:
        "4,5,6" -> [4, 5, 6]
        " 1, 2 ,10" -> [1, 2, 10]

    Raises:
        ValueError: if the spec is empty or contains non-integers.
    """
    parts = [p.strip() for p in str(spec).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("List spec is empty.")
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception as e:
            raise ValueError(f"Invalid integer in list spec: {p!r}") from e
        out.append(v)
    return out


def suggest_q_list_for_bq_range(
    matrices: Dict[str, np.ndarray],
    *,
    bq_min: int,
    bq_max: int,
    q_max: int = 30,
    require_all_matrices: bool = False,
) -> List[int]:
    """Suggest a q-list such that |b_q| lies in a target range.

    Money Plot 1 becomes informative only when the x-axis quantity K/|b_q| spans a
    nontrivial range (not almost always ~0). A simple practical proxy is to choose
    horizons q for which |b_q| is comparable to the K values under study.

    This helper scans q=1..q_max and returns the sorted set of q values where
    |b_q| is within [bq_min, bq_max] for at least one matrix (default) or for all
    matrices (if require_all_matrices=True).

    Args:
        matrices: Dict of matrix names to matrices.
        bq_min: Lower bound on |b_q| (inclusive).
        bq_max: Upper bound on |b_q| (inclusive).
        q_max: Maximum q to scan.
        require_all_matrices: If True, keep q only if all matrices satisfy the range.

    Returns:
        Sorted list of q values.
    """
    if bq_min < 1 or bq_max < bq_min:
        raise ValueError("Require 1 <= bq_min <= bq_max.")
    if q_max < 1:
        raise ValueError("q_max must be >= 1.")

    # Validate matrices once.
    mats = list(matrices.values())
    if not mats:
        raise ValueError("matrices is empty.")
    for A in mats:
        check_sl2z_hyperbolic(A)

    keep: List[int] = []
    for q in range(1, int(q_max) + 1):
        vals = [abs(int(bq(A, q))) for A in mats]
        if require_all_matrices:
            ok = all((bq_min <= v <= bq_max) for v in vals)
        else:
            ok = any((bq_min <= v <= bq_max) for v in vals)
        if ok:
            keep.append(int(q))
    return sorted(set(keep))


def default_K_list(K_max: int) -> List[int]:
    """Return the canonical K grid {1,2,4,...,K_max} (powers of two)."""
    if K_max < 1:
        raise ValueError("K_max must be >= 1")
    K = 1
    out: List[int] = []
    while K <= K_max:
        out.append(K)
        K *= 2
    if out[-1] != K_max:
        # Ensure K_max included if it is not a power of two.
        out.append(K_max)
    return out


def compute_global_epsilon_embedding_x(
    matrices: Dict[str, np.ndarray],
    q_list: Sequence[int],
    *,
    factor: float = 0.25,
) -> float:
    """Compute a conservative global success threshold epsilon in embedding space.

    The plan requires a fixed global epsilon chosen below inter-branch separation.
    We compute the minimum predicted adjacent-branch separation in x at time q-1
    across the grid, convert it to the exact chordal distance on the unit circle,
    and multiply by a safety factor.

    Specifically:
        sep_T1(A,q) = d_{T1}(0, b_{q-1}/b_q)
        sep_E_x(A,q) = 2 sin(pi * sep_T1(A,q))
        epsilon = factor * min_{A,q} sep_E_x(A,q).

    This epsilon controls a sufficient condition: if a hypothesis has x_{q-1}
    within epsilon of truth in embedding space, it cannot be an adjacent branch
    in the digital regime.

    Returns:
        epsilon > 0.
    """
    if not (0.0 < factor < 1.0):
        raise ValueError("factor must be in (0,1).")

    seps: List[float] = []
    for _, A in matrices.items():
        check_sl2z_hyperbolic(A)
        for q in q_list:
            if q < 2:
                continue
            # Requires b_q != 0 (guaranteed by hyperbolicity).
            sep_t1 = interior_x_separation(A, q)
            sep_ex = embedding_circle_distance_from_torus_distance(sep_t1)
            if sep_ex <= 0.0:
                continue
            seps.append(sep_ex)

    if not seps:
        raise ValueError("No valid (A,q) pairs found to compute epsilon.")

    eps = float(factor * min(seps))
    if eps <= 0.0:
        raise RuntimeError("Computed non-positive epsilon; check separation computation.")
    return eps
