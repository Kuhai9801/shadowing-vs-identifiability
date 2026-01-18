"""Newton / Gauss--Newton strong-constraint baseline for endpoint conditioning.

Purpose
-------
This solver is an optimization baseline to neutralize the critique
"PINNs fail because optimization is finicky". In the noiseless endpoint case (sigma=0),
conditioning on fixed endpoints reduces to the scalar congruence

    x_q ≡ a_q x_0 + b_q y_0 (mod 1).

We define a smooth residual in embedding space (no wrap discontinuities):

    R(y0) = (cos(2π(a_q x0 + b_q y0 - x_q)) - 1,
             sin(2π(a_q x0 + b_q y0 - x_q))).

Roots correspond exactly to admissible branches.

We implement damped Gauss--Newton on ||R||^2 with random restarts.

Important note:
--------------
This procedure operates only on the scalar unknown y0; once a root is found, the full
orbit segment is obtained by exact forward iteration of f_A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
import time
import numpy as np

from systems.torus_map import Aq_entries, bq, check_sl2z_hyperbolic, frac, iterate_map, dT1


@dataclass
class NewtonConfig:
    """Configuration for damped Gauss--Newton shooting."""

    K_max: int = 64
    max_iter: int = 50
    tol: float = 1e-12
    min_step_scale: float = 1e-6
    backtrack_factor: float = 0.5


@dataclass
class NewtonHypothesis:
    restart_id: int
    seed: int
    converged: bool
    iters: int

    y0: float  # in [0,1)

    residual_norm: float
    objective: float

    # Exact orbit in [0,1)^2.
    traj: np.ndarray  # shape (q+1,2)

    seconds: float


def _residual(a_q: int, b_q: int, x0: float, xq: float, y0: float) -> Tuple[float, float]:
    """Compute R(y0) as defined in the plan."""
    # Reduce the phase mod 1 before applying trigonometric functions.
    # This avoids catastrophic loss of precision in sin/cos for large |b_q|.
    phase = (a_q * x0 + b_q * y0 - xq) % 1.0
    phi = 2.0 * math.pi * phase
    return (math.cos(phi) - 1.0, math.sin(phi))


def solve_newton_shooting(
    *,
    A: np.ndarray,
    q: int,
    x0: float,
    xq: float,
    config: NewtonConfig,
    base_seed: int,
) -> List[NewtonHypothesis]:
    """Run K_max random-restart damped Gauss--Newton solves for y0.

    Args:
        A: 2x2 integer hyperbolic SL(2,Z) matrix.
        q: horizon length (>=1).
        x0: fixed endpoint x_0 in [0,1).
        xq: fixed endpoint x_q in [0,1).
        config: configuration.
        base_seed: seed for RNG.

    Returns:
        List[NewtonHypothesis] of length K_max.
    """
    check_sl2z_hyperbolic(A)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if config.K_max < 1:
        raise ValueError("K_max must be >= 1.")

    a_q, b_q, _, _ = Aq_entries(A, q)
    if b_q == 0:
        # By Proposition 1(a) this cannot occur for hyperbolic A in SL(2,Z).
        raise RuntimeError("b_q=0 contradicts hyperbolicity; abort.")

    rng = np.random.default_rng(int(base_seed))

    hyps: List[NewtonHypothesis] = []

    for j in range(config.K_max):
        seed_j = int(base_seed + 1009 * (j + 1))
        rng_j = np.random.default_rng(seed_j)
        y = float(rng_j.random())  # initial guess in [0,1)

        t0 = time.time()

        converged = False
        it_used = 0

        # Objective f(y) = ||R(y)||^2.
        r0, r1 = _residual(a_q, b_q, x0, xq, y)
        f = r0 * r0 + r1 * r1

        for it in range(1, config.max_iter + 1):
            it_used = it
            res_norm = math.sqrt(f)
            if res_norm <= config.tol:
                converged = True
                break

            # Gauss--Newton step for 1D variable with 2D residual.
            # J(y) = dR/dy = 2π b_q * (-sinφ, cosφ).
            # J^T R = 2π b_q * sinφ.
            # (J^T J) = (2π b_q)^2.
            # Δ = -(J^T R)/(J^T J) = - sinφ / (2π b_q).
            phase = (a_q * x0 + b_q * y - xq) % 1.0
            phi = 2.0 * math.pi * phase
            step = -math.sin(phi) / (2.0 * math.pi * float(b_q))

            # Backtracking line search: accept if objective decreases.
            alpha = 1.0
            accepted = False
            while alpha >= config.min_step_scale:
                y_new = (y + alpha * step) % 1.0
                nr0, nr1 = _residual(a_q, b_q, x0, xq, y_new)
                f_new = nr0 * nr0 + nr1 * nr1
                if f_new < f:
                    y = y_new
                    r0, r1 = nr0, nr1
                    f = f_new
                    accepted = True
                    break
                alpha *= config.backtrack_factor

            if not accepted:
                # If no decrease, terminate; this is an optimization failure.
                break

        seconds = float(time.time() - t0)

        # Construct exact orbit segment by forward iteration.
        z0 = np.array([float(x0) % 1.0, float(y) % 1.0], dtype=float)
        traj = iterate_map(A, z0, q)

        # Verify endpoint congruence numerically in T^1.
        # This is not enforced as a hard error because floating arithmetic is used.
        endpoint_err = float(dT1(traj[q, 0], float(xq) % 1.0))

        res_norm = float(math.sqrt(f))
        hyps.append(
            NewtonHypothesis(
                restart_id=j,
                seed=seed_j,
                converged=converged,
                iters=it_used,
                y0=float(y),
                residual_norm=res_norm,
                objective=float(f),
                traj=traj,
                seconds=seconds,
            )
        )

    return hyps
