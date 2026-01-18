"""Dataset generation for the toral automorphism gap-filling experiments.

This module generates synthetic "truth" trajectories and endpoint observations consistent with
the paper's setup:

  * Dynamics: z_{n+1} = f_A(z_n) = A z_n (mod 1) on T^2.
  * Observations: only x at endpoints n=0 and n=q, with bounded noise
        x0_obs = x0 ⊕ eps0,  xq_obs = xq ⊕ epsq,  eps0,epsq ∈ [-sigma, sigma].

The ground truth is used ONLY for evaluation; solvers see only (x0_obs, xq_obs), A, q, sigma.

All routines validate the mathematical assumptions (A hyperbolic in SL(2,Z), sigma ∈ [0,1/2)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import zlib

from systems.torus_map import check_sl2z_hyperbolic, iterate_map


def default_matrices() -> Dict[str, np.ndarray]:
    """Return three canonical hyperbolic matrices in SL(2,Z) with distinct entropies.

    These matrices satisfy det(A)=1 and tr(A)>2.

    Returns:
        Dict mapping names to 2x2 integer numpy arrays.
    """
    mats = {
        "A1": np.array([[2, 1], [1, 1]], dtype=int),
        "A2": np.array([[3, 2], [1, 1]], dtype=int),
        "A3": np.array([[4, 3], [1, 1]], dtype=int),
    }
    for A in mats.values():
        check_sl2z_hyperbolic(A)
    return mats


def sample_uniform_t2(rng: np.random.Generator) -> np.ndarray:
    """Sample z ∼ Unif([0,1)^2)."""
    return rng.random(2)


def add_bounded_noise_t1(x: float, sigma: float, rng: np.random.Generator) -> float:
    """Add bounded noise eps ∈ [-sigma, sigma] on T^1 (mod 1)."""
    if not (0.0 <= sigma < 0.5):
        raise ValueError("sigma must be in [0,1/2).")
    eps = rng.uniform(-sigma, sigma)
    return float((float(x) + float(eps)) % 1.0)


@dataclass(frozen=True)
class Instance:
    """One smoothing instance: truth trajectory, endpoint observations, metadata."""

    A_name: str
    A: np.ndarray
    q: int
    sigma: float
    seed: int

    z0_true: np.ndarray  # (2,)
    traj_true: np.ndarray  # (q+1,2) in [0,1)
    x0_obs: float
    xq_obs: float


def generate_instance(
    *,
    A_name: str,
    A: np.ndarray,
    q: int,
    sigma: float,
    seed: int,
) -> Instance:
    """Generate a single instance with its own RNG seed."""
    check_sl2z_hyperbolic(A)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if not (0.0 <= sigma < 0.5):
        raise ValueError("sigma must be in [0,1/2).")

    rng = np.random.default_rng(int(seed))
    z0_true = sample_uniform_t2(rng)
    traj_true = iterate_map(A, z0_true, int(q))

    x0_true = float(traj_true[0, 0])
    xq_true = float(traj_true[int(q), 0])

    x0_obs = add_bounded_noise_t1(x0_true, float(sigma), rng)
    xq_obs = add_bounded_noise_t1(xq_true, float(sigma), rng)

    return Instance(
        A_name=str(A_name),
        A=np.asarray(A, dtype=int),
        q=int(q),
        sigma=float(sigma),
        seed=int(seed),
        z0_true=z0_true,
        traj_true=traj_true,
        x0_obs=float(x0_obs),
        xq_obs=float(xq_obs),
    )


def generate_grid(
    *,
    matrices: Dict[str, np.ndarray],
    q_list: Sequence[int],
    sigma: float,
    n_instances: int,
    seed0: int,
) -> List[Instance]:
    """Generate a grid of instances across (A,q) pairs.

    Seeds are generated deterministically from (seed0, A_name, q, instance_index).

    IMPORTANT determinism note:
        Do NOT use Python's built-in hash() for seeding. hash() is intentionally salted per
        interpreter process unless PYTHONHASHSEED is fixed. Instead we use a stable 32-bit
        CRC over the matrix name.

    Args:
        matrices: Dict mapping matrix names to matrices.
        q_list: Gap lengths.
        sigma: Noise bound.
        n_instances: Instances per (A,q).
        seed0: Base seed.

    Returns:
        List[Instance].
    """
    if n_instances < 1:
        raise ValueError("n_instances must be >= 1.")

    def stable_name_hash_u32(name: str) -> int:
        return int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)

    instances: List[Instance] = []
    for A_name, A in matrices.items():
        name_hash = stable_name_hash_u32(str(A_name))
        for q in q_list:
            for i in range(n_instances):
                # Deterministic seed schedule.
                seed = int(
                    (int(seed0) * 1000003)
                    + (name_hash * 9176)
                    + (int(q) * 1009)
                    + int(i)
                )
                instances.append(
                    generate_instance(
                        A_name=str(A_name),
                        A=A,
                        q=int(q),
                        sigma=float(sigma),
                        seed=seed,
                    )
                )
    return instances
