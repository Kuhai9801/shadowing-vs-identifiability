"""Core torus-map utilities for the Chaos "shadowing vs inference" experiments.

This module implements:
  * Exact arithmetic for integer powers A^q (A \in SL(2,\Z)).
  * The torus maps f_A(z) = A z (mod 1) on T^2.
  * Quotient metrics d_{T^1}, d_{T^2} and chordal embedding metrics.
  * Stable/unstable splitting data and the explicit finite-horizon shadowing constant C(A).

Design requirements (methodological):
  * Training losses are computed strictly in embedding space (R^4) using chordal distances.
  * Modular projection (frac) is NEVER used inside a loss. It is used only for evaluation,
    reporting, and symbolic labeling.

All public functions enforce the mathematical assumptions needed by the proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, overload

import math
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# Small numerical constant reused in many hot paths.
TWO_PI = 2.0 * math.pi


def _is_torch(x: object) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def check_sl2z_hyperbolic(A: np.ndarray) -> None:
    """Validate that A is a 2x2 integer hyperbolic matrix in SL(2,Z).

    Requirements:
      * A is shape (2,2).
      * A has integer entries.
      * det(A) == 1.
      * |tr(A)| > 2  (hyperbolicity).

    Raises:
        ValueError: if any condition fails.
    """
    A = np.asarray(A)
    if A.shape != (2, 2):
        raise ValueError(f"A must be 2x2; got shape {A.shape}.")

    # Integer check (robust to dtype).
    if not np.all(np.equal(A, np.round(A))):
        raise ValueError("A must have integer entries.")

    a00 = int(A[0, 0])
    a01 = int(A[0, 1])
    a10 = int(A[1, 0])
    a11 = int(A[1, 1])
    det = a00 * a11 - a01 * a10
    if det != 1:
        raise ValueError(f"A must satisfy det(A)=1 (SL(2,Z)); got det(A)={det}.")

    tr = int(A[0, 0] + A[1, 1])
    if abs(tr) <= 2:
        raise ValueError(f"A must be hyperbolic with |tr(A)|>2; got tr(A)={tr}.")

    disc = tr * tr - 4
    if disc <= 0:
        raise ValueError(
            f"Hyperbolicity requires discriminant tr(A)^2-4>0; got {disc}."
        )


def matrix_power_int(A: np.ndarray, q: int) -> np.ndarray:
    """Compute A^q exactly as integers using exponentiation by squaring.

    The proof-scale experiments use q \le 22, but this routine is exact and does not
    rely on floating point arithmetic.

    Args:
        A: 2x2 integer matrix.
        q: integer power, q >= 0.

    Returns:
        A^q as a 2x2 integer numpy array (dtype=object to avoid overflow).

    Raises:
        ValueError: if q < 0 or A is not 2x2.
    """
    A = np.asarray(A)
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2.")
    if q < 0:
        raise ValueError("q must be nonnegative.")

    # Use Python ints (dtype=object) for correctness independent of magnitude.
    base = A.astype(object)
    result = np.eye(2, dtype=object)
    exp = int(q)

    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1

    return result


def eigenvalues_sl2_hyperbolic(A: np.ndarray) -> Tuple[float, float]:
    """Return (lambda_u, lambda_s) for a hyperbolic A \in SL(2,Z).

    Uses the closed-form eigenvalues for 2x2 matrices with det=1:
        lambda = (tr \pm sqrt(tr^2 - 4))/2.

    The returned ordering satisfies |lambda_u| > 1 > |lambda_s|.

    Raises:
        ValueError: if A is not hyperbolic SL(2,Z).
    """
    check_sl2z_hyperbolic(A)
    tr = float(A[0, 0] + A[1, 1])
    disc = tr * tr - 4.0
    sqrt_disc = math.sqrt(disc)
    lam1 = 0.5 * (tr + sqrt_disc)
    lam2 = 0.5 * (tr - sqrt_disc)

    if abs(lam1) >= abs(lam2):
        lam_u, lam_s = lam1, lam2
    else:
        lam_u, lam_s = lam2, lam1

    if not (abs(lam_u) > 1.0 and abs(lam_s) < 1.0):
        raise ValueError(
            "Internal eigenvalue classification failed; verify hyperbolicity assumptions."
        )

    return float(lam_u), float(lam_s)


def hks_entropy(A: np.ndarray) -> float:
    """Kolmogorov–Sinai entropy per step for a hyperbolic toral automorphism.

    For a linear hyperbolic toral automorphism, h_KS = log |lambda_u|.
    """
    lam_u, _ = eigenvalues_sl2_hyperbolic(A)
    return float(math.log(abs(lam_u)))


@dataclass(frozen=True)
class SplittingData:
    """Stable/unstable splitting data and the explicit finite-horizon shadowing constant."""

    lambda_u: float
    lambda_s: float
    u_dir: np.ndarray  # unit vector spanning E^u
    s_dir: np.ndarray  # unit vector spanning E^s
    P_u: np.ndarray  # 2x2 projection matrix onto E^u along E^s
    P_s: np.ndarray  # 2x2 projection matrix onto E^s along E^u
    norm_Pu: float
    norm_Ps: float
    angle: float  # angle between the lines E^u and E^s in radians (in (0,pi))
    C: float  # explicit shadowing constant C(A)


# -----------------------------------------------------------------------------
# Performance cache
# -----------------------------------------------------------------------------
#
# Evaluation computes eps_cert = C(A) * δ_T for every hypothesis. Without caching,
# splitting_data(A) would repeatedly compute eigenvectors, projections, and norms.
# In the experiment plan, A ranges over a small fixed set of integer matrices, so
# caching is safe and significantly reduces wall time.
_SPLITTING_CACHE: dict[tuple[int, int, int, int, float], SplittingData] = {}


def splitting_data(A: np.ndarray, *, angle_tol: float = 1e-12) -> SplittingData:
    """Compute stable/unstable splitting projections and constant C(A).

    This routine constructs the Euclidean-operator-norm projections P_u, P_s associated to
    the direct sum decomposition R^2 = E^u \oplus E^s. It then returns the explicit finite-horizon
    Lipschitz shadowing constant

        C(A) = ||P_s||/(1 - |lambda_s|) + ||P_u||/(|lambda_u| - 1).

    The formula is valid for hyperbolic linear maps; only magnitudes of eigenvalues enter the
    contraction/expansion bounds.

    Raises:
        ValueError: if A violates assumptions or if the splitting angle is degenerate.
    """
    # Cache by integer matrix entries (A is always tiny and reused heavily).
    A = np.asarray(A)
    if A.shape != (2, 2):
        raise ValueError(f"A must be 2x2; got shape {A.shape}.")
    key = (
        int(A[0, 0]),
        int(A[0, 1]),
        int(A[1, 0]),
        int(A[1, 1]),
        float(angle_tol),
    )
    cached = _SPLITTING_CACHE.get(key)
    if cached is not None:
        return cached

    lam_u, lam_s = eigenvalues_sl2_hyperbolic(A)

    # Numerically compute eigenvectors (stable/unstable directions).
    w, V = np.linalg.eig(np.asarray(A, dtype=float))
    # Match eigenvectors to the closed-form eigenvalues by absolute value.
    idx_u = int(np.argmax(np.abs(w)))
    idx_s = 1 - idx_u

    u = np.real(V[:, idx_u])
    s = np.real(V[:, idx_s])

    # Normalize to unit vectors.
    u_norm = np.linalg.norm(u)
    s_norm = np.linalg.norm(s)
    if u_norm == 0.0 or s_norm == 0.0:
        raise ValueError("Eigenvector computation failed (zero norm).")
    u = u / u_norm
    s = s / s_norm

    # Ensure the directions define a basis (non-colinear).
    # Angle between lines is arccos(|dot|).
    dot = float(np.clip(np.dot(u, s), -1.0, 1.0))
    angle = float(math.acos(abs(dot)))
    if angle <= angle_tol or abs(math.sin(angle)) <= angle_tol:
        raise ValueError(
            "Stable/unstable splitting angle too small; projection norms blow up."
        )

    # Construct projections via basis transform.
    B = np.column_stack([u, s])  # columns are basis vectors
    if abs(np.linalg.det(B)) <= angle_tol:
        raise ValueError("Stable/unstable eigenvectors are nearly linearly dependent.")
    B_inv = np.linalg.inv(B)

    # For v, coordinates c = B_inv v, and projection onto u is u * c0.
    P_u = np.outer(u, B_inv[0, :])
    P_s = np.outer(s, B_inv[1, :])

    # Projection-norm identity (2D Euclidean geometry): for unit spanning vectors u,s with
    # angle θ between the corresponding lines, the operator norms satisfy
    #     ||P_u|| = ||P_s|| = 1/|sin θ|.
    # We compute this explicitly for numerical stability and additionally sanity-check the
    # matrix-based computation.
    norm_formula = float(1.0 / abs(math.sin(angle)))
    norm_Pu_mat = float(np.linalg.norm(P_u, 2))
    norm_Ps_mat = float(np.linalg.norm(P_s, 2))
    if not (abs(norm_Pu_mat - norm_formula) <= 1e-6 * norm_formula and abs(norm_Ps_mat - norm_formula) <= 1e-6 * norm_formula):
        raise ValueError(
            "Projection norm sanity-check failed: expected ||P_u||=||P_s||=1/|sin θ|. "
            f"Got ||P_u||={norm_Pu_mat:.6e}, ||P_s||={norm_Ps_mat:.6e}, 1/|sin θ|={norm_formula:.6e}."
        )
    norm_Pu = norm_formula
    norm_Ps = norm_formula

    mu_u = abs(lam_u)
    # For SL(2,Z) matrices with det(A)=1, the stable magnitude is exactly 1/mu_u.
    # We compute it this way to avoid the accumulation of floating-point error.
    mu_s = 1.0 / mu_u
    if not (mu_u > 1.0 and mu_s < 1.0):
        raise ValueError("Eigenvalue magnitudes not consistent with hyperbolicity.")

    C = norm_Ps / (1.0 - mu_s) + norm_Pu / (mu_u - 1.0)

    out = SplittingData(
        lambda_u=float(lam_u),
        lambda_s=float(lam_s),
        u_dir=u,
        s_dir=s,
        P_u=P_u,
        P_s=P_s,
        norm_Pu=norm_Pu,
        norm_Ps=norm_Ps,
        angle=angle,
        C=float(C),
    )

    _SPLITTING_CACHE[key] = out
    return out


def frac(x: ArrayLike) -> ArrayLike:
    """Componentwise fractional part mapping to [0,1).

    This is used only for evaluation and symbolic labeling.
    """
    if _is_torch(x):
        # torch.frac returns fractional part with sign; use floor-based definition.
        return x - torch.floor(x)  # type: ignore[attr-defined]
    x_np = np.asarray(x)
    return x_np - np.floor(x_np)


def dT1(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Distance on T^1 = R/Z: d(x,y) = min_{m \in Z} |x-y+m| in [0,1/2]."""
    if _is_torch(x) or _is_torch(y):
        xt = x if _is_torch(x) else torch.as_tensor(x)  # type: ignore[union-attr]
        yt = y if _is_torch(y) else torch.as_tensor(y)  # type: ignore[union-attr]
        delta = xt - yt
        delta = (delta + 0.5) % 1.0 - 0.5
        return torch.abs(delta)
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    delta = x_np - y_np
    delta = (delta + 0.5) % 1.0 - 0.5
    return np.abs(delta)


def dT2(z: ArrayLike, w: ArrayLike) -> ArrayLike:
    """Distance on T^2 induced by Euclidean norm: d(z,w)=||((z-w)+1/2 mod 1)-1/2||_2."""
    if _is_torch(z) or _is_torch(w):
        zt = z if _is_torch(z) else torch.as_tensor(z)  # type: ignore[union-attr]
        wt = w if _is_torch(w) else torch.as_tensor(w)  # type: ignore[union-attr]
        delta = zt - wt
        delta = (delta + 0.5) % 1.0 - 0.5
        return torch.linalg.norm(delta, dim=-1)
    z_np = np.asarray(z, dtype=float)
    w_np = np.asarray(w, dtype=float)
    delta = z_np - w_np
    delta = (delta + 0.5) % 1.0 - 0.5
    return np.linalg.norm(delta, axis=-1)


def E_x(x: ArrayLike) -> ArrayLike:
    """Embedding E_x: T^1 -> R^2, x -> (cos 2πx, sin 2πx)."""
    if _is_torch(x):
        return torch.stack([torch.cos(TWO_PI * x), torch.sin(TWO_PI * x)], dim=-1)
    x_np = np.asarray(x, dtype=float)
    return np.stack([np.cos(TWO_PI * x_np), np.sin(TWO_PI * x_np)], axis=-1)


def E(z: ArrayLike) -> ArrayLike:
    """Embedding E: T^2 -> R^4 via two independent circle embeddings."""
    if _is_torch(z):
        zt = z
        # Compute cos/sin for both coordinates in one call each (fewer GPU kernel launches).
        ang = TWO_PI * zt[..., :2]
        c = torch.cos(ang)
        s = torch.sin(ang)
        return torch.stack([c[..., 0], s[..., 0], c[..., 1], s[..., 1]], dim=-1)
    z_np = np.asarray(z, dtype=float)
    x = z_np[..., 0]
    y = z_np[..., 1]
    return np.stack(
        [np.cos(TWO_PI * x), np.sin(TWO_PI * x), np.cos(TWO_PI * y), np.sin(TWO_PI * y)],
        axis=-1,
    )


def chordal_dist(a: ArrayLike, b: ArrayLike, *, squared: bool = False) -> ArrayLike:
    """Euclidean chordal distance in embedding space."""
    if _is_torch(a) or _is_torch(b):
        at = a if _is_torch(a) else torch.as_tensor(a)  # type: ignore[union-attr]
        bt = b if _is_torch(b) else torch.as_tensor(b)  # type: ignore[union-attr]
        diff = at - bt
        if squared:
            return torch.sum(diff * diff, dim=-1)
        return torch.linalg.norm(diff, dim=-1)
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    diff = a_np - b_np
    if squared:
        return np.sum(diff * diff, axis=-1)
    return np.linalg.norm(diff, axis=-1)


def iterate_map(A: np.ndarray, z0: np.ndarray, q: int) -> np.ndarray:
    """Simulate z_{n+1} = A z_n (mod 1) on T^2, with z0 in [0,1)^2.

    Args:
        A: 2x2 integer hyperbolic SL(2,Z) matrix.
        z0: shape (2,), initial state on the torus represented in [0,1)^2.
        q: integer horizon length >= 0.

    Returns:
        Array of shape (q+1,2) with entries in [0,1).
    """
    check_sl2z_hyperbolic(A)
    if q < 0:
        raise ValueError("q must be nonnegative.")
    z0 = np.asarray(z0, dtype=float)
    if z0.shape != (2,):
        raise ValueError("z0 must have shape (2,).")

    z = np.empty((q + 1, 2), dtype=float)
    z[0] = frac(z0)
    Af = np.asarray(A, dtype=float)
    for n in range(q):
        z[n + 1] = frac(Af @ z[n])
    return z


def Aq_entries(A: np.ndarray, q: int) -> Tuple[int, int, int, int]:
    """Return integer entries (a_q,b_q,c_q,d_q) of A^q."""
    if q < 0:
        raise ValueError("q must be nonnegative.")
    M = matrix_power_int(A, q)
    a_q = int(M[0, 0])
    b_q = int(M[0, 1])
    c_q = int(M[1, 0])
    d_q = int(M[1, 1])
    return a_q, b_q, c_q, d_q


def bq(A: np.ndarray, q: int) -> int:
    """Return b_q, the (1,2) entry of A^q."""
    _, b, _, _ = Aq_entries(A, q)
    if q >= 1 and b == 0:
        # This should never occur for hyperbolic A in SL(2,Z) by Proposition 1.
        raise RuntimeError("Computed b_q=0 for q>=1; verify A and arithmetic.")
    return b


def branch_solutions_y0(A: np.ndarray, q: int, x0: float, xq: float) -> np.ndarray:
    """Return the |b_q| branch solutions y0^{(k)} for fixed (x0,xq) in T^1.

    Implements Lemma 1 (Branching in the unobserved coordinate):
        y0^{(k)} = (xq - a_q x0 + k) / b_q (mod 1),  k=0,...,|b_q|-1.

    Args:
        A: 2x2 integer hyperbolic SL(2,Z) matrix.
        q: integer gap length, q >= 1.
        x0: initial observed x-coordinate in [0,1).
        xq: terminal observed x-coordinate in [0,1).

    Returns:
        Array y0s of shape (|b_q|,) in [0,1).
    """
    if q < 1:
        raise ValueError("q must be >= 1 for branching.")
    check_sl2z_hyperbolic(A)
    a_q, b_q, _, _ = Aq_entries(A, q)
    if b_q == 0:
        raise RuntimeError("b_q=0 contradicts hyperbolicity; abort.")

    b_abs = abs(b_q)
    x0 = float(x0 % 1.0)
    xq = float(xq % 1.0)

    # Choose real lifts as x0,xq in [0,1).
    base = xq - a_q * x0
    ks = np.arange(b_abs, dtype=float)
    y = (base + ks) / float(b_q)
    return frac(y)


def interior_x_separation(A: np.ndarray, q: int) -> float:
    """Return the predicted T^1 separation between adjacent branches at time q-1.

    Implements Lemma 2: separation = d_{T^1}(0, b_{q-1}/b_q).

    Args:
        A: hyperbolic SL(2,Z) matrix.
        q: integer horizon length, q >= 2.

    Returns:
        Separation in [0, 1/2].
    """
    if q < 2:
        raise ValueError("q must be >= 2 for interior separation at time q-1.")
    b_q = bq(A, q)
    b_qm1 = bq(A, q - 1)
    ratio = float(b_qm1) / float(b_q)
    # d_T1(0,ratio) is min_{m in Z} |ratio + m|.
    sep = float(dT1(ratio, 0.0))
    return sep


def embedding_circle_distance_from_torus_distance(dt1: float) -> float:
    """Exact chordal distance on the unit circle as a function of torus distance on T^1.

    For x,x' with d_{T^1}(x,x') = d in [0,1/2], the chordal distance satisfies
        ||E_x(x) - E_x(x')||_2 = 2 |sin(pi d)|.

    This is used to set an \varepsilon threshold in embedding space consistent with
    an inter-branch separation threshold in T^1.
    """
    if not (0.0 <= dt1 <= 0.5):
        raise ValueError("dt1 must lie in [0,1/2].")
    return float(2.0 * math.sin(math.pi * dt1))