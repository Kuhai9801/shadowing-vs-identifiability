"""Direct minimax check aligned with Theorem III.5 (single-output estimators).

This script implements the "direct minimax experiment" recommended in the review notes:

  * Fix observations (x0_obs, xq_obs).
  * Construct two admissible strong-constraint truths with the SAME observed x-endpoints,
    but on ADJACENT branches in the unobserved coordinate.
  * Evaluate a single-output estimator phi(x0_obs, xq_obs) on both truths and verify
    that at least one incurs error >= delta/2, where

        delta := d_{T^1}(x_{q-1}^{(k)}, x_{q-1}^{(k+1)}) = d_{T^1}(0, b_{q-1}/b_q).

This matches the minimax proof pattern: the conclusion is a deterministic consequence
of the triangle inequality once two admissible truths with separation delta are
exhibited.

Outputs
-------
Writes a CSV (minimax_direct.csv) to --outdir with per-trial diagnostics.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on PYTHONPATH when the script is executed by path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PIPELINE_DIRS = ["1_datasets", "4_data_analysis", "5_models"]
for rel in _PIPELINE_DIRS:
    path = os.path.join(_ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pandas as pd

from data.generate import default_matrices
from scripts.reproducibility import seed_everything
from solvers.newton_shooting import NewtonConfig, solve_newton_shooting
from systems.torus_map import (
    E_x,
    bq,
    branch_solutions_y0,
    chordal_dist,
    dT1,
    embedding_circle_distance_from_torus_distance,
    interior_x_separation,
    iterate_map,
)


def _phi_estimate_x_qm1(
    *,
    phi: str,
    A: np.ndarray,
    q: int,
    x0_obs: float,
    xq_obs: float,
    seed: int,
    newton_K: int,
    newton_max_iter: int,
) -> float:
    """Return a single x_{q-1} estimate in T^1 (represented in [0,1))."""

    phi = str(phi)

    if phi == "x0":
        return float(x0_obs % 1.0)
    if phi == "xq":
        return float(xq_obs % 1.0)

    if phi == "branch0":
        y0s = branch_solutions_y0(A, q, float(x0_obs), float(xq_obs))
        y0 = float(y0s[0])
        traj = iterate_map(A, np.array([float(x0_obs) % 1.0, y0], dtype=float), q)
        return float(traj[q - 1, 0])

    if phi == "newton_top1":
        cfg = NewtonConfig(K_max=int(newton_K), max_iter=int(newton_max_iter), tol=1e-12)
        hyps = solve_newton_shooting(A=A, q=q, x0=float(x0_obs), xq=float(xq_obs), config=cfg, base_seed=int(seed))
        # Single-output induced by the solver: choose the top-1 by score.
        objs = np.asarray([float(h.objective) for h in hyps], dtype=float)
        idx = int(np.argmin(objs))
        return float(hyps[idx].traj[q - 1, 0])

    raise ValueError(
        f"Unknown phi={phi!r}. Supported: x0, xq, branch0, newton_top1."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct minimax check for Theorem III.5.")
    parser.add_argument("--outdir", type=str, default="6_results")
    parser.add_argument("--A_name", type=str, default="A1", choices=["A1", "A2", "A3"])
    parser.add_argument("--q", type=int, default=6, help="Gap length q (must be >=2).")
    parser.add_argument("--sigma", type=float, default=0.0, help="Noise bound sigma (used only for regime checks).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for generating random observations.")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of random observation pairs to test.")
    parser.add_argument(
        "--phi",
        type=str,
        default="branch0",
        choices=["x0", "xq", "branch0", "newton_top1"],
        help="Single-output estimator mapping (x0_obs, xq_obs) -> x_hat_{q-1}.",
    )
    parser.add_argument(
        "--branch_k",
        type=int,
        default=0,
        help="Which adjacent pair (k,k+1) to use when constructing the two truths.",
    )
    parser.add_argument("--x0_obs", type=float, default=None, help="Optional fixed x0_obs in [0,1).")
    parser.add_argument("--xq_obs", type=float, default=None, help="Optional fixed xq_obs in [0,1).")
    parser.add_argument("--newton_K", type=int, default=16, help="K for the newton_top1 phi.")
    parser.add_argument("--newton_max_iter", type=int, default=80, help="Max iters for the newton_top1 phi.")
    args = parser.parse_args()

    seed_everything(int(args.seed))

    os.makedirs(args.outdir, exist_ok=True)

    mats = default_matrices()
    A = mats[str(args.A_name)]
    q = int(args.q)
    if q < 2:
        raise ValueError("Require q>=2 so that x_{q-1} is a genuine interior time.")

    b_abs = abs(int(bq(A, q)))
    if b_abs < 2:
        raise ValueError(f"Need at least two branches (|b_q|>=2). Got |b_q|={b_abs}.")

    # Digital-regime check from the proofs.
    sigma = float(args.sigma)
    spacing_y0 = 1.0 / float(b_abs)
    digital_regime = bool(sigma < 0.5 * spacing_y0)

    # Precompute the predicted adjacent separation (should equal the truth construction).
    delta_pred_t1 = float(interior_x_separation(A, q))
    lb_t1 = 0.5 * delta_pred_t1
    lb_ex = float(embedding_circle_distance_from_torus_distance(lb_t1))

    rng = np.random.default_rng(int(args.seed))

    rows = []
    for t in range(int(args.n_trials)):
        # Observation pair.
        if args.x0_obs is None:
            x0_obs = float(rng.random())
        else:
            x0_obs = float(args.x0_obs) % 1.0
        if args.xq_obs is None:
            xq_obs = float(rng.random())
        else:
            xq_obs = float(args.xq_obs) % 1.0

        # Construct two admissible truths: same (x0,xq) but adjacent branch indices.
        y0s = branch_solutions_y0(A, q, x0_obs, xq_obs)
        if y0s.shape[0] != b_abs:
            raise RuntimeError("Unexpected number of branch solutions.")

        k = int(args.branch_k) % int(b_abs)
        k2 = (k + 1) % int(b_abs)
        y0_k = float(y0s[k])
        y0_k2 = float(y0s[k2])

        traj1 = iterate_map(A, np.array([x0_obs, y0_k], dtype=float), q)
        traj2 = iterate_map(A, np.array([x0_obs, y0_k2], dtype=float), q)
        x_qm1_1 = float(traj1[q - 1, 0])
        x_qm1_2 = float(traj2[q - 1, 0])

        # Separation between the two admissible truths at the imputation time.
        delta_t1 = float(dT1(x_qm1_1, x_qm1_2))
        delta_ex = float(chordal_dist(E_x(x_qm1_1), E_x(x_qm1_2)))

        # Single-output estimate.
        x_hat = _phi_estimate_x_qm1(
            phi=str(args.phi),
            A=A,
            q=q,
            x0_obs=x0_obs,
            xq_obs=xq_obs,
            seed=int(args.seed) + 1009 * (t + 1),
            newton_K=int(args.newton_K),
            newton_max_iter=int(args.newton_max_iter),
        )

        # Errors to each truth.
        err1_t1 = float(dT1(x_hat, x_qm1_1))
        err2_t1 = float(dT1(x_hat, x_qm1_2))
        max_err_t1 = float(max(err1_t1, err2_t1))

        err1_ex = float(chordal_dist(E_x(x_hat), E_x(x_qm1_1)))
        err2_ex = float(chordal_dist(E_x(x_hat), E_x(x_qm1_2)))
        max_err_ex = float(max(err1_ex, err2_ex))

        # Deterministic minimax conclusion: max error >= delta/2.
        bound_t1 = 0.5 * delta_t1
        bound_ex_from_t1 = float(embedding_circle_distance_from_torus_distance(bound_t1))

        pass_t1 = bool(max_err_t1 + 1e-15 >= bound_t1)
        # If the torus-distance bound holds, the embedding-x bound holds because d -> 2 sin(pi d)
        # is increasing on [0,1/2]. We check it as a separate numeric diagnostic.
        pass_ex = bool(max_err_ex + 1e-15 >= bound_ex_from_t1)

        rows.append(
            {
                "trial": int(t),
                "A_name": str(args.A_name),
                "q": int(q),
                "sigma": float(sigma),
                "digital_regime": int(digital_regime),
                "bq_abs": int(b_abs),
                "spacing_y0": float(spacing_y0),
                "x0_obs": float(x0_obs),
                "xq_obs": float(xq_obs),
                "branch_k": int(k),
                "branch_k2": int(k2),
                "phi": str(args.phi),
                "x_hat_qm1": float(x_hat),
                "x_qm1_1": float(x_qm1_1),
                "x_qm1_2": float(x_qm1_2),
                "delta_t1": float(delta_t1),
                "delta_pred_t1": float(delta_pred_t1),
                "delta_ex": float(delta_ex),
                "err1_t1": float(err1_t1),
                "err2_t1": float(err2_t1),
                "max_err_t1": float(max_err_t1),
                "bound_t1": float(bound_t1),
                "lb_t1": float(lb_t1),
                "err1_ex": float(err1_ex),
                "err2_ex": float(err2_ex),
                "max_err_ex": float(max_err_ex),
                "bound_ex_from_t1": float(bound_ex_from_t1),
                "lb_ex": float(lb_ex),
                "pass_t1": int(pass_t1),
                "pass_ex": int(pass_ex),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "minimax_direct.csv")
    df.to_csv(out_csv, index=False)

    # Console summary (Kaggle-friendly).
    n_pass_t1 = int(df["pass_t1"].sum())
    n_pass_ex = int(df["pass_ex"].sum())
    print("=" * 80)
    print("Direct minimax check complete")
    print(f"A={args.A_name} q={q} | |b_q|={b_abs} | sigma={sigma:g} | digital_regime={int(digital_regime)}")
    print(f"phi={args.phi} | trials={len(df)}")
    print(f"Triangle-inequality check (T^1): {n_pass_t1}/{len(df)} passed")
    print(f"Embedding-x lower-bound check:      {n_pass_ex}/{len(df)} passed")
    print(f"Wrote: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
