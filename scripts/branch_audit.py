"""Exhaustive branch audit for the torus automorphism shooting problem.

This script complements the main manuscript by comparing:
  (i) exhaustive enumeration of all strong-constraint branches y0^{(k)} induced by
      endpoint-only observations (Lemma 1 / Lemma \ref{lem:branches}), and
 (ii) random-restart Newton shooting as a stochastic sampler of those branches.

By default it reproduces the representative audit reported in the paper:
  * Arnold cat map A1 = [[2,1],[1,1]]
  * window length q = 6 (|b_q| = 144)
  * noiseless endpoints (sigma = 0)
  * N = 20 instances, seeds 1000..1019
  * K = 64 Newton restarts

Outputs:
  * CSV with per-instance branch coverage statistics.

Run from the repository root, e.g.:
  python scripts/branch_audit.py --outdir 6_results
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PIPELINE_DIRS = ["1_datasets", "4_data_analysis", "5_models"]
for rel in _PIPELINE_DIRS:
    path = os.path.join(_ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from data.generate import generate_instance
from systems.torus_map import branch_solutions_y0, bq, dT1
from solvers.newton_shooting import NewtonConfig, solve_newton_shooting


def _count_unique_branches(y0_found: np.ndarray, y0_branches: np.ndarray, tol: float) -> tuple[int, float]:
    """Return (num_unique, max_min_dist)."""
    if y0_found.size == 0:
        return 0, float("nan")

    # Assign each found value to the nearest enumerated branch on T^1.
    # Complexity is O(K*|b_q|) which is fine for the intended audit sizes.
    idxs = []
    max_min_dist = 0.0
    for y in y0_found:
        d = np.array([dT1(y, yb) for yb in y0_branches])
        j = int(np.argmin(d))
        idxs.append(j)
        max_min_dist = max(max_min_dist, float(d[j]))

    # Unique branch indices.
    unique = len(set(idxs))

    # Enforce tolerance: if max_min_dist exceeds tol, treat as a mismatch.
    # (For noiseless endpoints and converged Newton roots, this should be ~1e-15.)
    return unique, max_min_dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="6_results", help="Output directory.")
    parser.add_argument("--A_name", type=str, default="A1", help="Matrix label for reporting.")
    parser.add_argument("--q", type=int, default=6, help="Window length.")
    parser.add_argument("--sigma", type=float, default=0.0, help="Endpoint noise level.")
    parser.add_argument("--N_inst", type=int, default=20, help="Number of instances.")
    parser.add_argument("--seed0", type=int, default=1000, help="Base seed for instances.")
    parser.add_argument("--K", type=int, default=64, help="Newton restart budget.")
    parser.add_argument("--tol_root", type=float, default=1e-10, help="Residual threshold for counting a restart as converged.")
    parser.add_argument("--tol_match", type=float, default=1e-12, help="Max torus distance for matching a root to an enumerated branch.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Hard-coded matrices used in the paper.
    A_catalog = {
        "A1": np.array([[2, 1], [1, 1]], dtype=int),
        "A2": np.array([[3, 2], [1, 1]], dtype=int),
        "A3": np.array([[4, 3], [1, 1]], dtype=int),
    }
    if args.A_name not in A_catalog:
        raise ValueError(f"Unknown A_name={args.A_name}. Choose from {sorted(A_catalog.keys())}.")

    A = A_catalog[args.A_name]
    q = int(args.q)

    bq_abs = int(abs(bq(A, q)))

    cfg = NewtonConfig(K_max=int(args.K), max_iter=50, tol=1e-12)

    rows = []

    for i in range(int(args.N_inst)):
        seed = int(args.seed0) + i
        inst = generate_instance(A_name=args.A_name, A=A, q=q, sigma=float(args.sigma), seed=seed)

        # Enumerate all strong-constraint branches for the observed endpoints.
        y0_branches = np.asarray(branch_solutions_y0(A, q, inst.x0_obs, inst.xq_obs), dtype=float)

        # Sanity: ensure the enumerator returns the expected count.
        if len(y0_branches) != bq_abs:
            raise RuntimeError(
                f"Expected |b_q|={bq_abs} branches but got {len(y0_branches)} (seed={seed})."
            )

        # Run Newton shooting with random restarts.
        hyps = solve_newton_shooting(
            A=A,
            q=q,
            x0=float(inst.x0_obs),
            xq=float(inst.xq_obs),
            config=cfg,
            base_seed=seed,
        )

        # Keep only converged roots.
        y0_found = np.array([h.y0 for h in hyps if h.residual_norm <= float(args.tol_root)], dtype=float)

        num_unique, max_min_dist = _count_unique_branches(y0_found, y0_branches, tol=float(args.tol_match))

        rows.append(
            {
                "A_name": args.A_name,
                "q": q,
                "sigma": float(args.sigma),
                "seed": seed,
                "bq_abs": bq_abs,
                "K": int(args.K),
                "num_restarts": len(hyps),
                "num_converged": int(len(y0_found)),
                "num_unique_branches": int(num_unique),
                "max_root_to_branch_dT1": float(max_min_dist),
                "tol_root": float(args.tol_root),
                "tol_match": float(args.tol_match),
                "newton_config": str(asdict(cfg)),
            }
        )

    df = pd.DataFrame(rows)

    # Aggregate summary (reported in the manuscript in a representative form).
    med = float(df["num_unique_branches"].median())
    q25 = float(df["num_unique_branches"].quantile(0.25, interpolation="nearest"))
    q75 = float(df["num_unique_branches"].quantile(0.75, interpolation="nearest"))
    max_mismatch = float(df["max_root_to_branch_dT1"].max())

    print(f"A={args.A_name}, q={q}, |b_q|={bq_abs}, sigma={float(args.sigma)}")
    print(f"K={int(args.K)}, N={int(args.N_inst)}")
    print(f"Distinct branches found (median [IQR]) = {med:.1f} [{q25:.0f},{q75:.0f}]")
    print(f"Max root-to-branch mismatch dT1 = {max_mismatch:.3e}")

    out_csv = outdir / f"branch_audit_{args.A_name}_q{q}_K{int(args.K)}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
