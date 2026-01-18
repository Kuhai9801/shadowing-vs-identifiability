"""Empirical x-imputation error floor vs the minimax lower bound (Theorem III.5).

This plot complements Money Plot 2 by focusing on the *observed coordinate* at the
historical time q-1. The theory provides an information-theoretic lower bound for
any estimator (in the noiseless endpoint setting) proportional to the adjacent-branch
separation:

    LB(q) = 0.5 * d_{T^1}(0, b_{q-1}/b_q).

In the codebase we report this bound in x-embedding chordal units:

    LB_E(q) = ||E_x(0) - E_x(LB(q))||_2 = 2 sin(pi * LB(q)).

This script reads a *_success.csv file (Block A or Block B) and plots an empirical
distribution of either:
  * single-output (top-1-by-score) x-error, or
  * best-of-K *coverage* x-error (oracle min-error within the top-K score-ranked set),
alongside LB_E(q).

Usage:
  # Single-output (matches the minimax theorem object):
  python 4_data_analysis/figures/plot_minimax_bound.py --success_csv 6_results/blockA_success.csv --solver newton --K 64 --metric single_error

  # Coverage (set-valued top-K mitigation beyond the theorem's scope):
  python 4_data_analysis/figures/plot_minimax_bound.py --success_csv 6_results/blockA_success.csv --solver newton --K 64 --metric best_error

Outputs:
  7_reports/figures/minimax_bound_plot.png
  7_reports/figures/minimax_bound_plot.pdf
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on PYTHONPATH when the script is executed by path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot empirical best-of-K x-error vs the minimax lower bound.")
    parser.add_argument("--success_csv", type=str, default="6_results/blockA_success.csv")
    parser.add_argument("--outdir", type=str, default="7_reports/figures")
    parser.add_argument("--solver", type=str, default=None, help="Which solver to plot (default: all).")
    parser.add_argument("--K", type=int, default=None, help="Which K to plot (default: max K in CSV).")
    parser.add_argument(
        "--metric",
        type=str,
        default="single_error",
        choices=["single_error", "single_error_full", "best_error", "best_error_full"],
        help=(
            "Which error column to plot. single_* corresponds to the solver-induced single-output estimator "
            "(top-1 by score). best_* corresponds to the best-of-K *coverage* oracle within the top-K score-ranked set. "
            "*_error is x-only; *_error_full is full-state embedding error."
        ),
    )
    parser.add_argument("--quantiles", type=str, default="0.25,0.5,0.75", help="Comma-separated quantiles.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.success_csv)
    if df.empty:
        raise RuntimeError("Success CSV is empty.")

    K_plot = int(args.K) if args.K is not None else int(df["K"].max())
    sub = df[df["K"] == K_plot].copy()
    if sub.empty:
        raise ValueError(f"No rows found for K={K_plot}.")

    if args.solver is not None:
        sub = sub[sub["solver"] == str(args.solver)].copy()
        if sub.empty:
            raise ValueError(f"No rows found for solver={args.solver!r} at K={K_plot}.")

    metric_col = str(args.metric)
    if metric_col not in sub.columns:
        raise ValueError(f"Requested metric={metric_col} not found in CSV. Columns: {list(sub.columns)}")
    if "minimax_lb_E_x" not in sub.columns:
        raise ValueError(
            "CSV is missing minimax_lb_E_x. Re-run the experiment scripts from this patched codebase."
        )

    # Parse quantiles.
    qs = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    if not qs:
        qs = [0.25, 0.5, 0.75]
    for qv in qs:
        if not (0.0 <= qv <= 1.0):
            raise ValueError("Quantiles must be in [0,1].")

    # Aggregate per q.
    rows = []
    for q in sorted(sub["q"].unique()):
        ss = sub[sub["q"] == q]
        vals = ss[metric_col].to_numpy(dtype=float)
        quant = np.quantile(vals, qs)
        lb = float(ss["minimax_lb_E_x"].iloc[0])
        rows.append({"q": int(q), "lb": lb, **{f"q{int(100*qv)}": float(v) for qv, v in zip(qs, quant)}})
    agg = pd.DataFrame(rows).sort_values("q")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Plot quantile curves.
    for qv in qs:
        col = f"q{int(100*qv)}"
        label = f"{int(100*qv)}th pct" if qv != 0.5 else "median"
        ax.plot(agg["q"], agg[col], marker="o", label=label)

    ax.plot(agg["q"], agg["lb"], linestyle="--", linewidth=1.2, label="minimax LB (x, chordal)")

    ax.set_xlabel("Gap length q")
    if metric_col == "single_error":
        ax.set_ylabel(r"Single-output x-error $\|E_x(\hat x_{q-1})-E_x(x_{q-1})\|_2$")
    elif metric_col == "single_error_full":
        ax.set_ylabel("Single-output full-state embedding error")
    elif metric_col == "best_error":
        ax.set_ylabel(r"Coverage x-error $\min_{j\leq K}\|E_x(\hat x_{q-1}^{(j)})-E_x(x_{q-1})\|_2$")
    else:
        ax.set_ylabel("Coverage full-state embedding error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    out_png = os.path.join(args.outdir, "minimax_bound_plot.png")
    out_pdf = os.path.join(args.outdir, "minimax_bound_plot.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
