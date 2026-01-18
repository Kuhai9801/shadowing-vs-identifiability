"""Digital -> analog crossover plot for Block B.

This script reads Block B summary CSV and plots P_success(q) at fixed sigma.
It overlays the predicted information horizon

    q_crit(\sigma; A) = (1/h_KS) * ln(1/\sigma).

Optionally, it also overlays the branch spacing 1/|b_q| as a reference curve.

Usage examples:
  python -m figures.plot_horizon --summary_csv 6_results/blockB_summary.csv --K 64

Outputs:
  7_reports/figures/horizon_plot.png
  7_reports/figures/horizon_plot.pdf
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

from systems.torus_map import bq, hks_entropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate digital->analog horizon plot (Block B).")
    parser.add_argument("--summary_csv", type=str, default="6_results/blockB_summary.csv")
    parser.add_argument("--outdir", type=str, default="7_reports/figures")
    parser.add_argument("--K", type=int, default=None, help="Which K to plot (default: max K in CSV).")
    parser.add_argument("--show_spacing", action="store_true", help="Overlay sigma and 1/|b_q|.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise RuntimeError("Summary CSV is empty.")

    # Identify sigma (should be constant in Block B).
    sigmas = sorted(df["sigma"].unique())
    if len(sigmas) != 1:
        raise ValueError(f"Expected a single sigma for Block B; got {sigmas}.")
    sigma = float(sigmas[0])

    # Determine K.
    K_plot = int(args.K) if args.K is not None else int(df["K"].max())

    sub = df[df["K"] == K_plot].copy()
    if sub.empty:
        raise ValueError(f"No rows found for K={K_plot}.")

    # Block B uses A1.
    A = np.array([[2, 1], [1, 1]], dtype=int)
    h = hks_entropy(A)
    q_crit = float((1.0 / h) * np.log(1.0 / sigma))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for solver in sorted(sub["solver"].unique()):
        ss = sub[sub["solver"] == solver].sort_values("q")
        ax.plot(ss["q"], ss["P_success"], marker="o", label=f"{solver} (K={K_plot})")

    ax.axvline(q_crit, linestyle="--", linewidth=1.2, label=r"$q_{crit}=(1/h_{KS})\ln(1/\sigma)$")

    ax.set_xlabel("Gap length q")
    ax.set_ylabel(r"Empirical success probability $P(\mathrm{Succ}(K)=1)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    if args.show_spacing:
        # Overlay sigma and 1/|b_q| on a twin axis.
        ax2 = ax.twinx()
        qs = sorted(sub["q"].unique())
        spacing = [1.0 / abs(bq(A, int(q))) for q in qs]
        half_spacing = [0.5 * s for s in spacing]
        ax2.plot(qs, spacing, linestyle="-", marker="x", label=r"$1/|b_q|$")
        ax2.plot(qs, half_spacing, linestyle="--", marker=".", label=r"$1/(2|b_q|)$")
        ax2.axhline(sigma, linestyle=":", linewidth=1.2, label=r"$\sigma$")
        ax2.set_ylabel(r"Branch scale $1/|b_q|$, $1/(2|b_q|)$ and noise $\sigma$")

        # Combine legends from both axes.
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)
    else:
        ax.legend(loc="best", frameon=True)

    out_png = os.path.join(args.outdir, "horizon_plot.png")
    out_pdf = os.path.join(args.outdir, "horizon_plot.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
