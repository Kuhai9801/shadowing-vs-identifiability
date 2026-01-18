"""Money Plot 2: L-shape scatter (shadowing vs inference paradox).

Scatter all hypotheses across runs:
  * x-axis: imputation error e^(j) in embedding space.
  * y-axis: defect δ^(j) (embedding-space max defect).
  * color: symbolic Hamming distance d_H^(j).

Expected structure: dense region of low defect but high error with nonzero d_H
("valid shadows, wrong history").

This script reads a hypotheses CSV (Block A or Block B) and saves a PNG/PDF.
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

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Money Plot 2 (L-shape scatter).")
    parser.add_argument("--hyp_csv", type=str, default="6_results/blockA_hypotheses.csv")
    parser.add_argument("--outdir", type=str, default="7_reports/figures")
    parser.add_argument(
        "--x_metric",
        type=str,
        default="e_impute_x",
        choices=["e_impute_x", "e_impute"],
        help="Which imputation error metric to plot on the x-axis. Default: e_impute_x (x-only).",
    )
    parser.add_argument("--max_points", type=int, default=200000, help="Optional subsample cap for plotting.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.hyp_csv)
    if df.empty:
        raise RuntimeError("Hypotheses CSV is empty.")

    # Optional subsample for readability.
    if len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    x_col = str(args.x_metric)
    if x_col not in df.columns:
        raise ValueError(f"Requested x_metric={x_col} not found in hypotheses CSV. Available columns: {list(df.columns)}")

    sc = ax.scatter(df[x_col], df["defect_E_max"], c=df["d_H"], s=10, alpha=0.75)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Normalized Hamming distance d_H")

    if x_col == "e_impute_x":
        ax.set_xlabel(r"X-imputation error $\|E_x(\hat x_{q-1})-E_x(x_{q-1})\|_2$")
    else:
        ax.set_xlabel("Imputation error e (full-state embedding chordal norm)")
    ax.set_ylabel("Defect δ (max embedding residual)")
    ax.grid(True, alpha=0.25)

    out_png = os.path.join(args.outdir, "money_plot2_Lshape.png")
    out_pdf = os.path.join(args.outdir, "money_plot2_Lshape.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
