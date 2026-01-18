"""Money Plot 1: scaling collapse with forbidden zone and Newton overlay.

This script reads Block A summary CSV and produces a scaling-collapse plot.

Specification (plan-aligned)
----------------------------
  * x-axis: eta := K * exp(-h_KS q) (option) OR exact K/|b_q| (recommended).
  * y-axis: empirical success probability P(Succ(K)=1).
  * overlay: theoretical wall K/|b_q|.
  * shade (or highlight) region where empirical P_success > K/|b_q| as the
    "Information-Theoretic Forbidden Zone".
  * overlay neural and Newton results.

The output is saved as a PNG and PDF in the output directory.
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
    parser = argparse.ArgumentParser(description="Generate Money Plot 1 (scaling collapse).")
    parser.add_argument("--summary_csv", type=str, default="6_results/blockA_summary.csv")
    parser.add_argument("--outdir", type=str, default="7_reports/figures")
    parser.add_argument(
        "--xaxis",
        type=str,
        default="K_over_bq",
        choices=["K_over_bq", "eta"],
        help="x-axis choice: exact K/|b_q| (recommended) or eta=K*exp(-h q).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise RuntimeError("Summary CSV is empty.")

    # Compute x-axis values and the theoretical wall.
    x_vals = []
    wall_vals = []
    eta_vals = []

    for _, row in df.iterrows():
        A_name = str(row["A_name"])
        q = int(row["q"])
        K = int(row["K"])

        # Reconstruct A by name using the same defaults.
        # (We avoid importing data.generate to keep plotting scripts lightweight.)
        if A_name == "A1":
            A = np.array([[2, 1], [1, 1]], dtype=int)
        elif A_name == "A2":
            A = np.array([[3, 2], [1, 1]], dtype=int)
        elif A_name == "A3":
            A = np.array([[4, 3], [1, 1]], dtype=int)
        else:
            raise ValueError(f"Unknown A_name={A_name}.")

        h = hks_entropy(A)
        eta = float(K * np.exp(-h * q))
        b_abs = abs(bq(A, q))
        wall = float(min(1.0, K / b_abs))

        eta_vals.append(eta)
        wall_vals.append(wall)
        if args.xaxis == "eta":
            x_vals.append(eta)
        else:
            x_vals.append(float(K / b_abs))

    df = df.copy()
    df["x"] = x_vals
    df["eta"] = eta_vals
    df["wall"] = wall_vals
    df["violation"] = df["P_success"] > df["wall"] + 1e-12

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for solver in sorted(df["solver"].unique()):
        sub = df[df["solver"] == solver]
        ax.scatter(sub["x"], sub["P_success"], label=solver, alpha=0.85)

    # Plot the wall as a diagonal if xaxis=K/|b_q|, else plot wall points.
    if args.xaxis == "K_over_bq":
        # y = x clipped to [0,1].
        xx = np.linspace(0.0, max(1e-12, float(df["x"].max())), 200)
        yy = np.minimum(1.0, xx)
        ax.plot(xx, yy, linestyle="--", linewidth=1.2, label="wall: K/|b_q|")
        # Shade forbidden zone above the wall.
        ax.fill_between(xx, yy, 1.0, alpha=0.12)
        ax.text(
            0.6 * float(xx.max()),
            0.92,
            "Information-Theoretic\nForbidden Zone",
            ha="center",
            va="top",
            fontsize=9,
        )
    else:
        # Plot wall points in (eta, wall).
        wsub = df.sort_values("x")
        ax.plot(wsub["x"], wsub["wall"], linestyle="--", linewidth=1.2, label="wall: K/|b_q| (points)")
        # Highlight violations.
        viol = df[df["violation"]]
        if not viol.empty:
            ax.scatter(viol["x"], viol["P_success"], marker="x", s=60, label="violations")

    ax.set_xlabel("K/|b_q|" if args.xaxis == "K_over_bq" else r"$\eta = K e^{-h_{KS} q}$")
    ax.set_ylabel(r"Empirical success probability $P(\mathrm{Succ}(K)=1)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    out_png = os.path.join(args.outdir, f"money_plot1_collapse_{args.xaxis}.png")
    out_pdf = os.path.join(args.outdir, f"money_plot1_collapse_{args.xaxis}.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
