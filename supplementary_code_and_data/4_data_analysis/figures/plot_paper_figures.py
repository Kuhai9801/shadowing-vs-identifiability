"""Generate the two manuscript figures used in the CHAOS submission.

This script reads the per-instance success table produced by Block A
(`6_results/blockA_success.csv`) and produces:

  - fig2_coverage_success_ratio_A1_newton.pdf
  - fig1_minimax_bound_A1_newton.pdf

The filenames match the manuscript includegraphics statements.

Usage (from repository root):

  python 4_data_analysis/figures/plot_paper_figures.py --success_csv 6_results/blockA_success.csv --outdir 7_reports/figures

The script is deterministic given the underlying CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper) in [0,1].
    """
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z**2) / (4.0 * n**2))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--success_csv', type=str, default='6_results/blockA_success.csv', help='Path to blockA_success.csv')
    parser.add_argument('--outdir', type=str, default='7_reports/figures', help='Output directory for figures')
    parser.add_argument('--A_name', type=str, default='A1', help='Matrix name (default: A1)')
    parser.add_argument('--solver', type=str, default='newton', choices=['newton', 'neural'], help='Solver (default: newton)')
    parser.add_argument('--sigma', type=float, default=0.0, help='Endpoint noise level sigma (default: 0.0)')
    parser.add_argument('--K_for_minimax', type=int, default=64, help='K used for the single-output minimax plot')
    args = parser.parse_args()

    success_csv = Path(args.success_csv)
    outdir = Path(args.outdir)
    figdir = outdir
    figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(success_csv)

    # Use AIP-friendly PDF font embedding (avoid Type 3 fonts).
    mpl.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 1.0,
        'lines.markersize': 3.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
    })

    # ---------------------------------------------------------------------
    # Figure: minimax bound vs single-output error (Block A, sigma=0)
    # ---------------------------------------------------------------------
    df_min = df[(df['block'] == 'A')
                & (df['A_name'] == args.A_name)
                & (df['solver'] == args.solver)
                & (df['sigma'] == args.sigma)
                & (df['K'] == args.K_for_minimax)].copy()

    if df_min.empty:
        raise RuntimeError('No rows matched minimax plot filters. Check args.')

    qs = np.array(sorted(df_min['q'].unique()))
    med, q25, q75, lb = [], [], [], []
    for q in qs:
        sub = df_min[df_min['q'] == q]
        vals = sub['single_error'].to_numpy(dtype=float)
        med.append(np.nanmedian(vals))
        q25.append(np.nanquantile(vals, 0.25))
        q75.append(np.nanquantile(vals, 0.75))
        lb.append(float(sub['minimax_lb_E_x'].iloc[0]))

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.fill_between(qs, q25, q75, alpha=0.2, linewidth=0)
    ax.plot(qs, med, marker='o')
    ax.plot(qs, lb, linestyle='--')
    ax.set_xlabel(r'Gap length $q$')
    ax.set_ylabel(r'Chordal $x$-error at $q-1$')
    ax.set_xticks(qs)
    ax.set_xlim(qs.min() - 0.25, qs.max() + 0.25)
    ymax = max(max(q75), max(lb))
    ax.set_ylim(0.0, 1.05 * ymax)
    ax.grid(True, which='major', alpha=0.25)
    fig.tight_layout(pad=0.2)
    fig_path = figdir / 'fig1_minimax_bound_A1_newton.pdf'
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure: coverage success rate vs restart ratio K/|b_q| (Block A, sigma=0)
    # ---------------------------------------------------------------------
    df_cov = df[(df['block'] == 'A')
                & (df['A_name'] == args.A_name)
                & (df['solver'] == args.solver)
                & (df['sigma'] == args.sigma)].copy()

    if df_cov.empty:
        raise RuntimeError('No rows matched coverage plot filters. Check args.')

    grp = df_cov.groupby(['q', 'K'], as_index=False).agg(
        k=('success', 'sum'),
        n=('success', 'count'),
        bq_abs=('bq_abs', 'first')
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.4))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']
    linestyles = ['-', '--', '-.', ':']

    for i, q in enumerate(sorted(grp['q'].unique())):
        sub = grp[grp['q'] == q].sort_values('K')
        bq = float(sub['bq_abs'].iloc[0])
        x = sub['K'].to_numpy(dtype=float) / bq
        p = sub['k'].to_numpy(dtype=float) / sub['n'].to_numpy(dtype=float)
        lo, hi = [], []
        for k_i, n_i in zip(sub['k'].astype(int), sub['n'].astype(int)):
            l, u = wilson_interval(k_i, n_i)
            lo.append(l)
            hi.append(u)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        yerr = np.vstack([p - lo, hi - p])

        label = rf'$q={int(q)}\ (|b_q|={int(bq)})$'
        ax.errorbar(
            x, p, yerr=yerr,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            capsize=2.0, elinewidth=0.8, capthick=0.8,
            label=label,
        )

    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'Restart ratio $K/|b_q|$')
    ax.set_ylabel('Coverage success rate')
    ax.grid(True, which='major', alpha=0.20)
    # Place the legend in an information-sparse region (upper left on a log-x axis)
    # to avoid occluding the transition region near K/|b_q| \sim 1.
    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.90,
        borderpad=0.3,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    fig.tight_layout(pad=0.2)
    fig_path = figdir / 'fig2_coverage_success_ratio_A1_newton.pdf'
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print('Wrote:', figdir / 'fig1_minimax_bound_A1_newton.pdf')
    print('Wrote:', figdir / 'fig2_coverage_success_ratio_A1_newton.pdf')


if __name__ == '__main__':
    main()
