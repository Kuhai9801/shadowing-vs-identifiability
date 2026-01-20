# Reproducibility Record

## Environment

- OS: Microsoft Windows 11 Home Single Language (10.0.26100, 64-bit, build 26100.1.amd64fre.ge_release.240331-1435)
- Python: 3.12.4
- CPU: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (4 cores, 8 logical processors)
- RAM: 8.36 GB (7.79 GiB)

## Hardware expectations

- CPU-only execution is supported.
- Minimum memory: 8 GB RAM; recommended: 16 GB RAM for full runs.
- GPU acceleration is optional and not required for reproducibility.

## Dependencies

- `requirements.txt` lists top-level dependencies.
- `requirements.lock` pins a fully resolved set for Windows and Python 3.12; regenerate when the OS or Python version changes.

## Full pipeline run transcript

### Environment setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.lock
```

### Full pipeline execution

```bash
python scripts/run_pipeline.py --mode full --device cpu --num_workers 1 --seed 12345 --results_dir 6_results --reports_dir 7_reports
```

Expected outputs:

- `6_results/blockA_hypotheses.csv`
- `6_results/blockA_success.csv`
- `6_results/blockA_summary.csv`
- `6_results/blockA_single.csv`
- `6_results/blockA_single_summary.csv`
- `6_results/blockA_meta.json`
- `6_results/blockB_hypotheses.csv`
- `6_results/blockB_success.csv`
- `6_results/blockB_summary.csv`
- `6_results/blockB_single.csv`
- `6_results/blockB_single_summary.csv`
- `6_results/blockB_meta.json`
- `7_reports/figures/money_plot1_collapse_K_over_bq.png`
- `7_reports/figures/money_plot1_collapse_K_over_bq.pdf`
- `7_reports/figures/money_plot2_Lshape.png`
- `7_reports/figures/money_plot2_Lshape.pdf`
- `7_reports/figures/minimax_bound_plot.png`
- `7_reports/figures/minimax_bound_plot.pdf`
- `7_reports/figures/horizon_plot.png`
- `7_reports/figures/horizon_plot.pdf`

### Manuscript figures

```bash
python 4_data_analysis/figures/plot_paper_figures.py --success_csv 6_results/blockA_success.csv --outdir 7_reports/figures --A_name A1 --solver newton --K_for_minimax 64
```

Expected outputs:

- `7_reports/figures/fig1_minimax_bound_A1_newton.pdf`
- `7_reports/figures/fig2_coverage_success_ratio_A1_newton.pdf`

### Data integrity verification

```bash
python scripts/validate_data.py --manifest data_manifest.json --root .
```

Expected output:

- `[ok] <count> files validated`
