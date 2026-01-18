# Shadowing vs Identifiability: Supplementary Code and Data

This package provides the complete computational workflow and accompanying outputs for the shadowing versus identifiability study on hyperbolic toral automorphisms. It includes the synthetic data generator, solver implementations, evaluation metrics, plotting scripts, and the precomputed results and figures reported in the supplementary material. All outputs can be regenerated from the provided code and fixed seeds.

## Repository layout

- `0_domain_study/`: reserved (intentionally empty in this package)
- `1_datasets/`: synthetic data generation code and dataset metadata
- `2_data_preparation/`: data preparation steps (reserved)
- `3_data_exploration/`: exploratory notebooks and diagnostics
- `4_data_analysis/`: evaluation metrics and plotting scripts
- `5_models/`: system definitions and solver implementations
- `6_results/`: precomputed result tables and summaries
- `7_reports/`: precomputed figures and supplementary plots
- `scripts/`: experimental runners and utilities
- `tests/`: unit tests
- `docs/`: extended documentation (reserved)

## Environment

Python 3.10 or later is required. Install pinned dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducible workflow

### Run Block A (noiseless endpoints + Newton baseline)

```bash
python scripts/run_blockA.py --outdir 6_results --N_inst 10 --K_max 64 --max_steps 1500 --lambda_dyn 1.0
```

### Run Block B (digital-to-analog crossover)

```bash
python scripts/run_blockB.py --outdir 6_results --N_inst 20 --K_max 64 --max_steps 1500 --lambda_dyn 1.0 --sigma 1e-3
```

### Generate figures

```bash
python 4_data_analysis/figures/plot_collapse.py --summary_csv 6_results/blockA_summary.csv --outdir 7_reports/figures --xaxis K_over_bq
python 4_data_analysis/figures/plot_Lshape.py --hyp_csv 6_results/blockA_hypotheses.csv --outdir 7_reports/figures
python 4_data_analysis/figures/plot_minimax_bound.py --success_csv 6_results/blockA_success.csv --outdir 7_reports/figures --solver newton --K 64 --metric single_error
python 4_data_analysis/figures/plot_horizon.py --summary_csv 6_results/blockB_summary.csv --outdir 7_reports/figures --K 64 --show_spacing
```

### Manuscript figures

```bash
python 4_data_analysis/figures/plot_paper_figures.py --success_csv 6_results/blockA_success.csv --outdir 7_reports/figures --A_name A1 --solver newton --K_for_minimax 64
```

## Determinism notes

- All experiment runners accept explicit seeds; default seeds match the precomputed outputs.
- Torch deterministic algorithms can be enabled where required by the computational environment.

## Data policy

- All datasets and outputs in this package are synthetic and generated locally.
- Precomputed results are included in `6_results/` and figures in `7_reports/`.
- Re-running the pipeline will overwrite existing outputs in these directories.

## Tests

```bash
pytest -q
```
