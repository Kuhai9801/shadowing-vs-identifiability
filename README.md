# Shadowing vs Identifiability

This repository provides a reproducible pipeline for synthetic chaos experiments on hyperbolic toral automorphisms, comparing shadowing-based reconstruction against identifiability limits. The workflow generates controlled instances, runs neural and Newton-based solvers, computes evaluation metrics, and produces the figures used to analyze the shadowing versus identifiability gap.

Python 3.10+ is required. Dependencies are pinned in `requirements.txt`.

## Repository layout

- `0_domain_study/`: reserved (intentionally empty in this package)
- `1_datasets/`: synthetic data generation code and dataset metadata
- `2_data_preparation/`: data cleaning and preparation steps
- `3_data_exploration/`: exploratory notebooks and diagnostics
- `4_data_analysis/`: metrics and plotting scripts
- `5_models/`: system definitions and solver implementations
- `6_results/`: generated result tables and summaries
- `7_reports/`: figures, tables, and writeups derived from results
- `scripts/`: entrypoints and CLI utilities
- `tests/`: unit tests and smoke checks
- `docs/`: extended documentation

## Reproducible workflow

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke test

```bash
pytest -q tests/test_smoke.py
```

### Full run

```bash
python scripts/run_pipeline.py --mode full --device cpu --num_workers 1 --seed 12345
```

## Determinism notes

- All CLI entrypoints accept explicit seeds; `scripts/run_pipeline.py` propagates fixed seeds to every stage.
- Deterministic algorithms are supported via `--torch_deterministic` for Block A/B, and are enabled by default in the pipeline runner; use `--no_deterministic` to disable if required by backend constraints.
- GPU determinism depends on backend support and may reduce performance.

## Data policy

- All data are synthetic and generated locally by the pipeline. No external downloads are required.
- This supplementary package tracks generated data, results, and figures under `6_results/` and `7_reports/`.
- Recomputations may overwrite existing outputs in these directories.

## Reproducing results

Run the stages explicitly (outputs are shown relative to repo root):

```bash
python scripts/run_blockA.py --outdir 6_results --seed 12345 --N_inst 10 --K_max 64 --max_steps 1500 --lambda_dyn 1.0 --q_list_mode legacy --device cpu --num_workers 1 --torch_deterministic
python scripts/run_blockB.py --outdir 6_results --seed 12346 --N_inst 20 --K_max 64 --max_steps 1500 --lambda_dyn 1.0 --sigma 1e-3 --q_list 4,5,6,7,8,9,10 --device cpu --num_workers 1 --torch_deterministic

python 4_data_analysis/figures/plot_collapse.py --summary_csv 6_results/blockA_summary.csv --outdir 7_reports/figures --xaxis K_over_bq
python 4_data_analysis/figures/plot_Lshape.py --hyp_csv 6_results/blockA_hypotheses.csv --outdir 7_reports/figures --x_metric e_impute_x
python 4_data_analysis/figures/plot_minimax_bound.py --success_csv 6_results/blockA_success.csv --outdir 7_reports/figures --solver newton --K 64 --metric single_error
python 4_data_analysis/figures/plot_horizon.py --summary_csv 6_results/blockB_summary.csv --outdir 7_reports/figures --K 64
```

Expected outputs:
- Results: `6_results/blockA_*.csv`, `6_results/blockB_*.csv`, `6_results/*_meta.json`
- Figures: `7_reports/figures/money_plot1_collapse_K_over_bq.png`, `7_reports/figures/money_plot2_Lshape.png`, `7_reports/figures/minimax_bound_plot.png`, `7_reports/figures/horizon_plot.png`

## Troubleshooting

- Torch installation issues: install CPU-only wheels or follow the PyTorch installation guidance for CUDA-enabled environments.
- Deterministic mode errors: rerun with `--no_deterministic` in the pipeline or omit `--torch_deterministic` in Block A/B.
- Import errors: run commands from the repository root so that pipeline paths resolve correctly.

## License

MIT License. See `LICENSE`.

## Citation

```bibtex
@software{zarceno_shadowing_vs_identifiability_2026,
  author = {Zarceno, Cyne Jarvis J.},
  title = {Shadowing vs Identifiability},
  year = {2026},
  url = {https://github.com/Kuhai9801/shadowing-vs-identifiability}
}
```

## Contact

cynejarvis.res@gmail.com
