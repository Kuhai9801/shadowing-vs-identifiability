"""One-command reproducible pipeline runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: List[str], *, env: dict) -> None:
    printable = " ".join(cmd)
    print(f"[pipeline] {printable}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full shadowing vs identifiability pipeline.")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "smoke"])
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for Block A.")
    parser.add_argument("--seed_blockB", type=int, default=None, help="Optional seed override for Block B.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda).")
    parser.add_argument("--num_workers", type=int, default=1, help="Worker processes for Block A/B.")
    parser.add_argument("--results_dir", type=str, default="6_results", help="Base results directory.")
    parser.add_argument("--reports_dir", type=str, default="7_reports", help="Base reports directory.")
    parser.add_argument("--smoke_subdir", type=str, default="smoke", help="Subdir appended in smoke mode.")
    parser.add_argument("--skip_blockB", action="store_true", help="Skip Block B stage.")
    parser.add_argument("--skip_figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--with_newton_blockB", action="store_true", help="Enable Newton baseline in Block B.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable torch deterministic algorithms.",
    )
    parser.add_argument(
        "--no_deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable torch deterministic algorithms.",
    )
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    results_dir = args.results_dir
    reports_dir = args.reports_dir
    if args.mode == "smoke":
        results_dir = os.path.join(results_dir, args.smoke_subdir)
        reports_dir = os.path.join(reports_dir, args.smoke_subdir)

    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    seed_blockA = int(args.seed)
    seed_blockB = int(args.seed_blockB) if args.seed_blockB is not None else int(args.seed) + 1

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed_blockA)

    if args.mode == "smoke":
        blockA_args = [
            sys.executable,
            os.path.join(_ROOT, "scripts", "run_blockA.py"),
            "--outdir",
            results_dir,
            "--seed",
            str(seed_blockA),
            "--N_inst",
            "1",
            "--K_max",
            "2",
            "--max_steps",
            "10",
            "--lambda_dyn",
            "1.0",
            "--q_list",
            "6",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--quiet",
        ]
        if args.deterministic:
            blockA_args.append("--torch_deterministic")

        blockB_args = [
            sys.executable,
            os.path.join(_ROOT, "scripts", "run_blockB.py"),
            "--outdir",
            results_dir,
            "--seed",
            str(seed_blockB),
            "--N_inst",
            "1",
            "--K_max",
            "2",
            "--max_steps",
            "10",
            "--lambda_dyn",
            "1.0",
            "--sigma",
            "1e-3",
            "--q_list",
            "4",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--quiet",
        ]
        if args.with_newton_blockB:
            blockB_args.append("--with_newton")
        if args.deterministic:
            blockB_args.append("--torch_deterministic")

    else:
        blockA_args = [
            sys.executable,
            os.path.join(_ROOT, "scripts", "run_blockA.py"),
            "--outdir",
            results_dir,
            "--seed",
            str(seed_blockA),
            "--N_inst",
            "10",
            "--K_max",
            "64",
            "--max_steps",
            "1500",
            "--lambda_dyn",
            "1.0",
            "--q_list_mode",
            "legacy",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        if args.deterministic:
            blockA_args.append("--torch_deterministic")

        blockB_args = [
            sys.executable,
            os.path.join(_ROOT, "scripts", "run_blockB.py"),
            "--outdir",
            results_dir,
            "--seed",
            str(seed_blockB),
            "--N_inst",
            "20",
            "--K_max",
            "64",
            "--max_steps",
            "1500",
            "--lambda_dyn",
            "1.0",
            "--sigma",
            "1e-3",
            "--q_list",
            "4,5,6,7,8,9,10",
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        if args.with_newton_blockB:
            blockB_args.append("--with_newton")
        if args.deterministic:
            blockB_args.append("--torch_deterministic")

    _run(blockA_args, env=env)

    if not args.skip_blockB:
        env_blockB = env.copy()
        env_blockB["PYTHONHASHSEED"] = str(seed_blockB)
        _run(blockB_args, env=env_blockB)

    if args.skip_figures:
        return

    # Figures (Block A).
    _run(
        [
            sys.executable,
            os.path.join(_ROOT, "4_data_analysis", "figures", "plot_collapse.py"),
            "--summary_csv",
            os.path.join(results_dir, "blockA_summary.csv"),
            "--outdir",
            figures_dir,
            "--xaxis",
            "K_over_bq",
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            os.path.join(_ROOT, "4_data_analysis", "figures", "plot_Lshape.py"),
            "--hyp_csv",
            os.path.join(results_dir, "blockA_hypotheses.csv"),
            "--outdir",
            figures_dir,
            "--x_metric",
            "e_impute_x",
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            os.path.join(_ROOT, "4_data_analysis", "figures", "plot_minimax_bound.py"),
            "--success_csv",
            os.path.join(results_dir, "blockA_success.csv"),
            "--outdir",
            figures_dir,
            "--solver",
            "newton",
            "--K",
            "2" if args.mode == "smoke" else "64",
            "--metric",
            "single_error",
        ],
        env=env,
    )

    # Figures (Block B).
    if not args.skip_blockB:
        _run(
            [
                sys.executable,
                os.path.join(_ROOT, "4_data_analysis", "figures", "plot_horizon.py"),
                "--summary_csv",
                os.path.join(results_dir, "blockB_summary.csv"),
                "--outdir",
                figures_dir,
                "--K",
                "2" if args.mode == "smoke" else "64",
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
