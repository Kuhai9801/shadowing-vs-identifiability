"""Run Block A experiments (noiseless endpoints) and write results to disk.

Block A specification (from the plan)
------------------------------------
  * Matrices: A in {A1,A2,A3}.
  * Gap lengths: q in {6,10,14,18,22}.
  * Noise: sigma = 0.
  * Instances per (A,q): N_inst = 10.
  * Hypotheses per instance: K_max = 64.
  * Solvers: Neural-DA and Newton.

Outputs
-------
Writes three CSV files in the chosen output directory:
  * blockA_hypotheses.csv : per-hypothesis metrics.
  * blockA_success.csv    : per-instance success indicators for each K.
  * blockA_summary.csv    : empirical success probabilities aggregated across instances.
Also writes:
  * blockA_meta.json
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on PYTHONPATH when the script is executed by path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PIPELINE_DIRS = ["1_datasets", "4_data_analysis", "5_models"]
for rel in _PIPELINE_DIRS:
    path = os.path.join(_ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from data.generate import default_matrices, generate_grid
from eval.metrics import compute_metrics, symbolic_itinerary
from systems.torus_map import (
    bq,
    embedding_circle_distance_from_torus_distance,
    interior_x_separation,
    splitting_data,
)
from scripts.common import (
    compute_global_epsilon_embedding_x,
    default_K_list,
    parse_int_list,
    suggest_q_list_for_bq_range,
)
from solvers.neural_4dvar import Neural4DVarConfig, solve_neural_4dvar
from solvers.newton_shooting import NewtonConfig, solve_newton_shooting


def _optional_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None
    return tqdm


def _print_device_banner() -> None:
    """Print a short, Kaggle-friendly hardware banner so you can see what the script is using."""
    print("=" * 80)
    print("Block A runner starting")
    print(f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"cuda_device_count={n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            name = getattr(props, "name", "unknown")
            total_mem_gb = getattr(props, "total_memory", 0) / (1024**3)
            print(f"  GPU {i}: {name} | {total_mem_gb:.1f} GB")
    print("=" * 80)


def _run_instances(
    *,
    instances_with_ids: Sequence[Tuple[int, Any]],
    ncfg: Neural4DVarConfig,
    newton_cfg: NewtonConfig,
    K_list: Sequence[int],
    eps_success: float,
    progress_q: Optional[Any] = None,
    quiet: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run solvers for a batch of instances and return (hyp_rows, succ_rows)."""

    hyp_rows: List[Dict[str, Any]] = []
    succ_rows: List[Dict[str, Any]] = []

    t_batch0 = time.time()

    for j_inst, (inst_id, inst) in enumerate(instances_with_ids):
        t0 = time.time()

        # Per-instance invariants reused across all hypotheses.
        # (These are inexpensive but occur many thousands of times across the full grid.)
        m_true = symbolic_itinerary(inst.A, inst.traj_true)
        C_A = float(splitting_data(inst.A).C)

        # Theory invariants (depend only on A,q and are used for auditing/plotting).
        bq_abs = abs(int(bq(inst.A, inst.q)))
        bq_m1_abs = abs(int(bq(inst.A, inst.q - 1))) if int(inst.q) >= 1 else 0
        spacing_y0 = 1.0 / float(bq_abs) if bq_abs > 0 else float("nan")
        interior_sep_t1 = float(interior_x_separation(inst.A, inst.q)) if int(inst.q) >= 2 else float("nan")
        minimax_lb_t1 = 0.5 * interior_sep_t1 if np.isfinite(interior_sep_t1) else float("nan")
        minimax_lb_ex = (
            float(embedding_circle_distance_from_torus_distance(minimax_lb_t1))
            if np.isfinite(minimax_lb_t1)
            else float("nan")
        )
        digital_regime = int((float(inst.sigma) < (0.5 * spacing_y0))) if np.isfinite(spacing_y0) else 0

        # Neural solver ensemble.
        neural_hyps = solve_neural_4dvar(
            A=inst.A,
            q=inst.q,
            x0_obs=inst.x0_obs,
            xq_obs=inst.xq_obs,
            config=ncfg,
            base_seed=inst.seed + 17,
        )

        neural_errors: List[float] = []
        neural_errors_x: List[float] = []
        neural_dHs: List[float] = []
        neural_scores: List[float] = []
        for h in neural_hyps:
            metrics = compute_metrics(
                A=inst.A,
                z_hat=h.z_hat,
                traj_true=inst.traj_true,
                t_impute=inst.q - 1,
                m_true=m_true,
                C=C_A,
            )
            neural_errors.append(metrics.e_impute)
            neural_errors_x.append(metrics.e_impute_x)
            neural_dHs.append(metrics.d_H)
            neural_scores.append(h.loss_total)

            hyp_rows.append(
                {
                    "block": "A",
                    "solver": "neural",
                    "A_name": inst.A_name,
                    "q": inst.q,
                    "sigma": inst.sigma,
                    "instance_id": inst_id,
                    "instance_seed": inst.seed,
                    "restart_id": h.restart_id,
                    "restart_seed": h.seed,
                    "converged": int(h.converged),
                    "steps": h.steps,
                    "score": h.loss_total,
                    "train_seconds": h.train_seconds,
                    "loss_obs": h.loss_obs,
                    "loss_dyn": h.loss_dyn,
                    "defect_E_train": h.defect_E_max,
                    "e_impute": metrics.e_impute,
                    "e_impute_x": metrics.e_impute_x,
                    "defect_E_max": metrics.defect_E_max,
                    "defect_T_max": metrics.defect_T_max,
                    "eps_cert": metrics.eps_cert,
                    "d_H": metrics.d_H,
                    "bq_abs": int(bq_abs),
                    "bq_minus1_abs": int(bq_m1_abs),
                    "spacing_y0": float(spacing_y0),
                    "interior_sep_T1": float(interior_sep_t1),
                    "minimax_lb_T1": float(minimax_lb_t1),
                    "minimax_lb_E_x": float(minimax_lb_ex),
                    "digital_regime": int(digital_regime),
                }
            )

        # Sort hypotheses by solver score (available at inference time).
        #
        # IMPORTANT methodology note (rigor):
        #   * A single-output estimator corresponds to selecting the top-1 by score.
        #   * A best-of-K *coverage* metric corresponds to asking whether ANY of the top-K
        #     score-ranked hypotheses contains a correct branch / small error.
        #
        # These are distinct objects and must be reported separately.
        order = np.argsort(np.asarray(neural_scores, dtype=float), kind="mergesort")
        errs_sorted = np.asarray(neural_errors, dtype=float)[order]
        errs_sorted_x = np.asarray(neural_errors_x, dtype=float)[order]
        scores_sorted = np.asarray(neural_scores, dtype=float)[order]
        dHs_sorted = np.asarray(neural_dHs, dtype=float)[order]
        restart_ids_sorted = np.asarray([neural_hyps[int(i)].restart_id for i in order], dtype=int)

        # Single-output (top-1 by score).
        top1_restart_id = int(restart_ids_sorted[0])
        top1_score = float(scores_sorted[0])
        top1_error = float(errs_sorted_x[0])
        top1_error_full = float(errs_sorted[0])
        top1_d_H = float(dHs_sorted[0])
        top1_succ_impute = int(top1_error < eps_success)
        top1_succ_symbolic = int(top1_d_H <= 0.0)

        for K in K_list:
            K_eff = min(int(K), len(errs_sorted))

            # (A) Coverage imputation success (x-only): min e_impute_x within the top-K score-ranked candidates.
            # eps_success is computed in embedding-x units; using e_impute_x avoids an overly strict
            # success test that would inadvertently depend on the unobserved y coordinate.
            j_rel_err = int(np.argmin(errs_sorted_x[:K_eff]))
            best_restart_id = int(restart_ids_sorted[j_rel_err])
            best_error = float(errs_sorted_x[j_rel_err])
            best_error_full = float(errs_sorted[j_rel_err])
            best_score = float(scores_sorted[j_rel_err])
            succ_impute = int(best_error < eps_success)

            # (B) Symbolic/branch success: exact itinerary match (d_H == 0) within the top-K.
            j_rel_sym = int(np.argmin(dHs_sorted[:K_eff]))
            best_restart_id_sym = int(restart_ids_sorted[j_rel_sym])
            best_d_H = float(dHs_sorted[j_rel_sym])
            best_score_sym = float(scores_sorted[j_rel_sym])
            succ_symbolic = int(best_d_H <= 0.0)

            succ_rows.append(
                {
                    "block": "A",
                    "solver": "neural",
                    "A_name": inst.A_name,
                    "q": inst.q,
                    "sigma": inst.sigma,
                    "instance_id": inst_id,
                    "K": int(K),
                    # Primary success definition for the information-theoretic barrier plots.
                    "success": int(succ_symbolic),
                    # Legacy / auxiliary success definition used for diagnostics.
                    "success_impute": int(succ_impute),
                    # Single-output estimator induced by the solver: choose the top-1 by score.
                    "success_single": int(top1_succ_symbolic),
                    "success_single_impute": int(top1_succ_impute),
                    "eps_success_impute": float(eps_success),
                    "best_restart_id": best_restart_id,
                    "best_error": best_error,
                    "best_error_full": best_error_full,
                    "best_score": best_score,
                    "single_restart_id": int(top1_restart_id),
                    "single_score": float(top1_score),
                    "single_error": float(top1_error),
                    "single_error_full": float(top1_error_full),
                    "single_d_H": float(top1_d_H),
                    "best_restart_id_symbolic": best_restart_id_sym,
                    "best_d_H": best_d_H,
                    "best_score_symbolic": best_score_sym,
                    "bq_abs": int(bq_abs),
                    "spacing_y0": float(spacing_y0),
                    "interior_sep_T1": float(interior_sep_t1),
                    "minimax_lb_E_x": float(minimax_lb_ex),
                    "digital_regime": int(digital_regime),
                }
            )

        # Newton baseline ensemble.
        newton_hyps = solve_newton_shooting(
            A=inst.A,
            q=inst.q,
            x0=inst.x0_obs,
            xq=inst.xq_obs,
            config=newton_cfg,
            base_seed=inst.seed + 29,
        )

        newton_errors: List[float] = []
        newton_errors_x: List[float] = []
        newton_dHs: List[float] = []
        newton_scores: List[float] = []
        for h in newton_hyps:
            metrics = compute_metrics(
                A=inst.A,
                z_hat=h.traj,
                traj_true=inst.traj_true,
                t_impute=inst.q - 1,
                m_true=m_true,
                C=C_A,
            )
            newton_errors.append(metrics.e_impute)
            newton_errors_x.append(metrics.e_impute_x)
            newton_dHs.append(metrics.d_H)
            newton_scores.append(h.objective)

            hyp_rows.append(
                {
                    "block": "A",
                    "solver": "newton",
                    "A_name": inst.A_name,
                    "q": inst.q,
                    "sigma": inst.sigma,
                    "instance_id": inst_id,
                    "instance_seed": inst.seed,
                    "restart_id": h.restart_id,
                    "restart_seed": h.seed,
                    "converged": int(h.converged),
                    "steps": h.iters,
                    "score": h.objective,
                    "train_seconds": h.seconds,
                    "loss_obs": np.nan,
                    "loss_dyn": np.nan,
                    "defect_E_train": np.nan,
                    "e_impute": metrics.e_impute,
                    "e_impute_x": metrics.e_impute_x,
                    "defect_E_max": metrics.defect_E_max,
                    "defect_T_max": metrics.defect_T_max,
                    "eps_cert": metrics.eps_cert,
                    "d_H": metrics.d_H,
                    "bq_abs": int(bq_abs),
                    "bq_minus1_abs": int(bq_m1_abs),
                    "spacing_y0": float(spacing_y0),
                    "interior_sep_T1": float(interior_sep_t1),
                    "minimax_lb_T1": float(minimax_lb_t1),
                    "minimax_lb_E_x": float(minimax_lb_ex),
                    "digital_regime": int(digital_regime),
                }
            )

        order = np.argsort(np.asarray(newton_scores, dtype=float), kind="mergesort")
        errs_sorted = np.asarray(newton_errors, dtype=float)[order]
        errs_sorted_x = np.asarray(newton_errors_x, dtype=float)[order]
        scores_sorted = np.asarray(newton_scores, dtype=float)[order]
        dHs_sorted = np.asarray(newton_dHs, dtype=float)[order]
        restart_ids_sorted = np.asarray([newton_hyps[int(i)].restart_id for i in order], dtype=int)

        # Single-output (top-1 by score).
        top1_restart_id = int(restart_ids_sorted[0])
        top1_score = float(scores_sorted[0])
        top1_error = float(errs_sorted_x[0])
        top1_error_full = float(errs_sorted[0])
        top1_d_H = float(dHs_sorted[0])
        top1_succ_impute = int(top1_error < eps_success)
        top1_succ_symbolic = int(top1_d_H <= 0.0)

        for K in K_list:
            K_eff = min(int(K), len(errs_sorted))

            # (A) Imputation success (x-only auxiliary): min e_impute_x within the top-K.
            j_rel_err = int(np.argmin(errs_sorted_x[:K_eff]))
            best_restart_id = int(restart_ids_sorted[j_rel_err])
            best_error = float(errs_sorted_x[j_rel_err])
            best_error_full = float(errs_sorted[j_rel_err])
            best_score = float(scores_sorted[j_rel_err])
            succ_impute = int(best_error < eps_success)

            # (B) Symbolic/branch success (primary): exact itinerary match (d_H == 0).
            j_rel_sym = int(np.argmin(dHs_sorted[:K_eff]))
            best_restart_id_sym = int(restart_ids_sorted[j_rel_sym])
            best_d_H = float(dHs_sorted[j_rel_sym])
            best_score_sym = float(scores_sorted[j_rel_sym])
            succ_symbolic = int(best_d_H <= 0.0)

            succ_rows.append(
                {
                    "block": "A",
                    "solver": "newton",
                    "A_name": inst.A_name,
                    "q": inst.q,
                    "sigma": inst.sigma,
                    "instance_id": inst_id,
                    "K": int(K),
                    "success": int(succ_symbolic),
                    "success_impute": int(succ_impute),
                    "success_single": int(top1_succ_symbolic),
                    "success_single_impute": int(top1_succ_impute),
                    "eps_success_impute": float(eps_success),
                    "best_restart_id": best_restart_id,
                    "best_error": best_error,
                    "best_error_full": best_error_full,
                    "best_score": best_score,
                    "single_restart_id": int(top1_restart_id),
                    "single_score": float(top1_score),
                    "single_error": float(top1_error),
                    "single_error_full": float(top1_error_full),
                    "single_d_H": float(top1_d_H),
                    "best_restart_id_symbolic": best_restart_id_sym,
                    "best_d_H": best_d_H,
                    "best_score_symbolic": best_score_sym,
                    "bq_abs": int(bq_abs),
                    "spacing_y0": float(spacing_y0),
                    "interior_sep_T1": float(interior_sep_t1),
                    "minimax_lb_E_x": float(minimax_lb_ex),
                    "digital_regime": int(digital_regime),
                }
            )



        t1 = time.time()
        if progress_q is not None:
            progress_q.put(
                {
                    "inst_id": int(inst_id),
                    "A_name": str(inst.A_name),
                    "q": int(inst.q),
                    "seconds": float(t1 - t0),
                }
            )
        elif not quiet:
            elapsed = float(t1 - t_batch0)
            print(
                f"[{j_inst + 1}/{len(instances_with_ids)}] instance_id={inst_id} "
                f"A={inst.A_name} q={inst.q} | {t1 - t0:.2f}s (batch_elapsed={elapsed/60.0:.1f} min)"
            )

    return hyp_rows, succ_rows


def _worker_entry(
    *,
    worker_id: int,
    device_str: str,
    instances_with_ids: Sequence[Tuple[int, Any]],
    ncfg_dict: Dict[str, Any],
    newton_cfg_dict: Dict[str, Any],
    K_list: Sequence[int],
    eps_success: float,
    out_h_path: str,
    out_s_path: str,
    progress_q: Any,
    quiet: bool,
) -> None:
    """Worker process entrypoint (must be top-level for multiprocessing spawn)."""

    # Keep CPU thread usage under control when running multiple workers.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if device_str.startswith("cuda"):
        try:
            # device_str like "cuda:0"
            gpu_id = int(device_str.split(":")[1])
            torch.cuda.set_device(gpu_id)
        except Exception:
            pass

    ncfg = Neural4DVarConfig(**ncfg_dict)
    ncfg.device = device_str
    newton_cfg = NewtonConfig(**newton_cfg_dict)

    hyp_rows, succ_rows = _run_instances(
        instances_with_ids=instances_with_ids,
        ncfg=ncfg,
        newton_cfg=newton_cfg,
        K_list=K_list,
        eps_success=eps_success,
        progress_q=progress_q,
        quiet=True,  # worker stays quiet; main process prints progress
    )

    pd.DataFrame(hyp_rows).to_csv(out_h_path, index=False)
    pd.DataFrame(succ_rows).to_csv(out_s_path, index=False)

    # Signal worker completion.
    progress_q.put({"worker_done": int(worker_id)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Block A (noiseless endpoints) experiments.")
    parser.add_argument("--outdir", type=str, default="6_results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed.")
    parser.add_argument("--N_inst", type=int, default=10, help="Instances per (A,q).")
    parser.add_argument("--K_max", type=int, default=64, help="Restarts / hypotheses per instance.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Max Adam steps (neural).")
    parser.add_argument("--lambda_dyn", type=float, default=1.0, help="Lambda weight on dynamics loss.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda or cpu.")

    # Speed / visibility controls.
    parser.add_argument("--obs_tol", type=float, default=None, help="Early-stop obs tol (embedding-space).")
    parser.add_argument("--dyn_tol", type=float, default=None, help="Early-stop dyn tol (embedding-space).")
    parser.add_argument("--check_every", type=int, default=None, help="Check early-stop every N steps (>=1).")
    parser.add_argument(
        "--log_restarts_every",
        type=int,
        default=0,
        help="Print neural progress every N restarts (0 disables). Useful to see activity in long runs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Speed preset for Kaggle: looser early-stop + fewer steps (still deterministic).",
    )
    parser.add_argument(
        "--q_list",
        type=str,
        default=None,
        help=(
            "Comma-separated list of gap lengths q to run (overrides q_list_mode). "
            "Example: --q_list 4,5,6,7"
        ),
    )
    parser.add_argument(
        "--q_list_mode",
        type=str,
        default="legacy",
        choices=["legacy", "collapse", "full"],
        help=(
            "Predefined q grids. legacy matches the original plan. collapse emphasizes horizons where |b_q| is "
            "comparable to typical K (more informative for Money Plot 1). full is the union of legacy and collapse."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")

    # Parallelism: split instances across workers (and GPUs if available).
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 0 to auto-use all visible GPUs (or 1 if none).",
    )
    parser.add_argument(
        "--use_all_gpus",
        action="store_true",
        help="Convenience flag: set --num_workers to torch.cuda.device_count().",
    )

    args = parser.parse_args()

    # Ensure timely console updates in notebook environments (e.g., Kaggle).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.makedirs(args.outdir, exist_ok=True)

    if not args.quiet:
        _print_device_banner()

    matrices = default_matrices()
    # Horizon grid.
    if args.q_list is not None:
        q_list = parse_int_list(args.q_list)
    else:
        q_list_legacy = [6, 10, 14, 18, 22]
        q_list_collapse = suggest_q_list_for_bq_range(
            matrices,
            bq_min=20,
            bq_max=2000,
            q_max=12,
            require_all_matrices=False,
        )
        if args.q_list_mode == "legacy":
            q_list = q_list_legacy
        elif args.q_list_mode == "collapse":
            q_list = q_list_collapse
        else:
            q_list = sorted(set(q_list_legacy) | set(q_list_collapse))
    sigma = 0.0

    eps_success = compute_global_epsilon_embedding_x(matrices, q_list, factor=0.25)

    # Solver configs.
    ncfg = Neural4DVarConfig(
        K_max=args.K_max,
        max_steps=args.max_steps,
        lambda_dyn=float(args.lambda_dyn),
    )

    # Kaggle-friendly speed preset (keeps methodology identical; only affects optimizer runtime).
    if args.fast:
        # These are deliberately conservative relative to eps_success (~0.30 in embedding-x units for defaults).
        ncfg.max_steps = min(int(ncfg.max_steps), 800)
        ncfg.obs_tol = 1e-6
        ncfg.dyn_tol = 1e-3
        ncfg.check_every = 25

    if args.obs_tol is not None:
        ncfg.obs_tol = float(args.obs_tol)
    if args.dyn_tol is not None:
        ncfg.dyn_tol = float(args.dyn_tol)
    if args.check_every is not None:
        ncfg.check_every = int(args.check_every)

    # Optional visibility: print a short line every N completed neural restarts.
    ncfg.log_every_restarts = max(0, int(args.log_restarts_every))

    if int(args.log_restarts_every) > 0:
        ncfg.log_every_restarts = int(args.log_restarts_every)

    if args.device is not None:
        ncfg.device = args.device

    newton_cfg = NewtonConfig(K_max=args.K_max)

    # K grid.
    K_list = default_K_list(args.K_max)

    instances = generate_grid(
        matrices=matrices,
        q_list=q_list,
        sigma=sigma,
        n_instances=args.N_inst,
        seed0=args.seed,
    )

    instances_with_ids: List[Tuple[int, Any]] = list(enumerate(instances))

    total_instances = len(instances_with_ids)
    total_restarts = int(total_instances) * int(args.K_max)
    if not args.quiet:
        total_steps = total_restarts * int(ncfg.max_steps)
        print(f"Instances: {total_instances} | Neural restarts total: {total_restarts}")
        print(f"Neural config: device={ncfg.device} batched={int(getattr(ncfg, 'batched_restarts', True))} "f"K_max={ncfg.K_max} depth={ncfg.depth} width={ncfg.width} max_steps={ncfg.max_steps} "f"obs_tol={ncfg.obs_tol:g} dyn_tol={ncfg.dyn_tol:g} check_every={ncfg.check_every}")
        print(f"Upper bound on Adam steps: {total_steps:,}")

    # Decide parallelism.
    n_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if args.use_all_gpus:
        args.num_workers = max(1, n_gpus)

    if int(args.num_workers) == 0:
        args.num_workers = max(1, n_gpus) if n_gpus > 0 else 1

    num_workers = int(max(1, args.num_workers))

    t_global0 = time.time()

    if num_workers == 1:
        hyp_rows, succ_rows = _run_instances(
            instances_with_ids=instances_with_ids,
            ncfg=ncfg,
            newton_cfg=newton_cfg,
            K_list=K_list,
            eps_success=eps_success,
            progress_q=None,
            quiet=args.quiet,
        )
        df_h = pd.DataFrame(hyp_rows)
        df_s = pd.DataFrame(succ_rows)

    else:
        if not args.quiet:
            print(f"Parallel mode: num_workers={num_workers} | visible_gpus={n_gpus}")
            if n_gpus >= 2:
                print("Using multi-GPU instance-level parallelism (best-effort).")
            elif n_gpus == 1:
                print("Only one GPU visible; still running multiple processes on that GPU (may or may not help).")
            else:
                print("No GPU visible; running CPU multiprocessing (may or may not help).")

        # Spawn context is safer for CUDA.
        ctx = get_context("spawn")
        progress_q = ctx.Queue()

        # Split instances into roughly-equal contiguous chunks to keep I/O + progress simple.
        chunks: List[List[Tuple[int, Any]]] = []
        for w in range(num_workers):
            chunks.append([])
        for idx, item in enumerate(instances_with_ids):
            chunks[idx % num_workers].append(item)

        ncfg_dict = asdict(ncfg)
        newton_cfg_dict = asdict(newton_cfg)

        procs = []
        worker_h_paths = []
        worker_s_paths = []

        for w in range(num_workers):
            if n_gpus > 0:
                device_str = f"cuda:{w % n_gpus}"
            else:
                device_str = "cpu"

            out_h = os.path.join(args.outdir, f"_blockA_hypotheses_worker{w}.csv")
            out_s = os.path.join(args.outdir, f"_blockA_success_worker{w}.csv")
            worker_h_paths.append(out_h)
            worker_s_paths.append(out_s)

            p = ctx.Process(
                target=_worker_entry,
                kwargs=dict(
                    worker_id=w,
                    device_str=device_str,
                    instances_with_ids=chunks[w],
                    ncfg_dict=ncfg_dict,
                    newton_cfg_dict=newton_cfg_dict,
                    K_list=list(K_list),
                    eps_success=float(eps_success),
                    out_h_path=out_h,
                    out_s_path=out_s,
                    progress_q=progress_q,
                    quiet=True,
                ),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        # Progress reporting in the main process.
        tqdm = _optional_tqdm()
        pbar = None
        if (not args.quiet) and (tqdm is not None):
            pbar = tqdm(total=total_instances, desc="Block A instances", unit="inst")

        done_instances = 0
        done_workers = 0
        while done_workers < num_workers:
            msg = progress_q.get()
            if "inst_id" in msg:
                done_instances += 1
                if pbar is not None:
                    pbar.update(1)
                elif not args.quiet:
                    if done_instances == 1 or done_instances % 5 == 0 or done_instances == total_instances:
                        print(f"Completed {done_instances}/{total_instances} instances...")
            if "worker_done" in msg:
                done_workers += 1

        if pbar is not None:
            pbar.close()

        for p in procs:
            p.join()

        # Merge worker outputs.
        df_h_list = [pd.read_csv(pth) for pth in worker_h_paths if os.path.exists(pth)]
        df_s_list = [pd.read_csv(pth) for pth in worker_s_paths if os.path.exists(pth)]
        if not df_h_list or not df_s_list:
            raise RuntimeError("Parallel workers produced no output files; check logs for errors.")

        df_h = pd.concat(df_h_list, axis=0, ignore_index=True)
        df_s = pd.concat(df_s_list, axis=0, ignore_index=True)

        # Optional cleanup of worker shards (leave them if you want to inspect).
        for pth in worker_h_paths + worker_s_paths:
            try:
                os.remove(pth)
            except Exception:
                pass

    # Write outputs.
    df_h.to_csv(os.path.join(args.outdir, "blockA_hypotheses.csv"), index=False)
    df_s.to_csv(os.path.join(args.outdir, "blockA_success.csv"), index=False)

    # Aggregate *coverage* success probabilities.
    # Primary coverage success = symbolic itinerary match (d_H==0), consistent with the branch-counting metric.
    # Auxiliary coverage success = imputation threshold (diagnostic).
    df_summary = (
        df_s.groupby(["block", "solver", "A_name", "q", "sigma", "K"], as_index=False)
        .agg(
            P_success=("success", "mean"),
            P_success_impute=("success_impute", "mean"),
            P_success_single=("success_single", "mean"),
            P_success_single_impute=("success_single_impute", "mean"),
        )
    )
    df_summary["eps_success_impute"] = float(eps_success)
    df_summary.to_csv(os.path.join(args.outdir, "blockA_summary.csv"), index=False)

    # Convenience: write the per-instance *single-output* table (K=1 rows).
    df_single = df_s[df_s["K"] == 1].copy()
    if not df_single.empty:
        df_single.to_csv(os.path.join(args.outdir, "blockA_single.csv"), index=False)

        df_single_summary = (
            df_single.groupby(["block", "solver", "A_name", "q", "sigma"], as_index=False)
            .agg(
                P_single=("success_single", "mean"),
                P_single_impute=("success_single_impute", "mean"),
                median_single_error=("single_error", "median"),
                median_single_d_H=("single_d_H", "median"),
            )
        )
        df_single_summary["eps_success_impute"] = float(eps_success)
        df_single_summary.to_csv(os.path.join(args.outdir, "blockA_single_summary.csv"), index=False)

    # Save config and epsilon.
    meta = {
        "block": "A",
        "q_list": q_list,
        "sigma": sigma,
        "N_inst": args.N_inst,
        "K_max": args.K_max,
        "K_list": list(K_list),
        "eps_success_impute": eps_success,
        "coverage_success_definition": "symbolic (d_H==0) within top-K score-ranked hypotheses",
        "single_success_definition": "top-1 by solver score",
        "neural_config": asdict(ncfg),
        "newton_config": asdict(newton_cfg),
        "seed": args.seed,
        "num_workers": num_workers,
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "elapsed_seconds": float(time.time() - t_global0),
    }
    with open(os.path.join(args.outdir, "blockA_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not args.quiet:
        elapsed = float(time.time() - t_global0)
        print(f"Done. Wrote outputs to: {args.outdir} | elapsed={elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
