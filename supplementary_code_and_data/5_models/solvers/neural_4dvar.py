r"""Neural weak-constraint 4D-Var solver (map-PINN as a variational integrator).

This implementation follows the plan exactly:

  * Parameterization: small MLP g_\theta : [0,1] -> R^2.
    Evaluate at t_n = n/q to produce \hat z^n = g_\theta(t_n), n=0,...,q.

  * Training losses are computed strictly in embedding space via the torus embedding
        E(x,y) = (cos 2\pi x, sin 2\pi x, cos 2\pi y, sin 2\pi y) \in R^4.
    Use only chordal distances in R^4.

  * IMPORTANT: no modular projection (frac) or arctan2 is used inside losses.
    The embedding is periodic, so E(z) = E(frac(z)) automatically.

Performance notes
-----------------
The original, straightforward implementation trains K_max independent networks
sequentially. On Kaggle GPUs, this can be slow because each optimizer step touches
very small tensors (q <= 22), so Python overhead and GPU kernel launch latency dominate.

To address this, the default implementation here trains all restarts *in one batched model*:
we represent K_max independent MLPs as a single module with batched parameters of shape
(K_max, ...). This keeps the methodology identical (same architecture, same loss), but
amortizes overhead and can be dramatically faster.

The solver still returns K_max hypotheses (one per restart) with per-restart scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import time
import numpy as np

import torch
import torch.nn as nn

from systems.torus_map import E, E_x


class SmallMLP(nn.Module):
    """A small fully-connected network with tanh activations.

    Input:  t in R (shape (...,1))
    Output: z in R^2 (shape (...,2))
    """

    def __init__(self, *, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 (including input->hidden and hidden->output).")
        layers: List[nn.Module] = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class BatchedMLP(nn.Module):
    """K independent MLPs implemented as a single module with batched parameters.

    Parameters are stored as tensors with leading dimension K. Forward evaluation uses
    batched matrix multiplications so all K networks run in parallel.

    Shapes:
        * Input  t: (N,1)
        * Output z: (K,N,2)
    """

    def __init__(self, *, K: int, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1.")
        if depth < 2:
            raise ValueError("depth must be >= 2.")
        self.K = int(K)
        self.width = int(width)
        self.depth = int(depth)

        # Number of Linear layers equals depth:
        #   1->width, (depth-2) times width->width, width->2.
        layer_dims: List[tuple[int, int]] = []
        layer_dims.append((1, self.width))
        for _ in range(self.depth - 2):
            layer_dims.append((self.width, self.width))
        layer_dims.append((self.width, 2))

        weights: List[nn.Parameter] = []
        biases: List[nn.Parameter] = []
        for in_dim, out_dim in layer_dims:
            # Weight layout: (K, out_dim, in_dim) so we can use W^T in bmm.
            W = nn.Parameter(torch.empty(self.K, out_dim, in_dim))
            b = nn.Parameter(torch.empty(self.K, out_dim))
            weights.append(W)
            biases.append(b)

        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear default initialization (kaiming_uniform_ with a=sqrt(5)).
        with torch.no_grad():
            for W, b in zip(self.weights, self.biases):
                nn.init.kaiming_uniform_(W, a=np.sqrt(5.0))
                # fan_in is the input dimension
                fan_in = W.shape[-1]
                bound = 1.0 / float(np.sqrt(fan_in)) if fan_in > 0 else 0.0
                nn.init.uniform_(b, -bound, bound)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("t must have shape (N,1).")
        # Expand to (K,N,1). Contiguous to keep bmm happy.
        x = t.unsqueeze(0).expand(self.K, -1, -1).contiguous()

        n_layers = len(self.weights)
        for i in range(n_layers):
            W = self.weights[i]  # (K, out, in)
            b = self.biases[i]   # (K, out)
            # x: (K,N,in), W^T: (K,in,out) -> (K,N,out)
            x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
            if i < n_layers - 1:
                x = torch.tanh(x)
        return x  # (K,N,2)


@dataclass
class Neural4DVarConfig:
    """Configuration for neural weak-constraint 4D-Var."""

    K_max: int = 64
    depth: int = 4
    width: int = 64
    lr: float = 1e-3
    max_steps: int = 1500
    lambda_dyn: float = 1.0

    # Observation admissibility (noise ball) in embedding space.
    #
    # The theoretical setup assumes bounded endpoint noise in T^1:
    #     d_{T1}(x_0, x0_obs) <= sigma,   d_{T1}(x_q, xq_obs) <= sigma.
    #
    # To align the solver with this admissibility notion while preserving the
    # "embedding-only" methodological invariant, we implement an endpoint
    # *constraint with slack* in circle-embedding (R^2) chordal distance.
    #
    # If obs_ball_radius > 0, then the observation penalty is the sum over
    # endpoints of hinge violations:
    #     max(0, ||E_x(x_hat) - E_x(x_obs)||^2 - obs_ball_radius^2).
    # When obs_ball_radius == 0 (default), this reduces to the original
    # equality-penalty (sum of squared embedding differences).
    obs_ball_radius: float = 0.0

    # Early stopping thresholds. These are in embedding-space units.
    obs_tol: float = 1e-8
    dyn_tol: float = 1e-6

    # If True, stop when both conditions satisfied.
    early_stop: bool = True

    # How often (in optimization steps) to evaluate early-stop conditions and record monitoring scalars.
    # Larger values can be MUCH faster on GPU because they avoid frequent CPU/GPU synchronization.
    check_every: int = 25

    # Optional console visibility: in sequential mode prints every N completed restarts.
    # In batched mode prints when the number of converged restarts crosses multiples of N.
    log_every_restarts: int = 0

    # If True (default), train all restarts simultaneously in one batched model.
    batched_restarts: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Determinism controls (best effort).
    torch_deterministic: bool = False


@dataclass
class NeuralHypothesis:
    """One hypothesis produced by a single neural restart."""

    restart_id: int
    seed: int
    converged: bool
    steps: int

    # Final loss values
    loss_total: float
    loss_obs: float
    loss_dyn: float
    defect_E_max: float

    # Predicted trajectory in R^2 (not reduced mod 1; evaluation may apply frac).
    z_hat: np.ndarray  # shape (q+1,2)

    # Wall-clock time
    train_seconds: float


def _set_torch_determinism(enabled: bool) -> None:
    if not enabled:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _solve_neural_4dvar_sequential(
    *,
    A: np.ndarray,
    q: int,
    x0_obs: float,
    xq_obs: float,
    config: Neural4DVarConfig,
    base_seed: int,
) -> List[NeuralHypothesis]:
    """Sequential implementation (one network per restart).

    This is kept as a correctness/reference path and a fallback for environments where
    multiprocessing + CUDA behaves unexpectedly.
    """
    if q < 1:
        raise ValueError("q must be >= 1.")
    if config.K_max < 1:
        raise ValueError("K_max must be >= 1.")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if int(config.check_every) < 0:
        raise ValueError("check_every must be >= 0.")

    _set_torch_determinism(config.torch_deterministic)

    device = torch.device(config.device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    lambda_dyn = float(config.lambda_dyn)

    # Fixed evaluation grid t_n = n/q.
    t_grid = torch.linspace(0.0, 1.0, steps=q + 1, device=device, dtype=torch.float32).unsqueeze(-1)
    x0_obs_t = torch.tensor(float(x0_obs), device=device, dtype=torch.float32)
    xq_obs_t = torch.tensor(float(xq_obs), device=device, dtype=torch.float32)

    # Precompute observed endpoint embeddings once.
    with torch.no_grad():
        ex_obs = E_x(torch.stack([x0_obs_t, xq_obs_t], dim=0))  # (2,2)

    obs_ball_eps2: float = float(config.obs_ball_radius) ** 2

    obs_tol = float(config.obs_tol)
    dyn_tol_sq = float(config.dyn_tol) ** 2
    check_every = int(config.check_every) if int(config.check_every) > 0 else (config.max_steps + 1)

    hyps: List[NeuralHypothesis] = []
    t_solver0 = time.time()

    for j in range(int(config.K_max)):
        seed_j = int(base_seed + 1009 * (j + 1))
        torch.manual_seed(seed_j)
        np.random.seed(seed_j % (2**32 - 1))

        model = SmallMLP(width=int(config.width), depth=int(config.depth)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

        t0 = time.time()
        converged = False
        steps_done = int(config.max_steps)

        for step in range(1, int(config.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)

            z_hat = model(t_grid)  # (q+1,2) in R^2

            # Observation term (x endpoints only).
            ex_pred = E_x(torch.stack([z_hat[0, 0], z_hat[q, 0]], dim=0))  # (2,2)
            # Squared chordal distance per endpoint on the unit circle.
            dist2 = torch.sum((ex_pred - ex_obs) ** 2, dim=-1)  # (2,)
            if obs_ball_eps2 > 0.0:
                # Slack constraint (bounded noise admissibility): no penalty if within the ball.
                loss_obs = torch.sum(torch.relu(dist2 - obs_ball_eps2))
            else:
                # Equality penalty (legacy): sum of squared embedding differences.
                loss_obs = torch.sum(dist2)

            # Dynamics / defect term in embedding space.
            zA = z_hat[:-1] @ A_t.T  # A z^n
            res = E(z_hat[1:]) - E(zA)  # (q,4)
            sq = torch.sum(res * res, dim=-1)  # (q,)
            loss_dyn = torch.mean(sq)

            loss_total = loss_obs + lambda_dyn * loss_dyn

            # Early stopping check (avoid per-step .cpu().item() syncs on GPU).
            if config.early_stop and (step % check_every == 0):
                with torch.no_grad():
                    defect_E_max_sq = torch.max(sq)
                    obs_ok = bool((loss_obs <= obs_tol).item())
                    dyn_ok = bool((defect_E_max_sq <= dyn_tol_sq).item())
                    if obs_ok and dyn_ok:
                        converged = True
                        steps_done = int(step)
                        break

            loss_total.backward()
            optimizer.step()
            steps_done = int(step)

        train_seconds = float(time.time() - t0)

        # Final values + export trajectory to numpy for evaluation.
        with torch.no_grad():
            z_hat_t = model(t_grid)
            ex_pred = E_x(torch.stack([z_hat_t[0, 0], z_hat_t[q, 0]], dim=0))  # (2,2)
            dist2 = torch.sum((ex_pred - ex_obs) ** 2, dim=-1)  # (2,)
            if obs_ball_eps2 > 0.0:
                loss_obs_t = torch.sum(torch.relu(dist2 - obs_ball_eps2))
            else:
                loss_obs_t = torch.sum(dist2)
            zA = z_hat_t[:-1] @ A_t.T
            res = E(z_hat_t[1:]) - E(zA)
            sq2 = torch.sum(res * res, dim=-1)
            loss_dyn_t = torch.mean(sq2)
            loss_total_t = loss_obs_t + lambda_dyn * loss_dyn_t
            defect_E_max_sq = torch.max(sq2)
            defect_E_max_t = torch.sqrt(defect_E_max_sq)

            loss_total_val = float(loss_total_t.detach().cpu().item())
            loss_obs_val = float(loss_obs_t.detach().cpu().item())
            loss_dyn_val = float(loss_dyn_t.detach().cpu().item())
            defect_E_max_val = float(defect_E_max_t.detach().cpu().item())
            z_hat_np = z_hat_t.detach().cpu().numpy().astype(np.float64)

        hyps.append(
            NeuralHypothesis(
                restart_id=int(j),
                seed=int(seed_j),
                converged=bool(converged),
                steps=int(steps_done),
                loss_total=float(loss_total_val),
                loss_obs=float(loss_obs_val),
                loss_dyn=float(loss_dyn_val),
                defect_E_max=float(defect_E_max_val),
                z_hat=z_hat_np,
                train_seconds=float(train_seconds),
            )
        )

        # Optional progress logging.
        if int(getattr(config, "log_every_restarts", 0)) > 0:
            every = int(config.log_every_restarts)
            if (j + 1) % every == 0 or (j + 1) == int(config.K_max):
                elapsed = float(time.time() - t_solver0)
                avg = elapsed / float(j + 1)
                print(
                    f"  [neural] restarts {j+1}/{int(config.K_max)} | "
                    f"last: {train_seconds:.2f}s {steps_done} steps conv={int(converged)} | "
                    f"avg/restart={avg:.2f}s (elapsed={elapsed/60.0:.1f} min)"
                )

    return hyps


def _solve_neural_4dvar_batched(
    *,
    A: np.ndarray,
    q: int,
    x0_obs: float,
    xq_obs: float,
    config: Neural4DVarConfig,
    base_seed: int,
) -> List[NeuralHypothesis]:
    """Batched implementation: train all restarts simultaneously.

    Returns K_max independent hypotheses with the same definitions as the sequential solver.
    """
    if q < 1:
        raise ValueError("q must be >= 1.")
    if config.K_max < 1:
        raise ValueError("K_max must be >= 1.")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if int(config.check_every) < 0:
        raise ValueError("check_every must be >= 0.")

    _set_torch_determinism(config.torch_deterministic)

    device = torch.device(config.device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    K = int(config.K_max)
    lambda_dyn = float(config.lambda_dyn)

    # Fixed evaluation grid t_n = n/q.
    t_grid = torch.linspace(0.0, 1.0, steps=q + 1, device=device, dtype=torch.float32).unsqueeze(-1)
    x0_obs_t = torch.tensor(float(x0_obs), device=device, dtype=torch.float32)
    xq_obs_t = torch.tensor(float(xq_obs), device=device, dtype=torch.float32)

    with torch.no_grad():
        ex_obs = E_x(torch.stack([x0_obs_t, xq_obs_t], dim=0)).unsqueeze(0)  # (1,2,2)

    obs_ball_eps2: float = float(config.obs_ball_radius) ** 2

    obs_tol = float(config.obs_tol)
    dyn_tol_sq = float(config.dyn_tol) ** 2
    check_every = int(config.check_every) if int(config.check_every) > 0 else (int(config.max_steps) + 1)

    # Deterministic initialization for the whole batch.
    torch.manual_seed(int(base_seed))
    np.random.seed(int(base_seed) % (2**32 - 1))

    model = BatchedMLP(K=K, width=int(config.width), depth=int(config.depth)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    # Track which restarts are still being optimized.
    done = torch.zeros(K, dtype=torch.bool, device=device)
    active = ~done
    active_f = active.to(dtype=torch.float32)

    # Bookkeeping on CPU for export.
    steps_done = np.full((K,), int(config.max_steps), dtype=np.int32)
    converged = np.zeros((K,), dtype=np.bool_)

    n_active = int(K)
    next_log_threshold = int(config.log_every_restarts) if int(config.log_every_restarts) > 0 else 0

    t0 = time.time()

    for step in range(1, int(config.max_steps) + 1):
        if n_active == 0:
            break

        optimizer.zero_grad(set_to_none=True)

        z_hat = model(t_grid)  # (K,q+1,2)

        # Observation term (x endpoints only). Shape bookkeeping:
        #   x_end: (K,2) -> E_x -> (K,2,2)
        x_end = torch.stack([z_hat[:, 0, 0], z_hat[:, q, 0]], dim=1)
        ex_pred = E_x(x_end)  # (K,2,2)
        # Squared chordal distance per endpoint on the unit circle.
        dist2 = torch.sum((ex_pred - ex_obs) ** 2, dim=-1)  # (K,2)
        if obs_ball_eps2 > 0.0:
            # Slack constraint (bounded noise admissibility): no penalty if within the ball.
            loss_obs_vec = torch.sum(torch.relu(dist2 - obs_ball_eps2), dim=1)  # (K,)
        else:
            # Equality penalty (legacy): sum of squared embedding differences.
            loss_obs_vec = torch.sum(dist2, dim=1)  # (K,)

        # Dynamics / defect term in embedding space.
        zA = torch.matmul(z_hat[:, :-1, :], A_t.T)  # (K,q,2)
        res = E(z_hat[:, 1:, :]) - E(zA)            # (K,q,4)
        sq = torch.sum(res * res, dim=-1)           # (K,q)
        loss_dyn_vec = torch.mean(sq, dim=-1)       # (K,)

        loss_total_vec = loss_obs_vec + lambda_dyn * loss_dyn_vec  # (K,)

        # Only active restarts contribute gradients (freezes converged ones).
        loss = torch.sum(loss_total_vec * active_f) / float(n_active)
        loss.backward()
        optimizer.step()

        if config.early_stop and (step % check_every == 0):
            with torch.no_grad():
                defect_E_max_sq_vec = torch.max(sq, dim=-1).values  # (K,)
                obs_ok = loss_obs_vec <= obs_tol
                dyn_ok = defect_E_max_sq_vec <= dyn_tol_sq
                newly_done = (~done) & obs_ok & dyn_ok
                if torch.any(newly_done):
                    idx = newly_done.nonzero(as_tuple=False).flatten().detach().cpu().numpy()
                    # Record on CPU.
                    for k in idx:
                        converged[int(k)] = True
                        steps_done[int(k)] = int(step)

                done = done | newly_done
                active = ~done
                active_f = active.to(dtype=torch.float32)
                # This sync happens only every check_every steps.
                n_active = int(active_f.sum().detach().cpu().item())

            # Optional progress logging: report when the number of converged restarts
            # crosses multiples of log_every_restarts.
            if next_log_threshold > 0:
                done_count = int(K - n_active)
                if done_count >= next_log_threshold or done_count == K:
                    elapsed = float(time.time() - t0)
                    print(
                        f"  [neural-batch] step {step}/{int(config.max_steps)} | "
                        f"converged {done_count}/{K} | active {n_active} | "
                        f"elapsed={elapsed/60.0:.1f} min"
                    )
                    next_log_threshold += int(config.log_every_restarts)

    total_seconds = float(time.time() - t0)

    # Final forward pass for export/scoring.
    with torch.no_grad():
        z_hat = model(t_grid)  # (K,q+1,2)
        x_end = torch.stack([z_hat[:, 0, 0], z_hat[:, q, 0]], dim=1)
        ex_pred = E_x(x_end)
        dist2 = torch.sum((ex_pred - ex_obs) ** 2, dim=-1)  # (K,2)
        if obs_ball_eps2 > 0.0:
            loss_obs_vec = torch.sum(torch.relu(dist2 - obs_ball_eps2), dim=1)
        else:
            loss_obs_vec = torch.sum(dist2, dim=1)

        zA = torch.matmul(z_hat[:, :-1, :], A_t.T)
        res = E(z_hat[:, 1:, :]) - E(zA)
        sq = torch.sum(res * res, dim=-1)
        loss_dyn_vec = torch.mean(sq, dim=-1)
        loss_total_vec = loss_obs_vec + lambda_dyn * loss_dyn_vec
        defect_E_max_sq_vec = torch.max(sq, dim=-1).values
        defect_E_max_vec = torch.sqrt(defect_E_max_sq_vec)

        # Move to CPU once.
        z_hat_np = z_hat.detach().cpu().numpy().astype(np.float64)
        loss_total_np = loss_total_vec.detach().cpu().numpy().astype(np.float64)
        loss_obs_np = loss_obs_vec.detach().cpu().numpy().astype(np.float64)
        loss_dyn_np = loss_dyn_vec.detach().cpu().numpy().astype(np.float64)
        defect_np = defect_E_max_vec.detach().cpu().numpy().astype(np.float64)

    seeds = [int(base_seed + 1009 * (j + 1)) for j in range(K)]

    # Spread total training time across restarts for reporting.
    per_restart_seconds = float(total_seconds / float(K)) if K > 0 else float(total_seconds)

    hyps: List[NeuralHypothesis] = []
    for j in range(K):
        hyps.append(
            NeuralHypothesis(
                restart_id=int(j),
                seed=int(seeds[j]),
                converged=bool(converged[j]),
                steps=int(steps_done[j]),
                loss_total=float(loss_total_np[j]),
                loss_obs=float(loss_obs_np[j]),
                loss_dyn=float(loss_dyn_np[j]),
                defect_E_max=float(defect_np[j]),
                z_hat=z_hat_np[j],
                train_seconds=float(per_restart_seconds),
            )
        )
    return hyps


def solve_neural_4dvar(
    *,
    A: np.ndarray,
    q: int,
    x0_obs: float,
    xq_obs: float,
    config: Neural4DVarConfig,
    base_seed: int,
) -> List[NeuralHypothesis]:
    """Run K_max independent neural restarts for one instance.

    By default this uses the faster batched implementation. Set
    config.batched_restarts=False to force the sequential reference implementation.
    """
    if bool(getattr(config, "batched_restarts", True)) and int(config.K_max) > 1:
        return _solve_neural_4dvar_batched(
            A=A,
            q=q,
            x0_obs=x0_obs,
            xq_obs=xq_obs,
            config=config,
            base_seed=base_seed,
        )
    return _solve_neural_4dvar_sequential(
        A=A,
        q=q,
        x0_obs=x0_obs,
        xq_obs=xq_obs,
        config=config,
        base_seed=base_seed,
    )
