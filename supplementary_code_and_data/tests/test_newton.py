import numpy as np

from data.generate import default_matrices, generate_instance
from solvers.newton_shooting import NewtonConfig, solve_newton_shooting
from systems.torus_map import dT1


def test_newton_finds_valid_endpoint_solution():
    A = default_matrices()["A1"]
    inst = generate_instance(A_name="A1", A=A, q=8, sigma=0.0, seed=123)
    cfg = NewtonConfig(K_max=8, max_iter=100, tol=1e-10)
    hyps = solve_newton_shooting(A=A, q=inst.q, x0=inst.x0_obs, xq=inst.xq_obs, config=cfg, base_seed=999)
    # At least one should converge and match the xq endpoint.
    errs = [float(dT1(h.traj[inst.q, 0], inst.xq_obs)) for h in hyps]
    assert min(errs) < 1e-8
