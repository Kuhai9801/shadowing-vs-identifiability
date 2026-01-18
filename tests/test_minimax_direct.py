import numpy as np

from data.generate import default_matrices
from systems.torus_map import (
    branch_solutions_y0,
    dT1,
    interior_x_separation,
    iterate_map,
    bq,
)


def test_minimax_triangle_inequality_adjacent_branches():
    """A minimal, deterministic unit test of the Theorem III.5 geometry.

    Construct two admissible strong-constraint truths that share the same observed endpoints
    but lie on adjacent branches. For an arbitrary single-output guess x_hat, triangle
    inequality implies max(err1, err2) >= delta/2.
    """

    A = default_matrices()["A1"]
    q = 6
    b_abs = abs(int(bq(A, q)))
    assert b_abs >= 2

    rng = np.random.default_rng(0)
    x0_obs = float(rng.random())
    xq_obs = float(rng.random())

    y0s = branch_solutions_y0(A, q, x0_obs, xq_obs)
    assert y0s.shape[0] == b_abs

    # Adjacent branches.
    y0_0 = float(y0s[0])
    y0_1 = float(y0s[1])

    traj0 = iterate_map(A, np.array([x0_obs, y0_0], dtype=float), q)
    traj1 = iterate_map(A, np.array([x0_obs, y0_1], dtype=float), q)
    x_qm1_0 = float(traj0[q - 1, 0])
    x_qm1_1 = float(traj1[q - 1, 0])

    delta = float(dT1(x_qm1_0, x_qm1_1))
    delta_pred = float(interior_x_separation(A, q))
    assert abs(delta - delta_pred) < 1e-10

    # Arbitrary estimator output.
    x_hat = 0.123456
    e0 = float(dT1(x_hat, x_qm1_0))
    e1 = float(dT1(x_hat, x_qm1_1))

    assert max(e0, e1) + 1e-15 >= 0.5 * delta
