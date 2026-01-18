import numpy as np

from systems.torus_map import Aq_entries, bq, check_sl2z_hyperbolic, matrix_power_int
from data.generate import default_matrices


def test_matrix_power_identity():
    A = default_matrices()["A1"]
    I = matrix_power_int(A, 0)
    assert int(I[0, 0]) == 1 and int(I[1, 1]) == 1
    assert int(I[0, 1]) == 0 and int(I[1, 0]) == 0


def test_bq_nonzero_default_matrices():
    mats = default_matrices()
    for A in mats.values():
        check_sl2z_hyperbolic(A)
        for q in range(1, 25):
            assert bq(A, q) != 0


def test_Aq_recursion_small():
    A = default_matrices()["A1"]
    for q in range(0, 10):
        Mq = matrix_power_int(A, q)
        Mq1 = matrix_power_int(A, q + 1)
        assert np.all(Mq1 == A.astype(object) @ Mq)
