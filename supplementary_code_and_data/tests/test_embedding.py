import numpy as np

from systems.torus_map import E, E_x, frac


def test_embedding_periodic_invariance():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 2)) * 5.0
    Ez = E(z)
    Ezm = E(frac(z))
    assert np.max(np.abs(Ez - Ezm)) < 1e-10


def test_embedding_shapes():
    z = np.array([[0.0, 0.0], [0.25, 0.5]])
    Ez = E(z)
    assert Ez.shape == (2, 4)
    x = np.array([0.0, 0.25])
    Ex = E_x(x)
    assert Ex.shape == (2, 2)
