import numpy as np

from data.generate import default_matrices
from systems.torus_map import splitting_data


def test_splitting_constant_finite():
    mats = default_matrices()
    for A in mats.values():
        sd = splitting_data(A)
        assert sd.C > 0.0
        assert sd.norm_Pu > 0.0
        assert sd.norm_Ps > 0.0
        assert 0.0 < sd.angle < np.pi


def test_projection_norm_identity():
    """Check ||P_u|| = ||P_s|| = 1/|sin θ| as required by the Methods."""
    mats = default_matrices()
    for A in mats.values():
        sd = splitting_data(A)
        formula = 1.0 / abs(np.sin(sd.angle))
        assert abs(sd.norm_Pu - formula) <= 1e-9 * formula
        assert abs(sd.norm_Ps - formula) <= 1e-9 * formula
        assert abs(sd.norm_Pu - sd.norm_Ps) <= 1e-12
