import numpy as np

from systems.torus_map import dT1, dT2, frac


def test_dT1_basic():
    assert float(dT1(0.0, 0.0)) == 0.0
    # 0.75 is distance 0.25 from 0 on T^1.
    assert abs(float(dT1(0.0, 0.75)) - 0.25) < 1e-12
    assert abs(float(dT1(0.75, 0.0)) - 0.25) < 1e-12


def test_frac_range():
    x = np.array([-0.2, 1.7, 2.0, 3.4])
    y = frac(x)
    assert np.all(y >= 0.0)
    assert np.all(y < 1.0)


def test_dT2_basic():
    z = np.array([0.1, 0.9])
    w = np.array([0.9, 0.1])
    # Differences are -0.8 and 0.8, which wrap to 0.2 and -0.2.
    expected = np.sqrt(0.2**2 + 0.2**2)
    assert abs(float(dT2(z, w)) - expected) < 1e-12
