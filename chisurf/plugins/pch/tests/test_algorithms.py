import numpy as np

from chisurf.plugins.pch.api.algorithms import (
    compute_p1,
    convolve_pch_numba,
    pch_mixture,
    pch_open_system,
    pch_single_species,
)


def test_compute_p1_basic():
    k_vals = np.arange(60, dtype=float)
    brightness = 5.0
    x_vals = np.linspace(0, 5, 500)
    dx = x_vals[1] - x_vals[0]
    p1 = compute_p1(k_vals, brightness, x_vals, dx)
    assert len(p1) == 60
    assert np.isclose(p1.sum(), 1.0, atol=1e-6)
    assert p1[0] <= 1


def test_pch_single_species():
    k_vals = np.arange(60, dtype=float)
    p = pch_single_species(k_vals, 3.0)
    assert len(p) == 60
    assert np.isclose(p.sum(), 1.0, atol=1e-6)


def test_convolve_pch_numba():
    p1 = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    result = convolve_pch_numba(p1, 2, 10)
    assert len(result) == 10
    assert np.isclose(result.sum(), 1.0, atol=1e-6)


def test_convolve_n0():
    p1 = np.array([0.5, 0.3, 0.2, 0.0, 0.0], dtype=float)
    result = convolve_pch_numba(p1, 0, 5)
    assert result[0] == 1.0
    assert result[1:].sum() == 0.0


def test_pch_open_system():
    k_vals = np.arange(60, dtype=float)
    p = pch_open_system(k_vals, 5.0, 2.0)
    assert len(p) == 60
    assert np.isclose(p.sum(), 1.0, atol=5e-3)


def test_pch_mixture():
    k_vals = np.arange(60, dtype=float)
    p = pch_mixture(k_vals, [3.0, 8.0], [2.0, 1.0])
    assert len(p) == 60
    assert np.isclose(p.sum(), 1.0, atol=5e-3)
