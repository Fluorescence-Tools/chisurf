"""Gradient-based 2D MEM solver: convergence and recovery on synthetic data."""

from __future__ import annotations

import numpy as np

from chisurf.plugins.fcs.flc_2d.fit.ilt import build_exp_basis, lifetime_grid
from chisurf.plugins.fcs.flc_2d.fit.mem_2d import solve_mem_2d


def _synthetic_2d(tau_grid, basis, peaks_ns, amps):
    """Build M = E P E.T with delta-like P at the requested lifetimes."""
    P = np.zeros((tau_grid.size, tau_grid.size))
    idx = [int(np.argmin(np.abs(tau_grid - p))) for p in peaks_ns]
    for a, i in zip(amps, idx):
        P[i, i] = a
    return basis @ P @ basis.T, P


def test_mem_2d_recovers_synthetic_spectrum():
    time_ns = np.linspace(0, 16, 60)
    tau = lifetime_grid(0.3, 8.0, 24)
    E = build_exp_basis(time_ns, tau, normalize=True)
    M, _P = _synthetic_2d(tau, E, peaks_ns=(1.0, 3.0), amps=(1.0e4, 6.0e3))

    res = solve_mem_2d(M, E, tau, regulator=1.0, n_outer=4, fit_offset=False)

    assert np.all(res.spectrum >= 0)
    assert np.isfinite(res.chi2)
    # the recovered marginal must concentrate around the two seeded lifetimes
    peaks = np.sort(res.peak_lifetimes(2))
    assert peaks.size >= 1
    band = (tau >= 0.7) & (tau <= 3.5)
    assert res.marginal[band].sum() > 0.6 * res.marginal.sum()
