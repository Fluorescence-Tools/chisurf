"""Rate-matrix kinetics, multi-dT scan, and global MEM."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fcs.flc_2d.api import (
    global_lifetime_mem,
    rate_matrix_kinetics,
    species_correlation,
    species_decay_patterns,
    two_d_fdc,
    two_d_fdc_scan,
)
from chisurf.plugins.fcs.flc_2d.fit.ilt import build_exp_basis, lifetime_grid
from chisurf.plugins.fcs.flc_2d.fit.kinetics import fit_rate_matrix


def test_fit_rate_matrix_synthetic_two_state():
    """Recover a known relaxation rate and rate matrix from synthetic curves."""
    lag = np.geomspace(1e-4, 1.0, 200)
    k = 40.0  # k12 + k21
    auto = 1.0 + 2.0 * np.exp(-k * lag)
    cross = 1.0 - 1.5 * np.exp(-k * lag)
    curves = {(0, 0): auto, (1, 1): auto * 0.8, (0, 1): cross}
    res = fit_rate_matrix(
        lag, curves, n_states=2, populations=np.array([0.25, 0.75]), t_min=1e-4, t_max=1.0
    )
    assert abs(res.relaxation_rates[0] - k) / k < 0.1
    # k12 = lambda * p_long = 40 * 0.75 = 30 ; k21 = 40 * 0.25 = 10
    assert res.rate_matrix is not None
    assert abs(res.rate_matrix[0, 1] - 30.0) < 4.0
    assert abs(res.rate_matrix[1, 0] - 10.0) < 4.0


def test_global_mem_synthetic_runs():
    """Global MEM jointly inverts several lag matrices and resolves the lifetime band."""
    time_ns = np.linspace(0, 16, 50)
    tau = lifetime_grid(0.3, 8.0, 18)
    E = build_exp_basis(time_ns, tau, normalize=True)
    # two species at 1 & 3 ns, with a lag-dependent cross term
    idx1 = int(np.argmin(np.abs(tau - 1.0)))
    idx3 = int(np.argmin(np.abs(tau - 3.0)))
    mats = []
    for cross in (0.05, 0.15, 0.25):
        P = np.zeros((tau.size, tau.size))
        P[idx1, idx1] = 1.0
        P[idx3, idx3] = 0.6
        P[idx1, idx3] = P[idx3, idx1] = cross
        mats.append(E @ P @ E.T * 1e4 + 50.0)
    res = global_lifetime_mem(mats, time_ns, n_states=2, n_components=18, regulator=1.0)
    assert res.amplitudes.shape == (18, 2)
    assert res.correlations.shape == (3, 2, 2)
    band = (tau >= 0.7) & (tau <= 3.5)
    assert res.marginal[band].sum() > 0.5 * res.marginal.sum()


@pytest.mark.slow
def test_multi_dt_scan_matches_single_builds(reference_photons):
    """One single-pass scan reproduces per-lag single builds exactly."""
    macro = reference_photons["macro_ticks"]
    micro = reference_photons["micro_ticks"]
    lags = [1000, 25000]
    scan = two_d_fdc_scan(macro, micro, lags, ddT=2000, tMin=1, tMax=3127, logt_imax=40)
    for i, L in enumerate(lags):
        single = two_d_fdc(
            macro, micro, dT=L, ddT=2000, tMin=1, tMax=3127, logt_imax=40, build_lin=False
        )
        assert scan["matrices"][i].sum() == single["mat_log"].sum()


@pytest.mark.slow
def test_rate_matrix_reference(reference_photons):
    """Rate-matrix fit on the reference recovers ~40/s (25 ms) and K ~ [[0,30],[10,0]]."""
    rp = reference_photons
    patterns = species_decay_patterns(
        (1.0, 3.0),
        rp["n_microtime_bins"],
        rp["micro_resolution_ns"],
        irf=rp["irf"],
        irf_time_ns=rp["irf_time_ns"],
    )
    dyn = species_correlation(
        rp["macro_ticks"],
        rp["micro_ticks"],
        patterns,
        None,
        rp["macro_resolution_s"],
        n_microtime_bins=rp["n_microtime_bins"],
        n_casc=25,
    )
    # equilibrium populations of K=[[0,30],[10,0]] are [0.25, 0.75] (short, long)
    rm = rate_matrix_kinetics(dyn.correlation, n_states=2, populations=[0.25, 0.75])
    assert 25.0 <= rm.relaxation_rates[0] <= 60.0
    assert rm.rate_matrix is not None
    assert 20.0 <= rm.rate_matrix[0, 1] <= 40.0  # k12 ~ 30
    assert 5.0 <= rm.rate_matrix[1, 0] <= 16.0  # k21 ~ 10
