"""Tests for the MATLAB feature ports: simulator, 1D-FDC, 1D-MEM, Gaussian, kinetics.

These are self-contained (no reference ``.mat`` needed): the event-driven simulator
generates a ground-truth photon stream and the analysis recovers the known parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fcs.flc_2d import api
from chisurf.plugins.fcs.flc_2d.fit.gaussian import fit_gaussian_multi
from chisurf.plugins.fcs.flc_2d.fit.helpers import histogram_1d
from chisurf.plugins.fcs.flc_2d.fit.ilt import build_exp_basis, ilt_1d, lifetime_grid
from chisurf.plugins.fcs.flc_2d.fit.kinetics import (
    equilibrium_populations,
    make_generator_matrix,
)
from chisurf.plugins.fcs.flc_2d.fit.mem_1d import mi_prior, solve_mem_1d

K_TWO_STATE = np.array([[0.0, 30.0], [10.0, 0.0]])  # eq pops [0.25, 0.75]


# --------------------------------------------------------------------------- kinetics


def test_generator_matrix_columns_sum_to_zero():
    G = make_generator_matrix(K_TWO_STATE)
    # master-equation generator conserves probability: columns sum to 0
    assert np.allclose(G.sum(axis=0), 0.0)


def test_equilibrium_populations_two_state():
    p = equilibrium_populations(K_TWO_STATE)
    # detailed balance: p_short / p_long = k21 / k12 = 10/30 -> [0.25, 0.75]
    assert np.allclose(p, [0.25, 0.75], atol=1e-6)


# -------------------------------------------------------------------------- simulator


@pytest.fixture(scope="module")
def sim_stream():
    return api.simulate_stream(
        K_TWO_STATE, [1.0, 3.0], [20000.0, 20000.0], total_time_s=40.0, seed=3
    )


def test_simulator_macro_ascending_and_counts(sim_stream):
    assert sim_stream.macro_times.size > 100_000
    assert np.all(np.diff(sim_stream.macro_times) >= 0)


def test_simulator_state_lifetimes_and_populations(sim_stream):
    s = sim_stream
    assert np.allclose(s.equilibrium_populations, [0.25, 0.75], atol=1e-6)
    # per-state mean micro-time matches the state lifetime (mono-exponential mean = tau)
    for st, tau in ((0, 1.0), (1, 3.0)):
        sel = s.states == st
        mean_ns = s.micro_times[sel].mean() * s.micro_time_resolution_ns
        assert mean_ns == pytest.approx(tau, abs=0.15)


# ---------------------------------------------------------------------------- 1D-FDC


def test_one_d_fdc_matches_microtime_histogram(sim_stream):
    s = sim_stream
    out = api.one_d_fdc(s.macro_times, s.micro_times, tMin=0, tMax=3127, lint_bin_factor=8)
    fdc = out["lin"].astype(float)
    # the zero-lag diagonal is dominated by self-coincidence == the micro-time histogram
    hist, _ = np.histogram(s.micro_times, bins=np.arange(0, 8 * fdc.size + 1, 8))
    n = min(fdc.size, hist.size)
    # high correlation confirms the 1D-FDC is the fluorescence decay; it is not exactly
    # equal because the kernel bins micro-times with ceil() vs histogram's floor() (a
    # half-bin shift) and adds the rare same-tick cross-coincidences.
    corr = np.corrcoef(fdc[1:n], hist[1:n])[0, 1]  # skip empty leading bin
    assert corr > 0.98


def test_one_d_fdc_nnls_recovers_lifetimes(sim_stream):
    s = sim_stream
    out = api.one_d_fdc(s.macro_times, s.micro_times, tMin=0, tMax=3127, max_bins=400)
    decay = out["lin"].astype(float)
    t_ns = out["lin_t"].astype(float) * 0.004
    # drop empty leading bin (the MATLAB FitStartI)
    start = int(np.flatnonzero(decay > 0)[0])
    decay, t_ns = decay[start:], t_ns[start:]
    tau = lifetime_grid(0.3, 8.0, 40)
    basis = build_exp_basis(t_ns, tau)
    res = ilt_1d(decay, basis, tau, method="nnls")
    peaks = np.sort(res.peak_lifetimes(2))
    assert peaks.size == 2
    assert peaks[0] == pytest.approx(1.0, abs=0.4)
    assert peaks[1] == pytest.approx(3.0, abs=0.5)


# ----------------------------------------------------------------------------- 1D-MEM


def test_mi_prior_types_positive_and_shaped():
    tau = lifetime_grid(0.3, 8.0, 30)
    A = np.exp(-((np.log(tau) - np.log(2.0)) ** 2))
    for mi_type in (0, 1, 2, 3):
        mi = mi_prior(A, tau, t_min=0.0, t_max=12.5, mi_type=mi_type)
        assert mi.shape == tau.shape
        assert np.all(mi > 0)


def test_mem_1d_recovers_biexponential():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 12.5, 400)
    truth = 0.4 * np.exp(-t / 1.0) + 0.6 * np.exp(-t / 3.0)
    counts = rng.poisson(truth / truth.max() * 1e5).astype(float)
    tau = lifetime_grid(0.3, 8.0, 40)
    basis = build_exp_basis(t, tau)
    res = solve_mem_1d(counts, basis, tau, reg=200.0, n_outer=15, t_min=0.0, t_max=12.5)
    # the fit should reproduce the decay well (no overflow, converged)
    assert np.isfinite(res.chi2)
    assert res.chi2 < 0.5
    # most of the weight sits between the two true lifetimes
    centroid = float((tau * res.amplitudes).sum() / res.amplitudes.sum())
    assert 0.8 < centroid < 3.5


# --------------------------------------------------------------------------- Gaussian


def test_gaussian_multi_recovers_two_peaks():
    x = np.linspace(0.3, 8.0, 200)
    y = 1.0 * np.exp(-(((x - 1.0) / 0.2) ** 2)) + 0.7 * np.exp(-(((x - 3.0) / 0.3) ** 2))
    res = fit_gaussian_multi(x, y, 2)
    centers = res.centers
    assert centers[0] == pytest.approx(1.0, abs=0.05)
    assert centers[1] == pytest.approx(3.0, abs=0.05)
    assert res.sse < 1e-6


def test_gaussian_multi_with_offset():
    x = np.linspace(0.0, 10.0, 150)
    y = 0.5 + 2.0 * np.exp(-(((x - 4.0) / 0.5) ** 2))
    res = fit_gaussian_multi(x, y, 1, fit_offset=True)
    assert res.offset == pytest.approx(0.5, abs=0.05)
    assert res.components[0].center == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------------------- helpers


def test_histogram_1d_basic():
    data = np.array([0.1, 0.2, 1.1, 1.2, 1.3, 2.5])
    centers, counts = histogram_1d(data, 1.0)
    assert counts.sum() == data.size
    assert counts[1] == 3  # the three values in [1, 2)
