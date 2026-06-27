"""Lifetime resolution must recover the two reference species (tau = 1 and 3 ns)."""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fcs.flc_2d.api import (
    lifetime_spectrum,
    two_d_fdc,
    two_d_spectrum,
)
from chisurf.plugins.fcs.flc_2d.fit.ilt import build_exp_basis, ilt_1d, lifetime_grid


def test_ilt_1d_recovers_synthetic_biexponential():
    """A clean synthetic 1+3 ns decay (no IRF) is resolved into two components."""
    time_ns = np.linspace(0, 20, 400)
    tau_true = (1.0, 3.0)
    decay = 0.6 * np.exp(-time_ns / tau_true[0]) + 0.4 * np.exp(-time_ns / tau_true[1])
    decay = decay / decay.max() * 1e5
    tau = lifetime_grid(0.3, 8.0, 40)
    E = build_exp_basis(time_ns, tau, normalize=True)
    res = ilt_1d(decay, E, tau, method="nnls", reg=None)
    peaks = np.sort(res.peak_lifetimes(2))
    assert peaks.size == 2
    assert 0.6 < peaks[0] < 1.5
    assert 2.4 < peaks[1] < 3.8


@pytest.mark.slow
def test_lifetime_spectrum_reference(reference_photons):
    """1D ILT of the reference micro-time decay yields ~1 ns and ~3 ns components."""
    res = lifetime_spectrum(
        reference_photons["micro_ticks"],
        n_microtime_bins=reference_photons["n_microtime_bins"],
        micro_time_resolution_ns=reference_photons["micro_resolution_ns"],
        tau_range=(0.3, 8.0),
        n_components=40,
        irf=reference_photons["irf"],
        irf_time_ns=reference_photons["irf_time_ns"],
        method="nnls",
    )
    peaks = np.sort(res.peak_lifetimes(2))
    assert peaks.size == 2, f"expected two components, got {peaks}"
    assert 0.6 <= peaks[0] <= 1.4, f"short lifetime off: {peaks[0]}"
    assert 2.4 <= peaks[1] <= 3.7, f"long lifetime off: {peaks[1]}"


@pytest.mark.slow
def test_two_d_spectrum_reference(reference_photons):
    """The 2D-FDC inverts to a non-negative spectrum whose marginal spans the species."""
    out = two_d_fdc(
        reference_photons["macro_ticks"],
        reference_photons["micro_ticks"],
        dT=1000,
        ddT=2000,
        tMin=1,
        tMax=reference_photons["n_microtime_bins"],
        logt_imax=60,
    )
    tstep = reference_photons["micro_resolution_ns"]
    time_ns = (out["mat_lin_t"] + 1) * tstep
    res = two_d_spectrum(
        out["mat_lin"],
        time_ns,
        tau_range=(0.3, 8.0),
        n_components=24,
        irf=reference_photons["irf"],
        irf_time_ns=reference_photons["irf_time_ns"],
        method="tikhonov",
        max_bins=80,
    )
    assert np.all(res.spectrum >= 0)
    assert np.isfinite(res.chi2)
    # marginal should carry weight in the 1-3 ns lifetime band
    band = (res.tau_grid >= 0.7) & (res.tau_grid <= 3.5)
    assert res.marginal[band].sum() > 0.5 * res.marginal.sum()
