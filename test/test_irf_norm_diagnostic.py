"""Diagnostic: IRF normalization mismatch between simulator and fitting model.

The simulator normalizes IRF to peak=1, the fitting model to sum=1.
This test verifies the convolution is scale-invariant (shape unaffected).
"""
import numpy as np
import pytest

import chisurf.core.fluorescence.tcspc.convolve as convolve_mod


@pytest.fixture
def setup():
    n_points = 4096
    dt = 0.0141
    time_axis = np.arange(n_points, dtype=float) * dt
    true_tau = 4.0
    lifetime_spectrum = np.array([1.0, true_tau])
    sigma = 0.2
    irf_raw = np.exp(-0.5 * (time_axis / sigma) ** 2)
    irf_peak_norm = irf_raw / np.max(irf_raw)
    irf_area_norm = irf_raw / np.sum(irf_raw)
    return {
        "n_points": n_points,
        "dt": dt,
        "time_axis": time_axis,
        "true_tau": true_tau,
        "lifetime_spectrum": lifetime_spectrum,
        "irf_peak_norm": irf_peak_norm,
        "irf_area_norm": irf_area_norm,
    }


def test_convolution_scale_invariant_tttrlib(setup):
    s = setup
    decay_peak = np.zeros(s["n_points"])
    decay_area = np.zeros(s["n_points"])
    convolve_mod.convolve_lifetime_spectrum(
        decay_peak, s["lifetime_spectrum"], s["irf_peak_norm"],
        s["n_points"], s["time_axis"],
    )
    convolve_mod.convolve_lifetime_spectrum(
        decay_area, s["lifetime_spectrum"], s["irf_area_norm"],
        s["n_points"], s["time_axis"],
    )
    decay_peak /= np.max(decay_peak)
    decay_area /= np.max(decay_area)
    diff = np.max(np.abs(decay_peak - decay_area))
    assert diff < 1e-6, f"tttrlib convolution not scale-invariant: diff={diff:.2e}"


def test_convolution_scale_invariant_numba(setup):
    s = setup
    decay_peak = np.zeros(s["n_points"])
    decay_area = np.zeros(s["n_points"])
    convolve_mod.convolve_lifetime_spectrum_nb(
        decay_peak, s["lifetime_spectrum"], s["irf_peak_norm"],
        s["n_points"], s["time_axis"],
    )
    convolve_mod.convolve_lifetime_spectrum_nb(
        decay_area, s["lifetime_spectrum"], s["irf_area_norm"],
        s["n_points"], s["time_axis"],
    )
    decay_peak /= np.max(decay_peak)
    decay_area /= np.max(decay_area)
    diff = np.max(np.abs(decay_peak - decay_area))
    assert diff < 1e-6, f"numba convolution not scale-invariant: diff={diff:.2e}"


def test_lifetime_accuracy(setup):
    s = setup

    def extract_lifetime(decay, t):
        t_start = t[-1] * 0.1
        t_end = t[-1] * 0.5
        mask = (t > t_start) & (t < t_end) & (decay > 0)
        if mask.sum() < 10:
            return None
        coeffs = np.polyfit(t[mask], np.log(decay[mask]), 1)
        slope = coeffs[0]
        return -1.0 / slope if slope < 0 else None

    decay = np.zeros(s["n_points"])
    convolve_mod.convolve_lifetime_spectrum(
        decay, s["lifetime_spectrum"], s["irf_peak_norm"],
        s["n_points"], s["time_axis"],
    )
    tau_eff = extract_lifetime(decay, s["time_axis"])
    assert tau_eff is not None, "Could not extract lifetime"
    error_pct = abs(tau_eff - s["true_tau"]) / s["true_tau"] * 100
    assert error_pct < 5.0, f"Lifetime error too large: {error_pct:.1f}% (tau={tau_eff:.3f} ns)"
