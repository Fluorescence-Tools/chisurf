"""Test TCSPC decay convolution: simulate decay, verify convolution shape, fit lifetime."""
import numpy as np
import pytest

from chisurf.core.fluorescence.tcspc.convolve import convolve_lifetime_spectrum


def test_convolution_shape():
    dt = 0.0141
    n_tac = 2048
    time_axis = np.arange(n_tac, dtype=float) * dt
    sigma = 0.2

    irf = np.exp(-0.5 * (time_axis / sigma) ** 2)
    irf = irf / np.max(irf)

    true_lifetime_spectrum = np.array([1.0, 4.0], dtype=np.float64)

    sim_decay = np.zeros_like(time_axis)
    convolve_lifetime_spectrum(
        output_decay=sim_decay,
        lifetime_spectrum=true_lifetime_spectrum,
        instrument_response_function=irf,
        convolution_stop=n_tac,
        time_axis=time_axis,
    )

    sim_decay = sim_decay / np.max(sim_decay) * 10000
    sim_counts = np.random.poisson(np.maximum(sim_decay, 0)).astype(float)

    irf_model = irf / np.sum(irf)
    model_decay = np.zeros_like(time_axis)
    convolve_lifetime_spectrum(
        output_decay=model_decay,
        lifetime_spectrum=true_lifetime_spectrum,
        instrument_response_function=irf_model,
        convolution_stop=n_tac,
        time_axis=time_axis,
    )

    sim_norm = sim_decay / np.max(sim_decay)
    model_norm = model_decay / np.max(model_decay)
    residual = sim_norm - model_norm
    max_residual = np.max(np.abs(residual))

    assert max_residual < 0.02, f"Max residual too large: {max_residual:.6f}"


def test_fitted_lifetime():
    dt = 0.0141
    n_tac = 2048
    time_axis = np.arange(n_tac, dtype=float) * dt
    sigma = 0.2

    irf = np.exp(-0.5 * (time_axis / sigma) ** 2)
    irf = irf / np.max(irf)

    true_lifetime_spectrum = np.array([1.0, 4.0], dtype=np.float64)

    sim_decay = np.zeros_like(time_axis)
    convolve_lifetime_spectrum(
        output_decay=sim_decay,
        lifetime_spectrum=true_lifetime_spectrum,
        instrument_response_function=irf,
        convolution_stop=n_tac,
        time_axis=time_axis,
    )

    sim_decay = sim_decay / np.max(sim_decay) * 10000
    sim_counts = np.random.poisson(np.maximum(sim_decay, 0)).astype(float)

    from scipy.optimize import curve_fit

    def multi_exp(t, a, tau):
        return a * np.exp(-t / tau)

    fit_range = (50, n_tac // 2)
    popt, _ = curve_fit(
        multi_exp,
        time_axis[fit_range[0]:fit_range[1]],
        sim_counts[fit_range[0]:fit_range[1]],
        p0=[10000, 4.0],
    )
    fitted_tau = popt[1]
    error_pct = abs(fitted_tau - 4.0) / 4.0 * 100

    assert error_pct < 10.0, f"Lifetime fit error too large: {error_pct:.1f}% (tau={fitted_tau:.3f} ns)"
