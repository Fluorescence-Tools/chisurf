from __future__ import annotations

import numpy as np

from chisurf.core.fluorescence.tcspc import IRFEstimator as _IRFEstimator

from ..api.models import IRFEstimationSettings, IRFEstimationResult


def estimate_irf(
    intensity: np.ndarray,
    dt: float,
    settings: IRFEstimationSettings | None = None,
    channel_axis: np.ndarray | None = None,
) -> IRFEstimationResult:
    """Run IRF estimation on intensity data.

    Parameters
    ----------
    intensity : np.ndarray
        1D intensity values (counts per channel).
    dt : float
        Time per channel in nanoseconds.
    settings : IRFEstimationSettings, optional
        Estimation parameters.
    channel_axis : np.ndarray, optional
        Time axis array (used for range selection masking).

    Returns
    -------
    IRFEstimationResult
        Estimation result with IRF, fitted params, and derived quantities.
    """
    if settings is None:
        settings = IRFEstimationSettings()

    # Ensure odd values
    window_length = settings.window_length
    if window_length % 2 == 0:
        window_length += 1
    regularization = settings.regularization
    if regularization > 1 and regularization % 2 == 0:
        regularization += 1

    # Prepare data
    decay = intensity.copy()

    # Apply background correction
    if settings.manual_background > 0:
        decay = np.maximum(decay - settings.manual_background, 0.0)

    # Apply range selection masking
    if settings.use_range_selection and channel_axis is not None:
        min_ch, max_ch = settings.range_bounds
        mask = (np.arange(len(decay)) >= min_ch) & (np.arange(len(decay)) <= max_ch)
        decay[~mask] = 0.0

    estimator = _IRFEstimator(decay.reshape(-1, 1), dt=1.0)

    estimator.find_t0_t1(
        window_length=window_length,
        polyorder=settings.polyorder,
    )
    estimator.fit_exponential()
    estimator.generate_data_fit()
    estimator.generate_kernel()
    estimator.richardson_lucy_deconvolution(
        iterations=settings.rl_iterations,
        regularization=regularization,
    )

    irf = estimator.irf[:, 0]
    params = estimator.params

    k_per_channel = params["k"]
    k_per_ns = k_per_channel / dt
    tau_channels = 1.0 / k_per_channel if k_per_channel > 0 else float("inf")
    tau_ns = tau_channels * dt

    return IRFEstimationResult(
        irf=irf.tolist(),
        params={"A": float(params["A"][0]), "C": float(params["C"][0]), "k": float(params["k"])},
        time_axis=(np.arange(len(irf)) * dt).tolist(),
        dt=dt,
        lifetime_ns=tau_ns,
        decay_rate_ns=k_per_ns,
        amplitude=float(params["A"][0]),
        offset=float(params["C"][0]),
    )
