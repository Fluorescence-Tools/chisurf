"""General instrument-response-function (IRF) helpers for TCSPC.

These are deliberately lightweight and dependency-free (NumPy only) so they can be reused
across the lifetime models, the FCS/2D-FLC plugins and any tool that needs an IRF when a
measured one is unavailable:

* :func:`synthetic_irf` -- build a (possibly skewed) Gaussian IRF on a time axis, reusing
  :func:`chisurf.core.math.functions.distributions.generalized_normal_distribution`.
* :func:`detect_rising_edge` -- locate the prompt/rise position of a measured decay or IRF.
* :func:`estimate_irf_from_decay` -- detect the rise of a measured decay and return a
  synthetic IRF centred there (a quick "synthetic IRF" when no measured IRF exists).

For full IRF extraction by Richardson-Lucy deconvolution see
:class:`chisurf.core.fluorescence.tcspc.irf_estimation.IRFEstimator`.
"""

from __future__ import annotations

import numpy as np

from chisurf.core.math.functions.distributions import generalized_normal_distribution

__all__ = ["synthetic_irf", "detect_rising_edge", "estimate_irf_from_decay", "FWHM_TO_SIGMA"]

# FWHM = 2*sqrt(2*ln2) * sigma  for a Gaussian
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def synthetic_irf(
    time_ns: np.ndarray,
    center_ns: float,
    fwhm_ns: float,
    *,
    shape: float = 0.0,
    norm: bool = True,
) -> np.ndarray:
    """Return a synthetic IRF sampled on ``time_ns``.

    A (generalized) normal pulse centred at ``center_ns`` with full-width-half-maximum
    ``fwhm_ns``. ``shape`` adds skewness (0 = symmetric Gaussian), which captures the
    asymmetric tail of a real detector response.

    Parameters
    ----------
    time_ns
        Time axis (ns).
    center_ns
        Pulse centre (ns).
    fwhm_ns
        Full width at half maximum (ns).
    shape
        Skewness parameter passed through to
        :func:`~chisurf.core.math.functions.distributions.generalized_normal_distribution`.
    norm
        Normalize the IRF to unit sum (default).
    """
    time_ns = np.asarray(time_ns, dtype=float)
    scale = max(float(fwhm_ns), 1e-12) * FWHM_TO_SIGMA
    return generalized_normal_distribution(
        time_ns, loc=float(center_ns), scale=scale, shape=float(shape), norm=norm
    )


def detect_rising_edge(decay: np.ndarray, *, smooth: int = 5) -> int:
    """Return the index of the steepest rising edge of a decay/IRF (the prompt position).

    The rise is the global maximum of the (optionally smoothed) first difference. This is a
    fast, robust estimate of the time-zero / prompt channel without any model fitting.

    Parameters
    ----------
    decay
        Measured decay or IRF histogram.
    smooth
        Box-smoothing width (samples) applied before differencing (default 5).
    """
    y = np.asarray(decay, dtype=float)
    if y.size < 3:
        return 0
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / float(smooth)
        y = np.convolve(y, k, mode="same")
    return int(np.argmax(np.diff(y)))


def estimate_irf_from_decay(
    decay: np.ndarray,
    time_ns: np.ndarray,
    *,
    fwhm_ns: float | None = None,
    shape: float = 0.0,
    smooth: int = 5,
) -> np.ndarray:
    """Detect the prompt position of a measured decay and return a matching synthetic IRF.

    Useful when no measured IRF is available: the rising edge of the decay marks the
    excitation pulse, so a synthetic pulse placed there is a serviceable IRF for
    deconvolution. ``fwhm_ns`` defaults to a few time-axis steps.

    Parameters
    ----------
    decay
        Measured decay histogram.
    time_ns
        Its time axis (ns).
    fwhm_ns
        IRF width (ns); defaults to ~4 time-axis bins.
    shape
        Skewness of the synthetic pulse.
    smooth
        Smoothing width for the rising-edge detection.
    """
    time_ns = np.asarray(time_ns, dtype=float)
    idx = detect_rising_edge(decay, smooth=smooth)
    center = float(time_ns[idx])
    if fwhm_ns is None:
        dt = float(np.median(np.diff(time_ns))) if time_ns.size > 1 else 1.0
        fwhm_ns = 4.0 * dt
    return synthetic_irf(time_ns, center, fwhm_ns, shape=shape)
