"""Background estimation for TTTR burst experiments.

This module implements an exponential tail fit to the interphoton time
histogram to estimate background count rates, following the approach used
in PAM's `Estimate_Background_From_Burst.m` and the methodology
of Ingargiola et al., PLoS ONE (2016).

The main public entry point is :func:`estimate_background_from_bursts`,
which is also re-exported via :mod:`chisurf.core.fluorescence.burst`.
"""

from typing import Any, Dict, Mapping

import numpy as np
import tttrlib

try:  # SciPy is a core dependency of ChiSurf, but fail clearly if missing
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "chisurf.core.fluorescence.burst.background requires SciPy. "
        "Please ensure that the 'scipy' package is installed."
    ) from exc


def estimate_background_from_interphoton_times(
    dt_ms: np.ndarray,
    *,
    binsize_ms: float = 0.1,
    tail_fraction: float = 0.2,
    min_counts: int = 1,
) -> float:
    """Estimate a background count rate (in kHz) from interphoton times.

    Parameters
    ----------
    dt_ms : np.ndarray
        One-dimensional array of interphoton times in **milliseconds**.
    binsize_ms : float, optional
        Histogram bin width in milliseconds. Default is 0.1 ms, matching
        the PAM implementation (``0:.1:max(MT)``).
    tail_fraction : float, optional
        Fraction of the histogram range used for the tail fit. Bins with
        centers strictly greater than ``tail_fraction * max(center)`` and
        at least ``min_counts`` counts are considered. A value of 0.2
        reproduces the PAM criterion ``dt > max(dt)/5`` (i.e. use the
        last ~80% of the range).
    min_counts : int, optional
        Minimum number of counts per histogram bin for inclusion in the
        fit. Default is 1.

    Returns
    -------
    float
        Estimated background count rate in **kHz**. Returns 0.0 if the
        estimate cannot be obtained (e.g. too few photons).
    """
    dt_ms = np.asarray(dt_ms, dtype=np.float64)
    # Remove non-positive intervals
    dt_ms = dt_ms[dt_ms > 0.0]
    if dt_ms.size == 0:
        return 0.0

    max_dt = float(dt_ms.max())
    if max_dt <= 0.0:
        return 0.0

    # Histogram similar to MATLAB's hist(MT, 0:.1:max(MT))
    edges = np.arange(0.0, max_dt + binsize_ms, binsize_ms, dtype=np.float64)
    if edges.size < 2:
        return 0.0

    counts, edges = np.histogram(dt_ms, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if centers.size == 0:
        return 0.0

    # Tail selection
    tail_threshold = tail_fraction * float(centers.max())
    valid = (centers > tail_threshold) & (counts >= min_counts)
    if not np.any(valid):
        return 0.0

    xdata = centers[valid]
    ydata = counts[valid].astype(np.float64)

    # Negative log-likelihood for Poisson counts y ~ Poisson(model),
    # model = A * exp(-lambda * x).
    def neg_log_likelihood(params: np.ndarray) -> float:
        """Negative log-likelihood for Poisson counts with exponential model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector (A, lam) where model = A * exp(-lam * x).

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        A, lam = params
        if A <= 0.0 or lam <= 0.0:
            return np.inf
        model = A * np.exp(-lam * xdata)
        # Avoid log(0) and division by zero
        eps = 1e-12
        model_safe = model + eps
        ratio = ydata / model_safe
        # Expression mirrors the MATLAB implementation
        term = ydata * np.log(np.maximum(ratio, eps)) - ydata + model_safe
        return float(np.sum(term))

    # Initial guess from first bin height and overall time span
    A0 = float(counts[0]) if counts[0] > 0 else float(ydata.max())
    lam0 = 3.0 / max_dt  # cf. x0 = [hMT(1), 3/max(dt)] in MATLAB
    x0 = np.array([A0, lam0], dtype=float)

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=((0.0, None), (0.0, None)),
    )

    if not result.success:
        # Fallback: estimate from mean of tail interphoton times
        mean_dt = float(np.mean(xdata))
        if mean_dt <= 0.0:
            return 0.0
        lam = 1.0 / mean_dt
    else:
        lam = float(result.x[1])

    # lambda has units 1/ms. Treat this as kHz (1/ms == kHz).
    if lam < 0.0:
        lam = 0.0
    return lam


def estimate_background_from_bursts(
    tttr: tttrlib.TTTR,
    detectors: Mapping[str, Mapping[str, Any]],
    *,
    binsize_ms: float = 0.1,
    tail_fraction: float = 0.8,
    min_counts: int = 1,
) -> Dict[str, float]:
    """Estimate background count rates (kHz) for multiple detector definitions.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR object containing the photon data. The object must provide
        ``macro_times``, ``micro_times``, ``routing_channel``, and a
        ``header.macro_time_resolution`` attribute (in seconds).
    detectors : Mapping[str, Mapping[str, Any]]
        Detector definitions, typically obtained from
        :meth:`DetectorWizardPage.get_settings` as
        ``settings["detectors"]``. Each detector configuration should
        contain at least:

        - ``"chs"``: list of routing-channel integers.
        - ``"micro_time_ranges"``: list of ``(start_bin, stop_bin)``
          micro-time ranges. If omitted or empty, all micro-times are used.

    binsize_ms : float, optional
        Histogram bin width in milliseconds. Default is 0.1 ms.
    tail_fraction : float, optional
        Fraction of the histogram range used for the tail fit.
    min_counts : int, optional
        Minimum number of counts per histogram bin for inclusion in the
        fit.

    Returns
    -------
    Dict[str, float]
        Mapping from detector name to estimated background rate in kHz.
        Detectors for which no estimate can be obtained are assigned 0.0.
    """
    macro = np.asarray(tttr.macro_times, dtype=np.int64)
    if macro.size < 2:
        return {name: 0.0 for name in detectors.keys()}

    rout = np.asarray(tttr.routing_channel)
    micro = np.asarray(tttr.micro_times)

    header = tttr.header
    # macro_time_resolution is in seconds; convert dt to milliseconds
    dt_scale = float(getattr(header, "macro_time_resolution", 1.0)) * 1000.0

    results: Dict[str, float] = {}

    for det_name, det_info in detectors.items():
        chs = np.asarray(det_info.get("chs", []), dtype=int)
        if chs.size == 0:
            results[det_name] = 0.0
            continue

        # Channel selection
        mask = np.isin(rout, chs)

        # Optional micro-time windowing
        mt_ranges = det_info.get("micro_time_ranges", []) or []
        if mt_ranges:
            mt_mask = np.zeros_like(mask, dtype=bool)
            for start, stop in mt_ranges:
                start_i = int(start)
                stop_i = int(stop)
                mt_mask |= (micro >= start_i) & (micro < stop_i)
            mask &= mt_mask

        times = macro[mask]
        if times.size < 2:
            results[det_name] = 0.0
            continue

        dt_ms = np.diff(times.astype(np.float64)) * dt_scale
        bg_khz = estimate_background_from_interphoton_times(
            dt_ms,
            binsize_ms=binsize_ms,
            tail_fraction=tail_fraction,
            min_counts=min_counts,
        )
        results[det_name] = float(bg_khz)

    return results
