"""Lifetime-filtered (species-resolved) correlation and interconversion kinetics.

This is the dynamics half of 2D-FLC. Once the lifetime species have been resolved
(by :mod:`chisurf.plugins.fcs.flc_2d.fit.ilt`), their interconversion is read out as a
*filtered* fluorescence correlation: each photon is weighted by lifetime-filter values
(Bohmer/Enderlein/Kapusta fFCS) and the species auto- and cross-correlations are computed
with the ``tttrlib`` multi-tau correlator. For a two-state exchange the species
auto-correlations decay and the cross-correlation is anti-correlated, both with the same
relaxation rate ``k = sum of the interconversion rates``.

The heavy lifting (filter construction, multi-tau correlation) reuses existing chisurf /
tttrlib infrastructure rather than reimplementing it:
:func:`chisurf.core.fluorescence.fcs.filtered.calc_ffcs_filters` and
``tttrlib.Correlator``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "species_filters",
    "filtered_correlation",
    "fit_relaxation",
    "SpeciesCorrelation",
]


def species_filters(
    total_decay: np.ndarray,
    species_decays: Sequence[np.ndarray],
) -> np.ndarray:
    """Return lifetime filters ``(n_species, n_microtime_bins)`` for fFCS.

    Thin wrapper over :func:`chisurf.core.fluorescence.fcs.filtered.calc_ffcs_filters`
    (weighted-least-squares fFCS filters, PAM/Kapusta convention).
    """
    from chisurf.core.fluorescence.fcs.filtered import calc_ffcs_filters

    filters, _recon, _resid = calc_ffcs_filters(
        np.asarray(total_decay, dtype=float),
        [np.asarray(s, dtype=float) for s in species_decays],
    )
    return filters


@dataclass
class SpeciesCorrelation:
    """Species-resolved filtered correlation curves.

    Attributes
    ----------
    lag_s
        Correlation lag times (seconds).
    auto
        ``{species_index: G(tau)}`` species auto-correlations.
    cross
        ``{(i, j): G(tau)}`` species cross-correlations (``i < j``).
    """

    lag_s: np.ndarray
    auto: dict[int, np.ndarray]
    cross: dict[tuple[int, int], np.ndarray]


def filtered_correlation(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    filters: np.ndarray,
    macro_time_resolution_s: float,
    *,
    n_bins: int = 8,
    n_casc: int = 25,
    n_microtime_bins: int | None = None,
) -> SpeciesCorrelation:
    """Compute species auto- and cross-correlations with ``tttrlib``.

    Parameters
    ----------
    macro_times
        Photon macro-times in clock ticks (ascending). For raw TTTR data these are the
        native macro-time tags.
    micro_times
        Photon micro-times as TCSPC-channel indices (same length as ``macro_times``).
    filters
        Lifetime filters ``(n_species, n_microtime_bins)`` from :func:`species_filters`.
    macro_time_resolution_s
        Seconds per macro-time tick (e.g. ``header.macro_time_resolution``); converts the
        lag axis to seconds.
    n_bins, n_casc
        Multi-tau correlator settings (channels per cascade, number of cascades).
    n_microtime_bins
        Number of micro-time channels; defaults to ``filters.shape[1]``.
    """
    import tttrlib

    macro = np.ascontiguousarray(macro_times, dtype=np.uint64)
    micro = np.asarray(micro_times)
    n_mt = n_microtime_bins or filters.shape[1]
    micro_idx = np.clip(micro, 0, n_mt - 1).astype(np.int64)
    n_species = filters.shape[0]

    # Per-photon weights for each species (filter value at the photon's micro-time).
    weights = [
        np.ascontiguousarray(filters[s, micro_idx], dtype=np.float64) for s in range(n_species)
    ]

    def _corr(wa: np.ndarray, wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = tttrlib.Correlator()
        c.n_bins = int(n_bins)
        c.n_casc = int(n_casc)
        c.set_macrotimes(macro, macro)
        c.set_weights(wa, wb)
        c.run()
        x = np.asarray(c.get_x_axis(), dtype=float) * macro_time_resolution_s
        g = np.asarray(c.get_corr_normalized(), dtype=float)
        return x, g

    lag = None
    auto: dict[int, np.ndarray] = {}
    cross: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_species):
        lag, auto[i] = _corr(weights[i], weights[i])
    for i in range(n_species):
        for j in range(i + 1, n_species):
            _, cross[(i, j)] = _corr(weights[i], weights[j])
    return SpeciesCorrelation(lag_s=lag, auto=auto, cross=cross)


def fit_relaxation(
    lag_s: np.ndarray,
    g: np.ndarray,
    *,
    t_min: float = 5e-4,
    t_max: float = 0.5,
    p0_rate: float = 50.0,
) -> dict[str, float]:
    """Fit a single-exponential relaxation ``G(t) = A*exp(-k t) + b`` to a curve.

    Returns a dict with ``rate`` (s^-1), ``relaxation_time_s`` (= 1/rate), ``amplitude``,
    ``offset`` and the fit-window-restricted ``r2``. ``A`` may be negative (anti-correlated
    cross terms); the magnitude of ``rate`` is what encodes the interconversion.
    """
    from scipy.optimize import curve_fit

    lag_s = np.asarray(lag_s, dtype=float)
    g = np.asarray(g, dtype=float)
    sel = np.isfinite(lag_s) & np.isfinite(g) & (lag_s > t_min) & (lag_s < t_max)
    x, y = lag_s[sel], g[sel]
    if x.size < 4:
        raise ValueError("not enough points in the fit window")

    def model(t, A, k, b):
        return A * np.exp(-k * t) + b

    p0 = [y[0] - y[-1], p0_rate, y[-1]]
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=40000)
    resid = y - model(x, *popt)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-300
    rate = abs(float(popt[1]))
    return {
        "rate": rate,
        "relaxation_time_s": 1.0 / rate if rate > 0 else np.inf,
        "amplitude": float(popt[0]),
        "offset": float(popt[2]),
        "r2": 1.0 - ss_res / ss_tot,
    }
