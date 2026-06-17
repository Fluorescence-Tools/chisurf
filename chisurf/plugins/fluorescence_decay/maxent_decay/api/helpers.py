"""High-level Python API for the MaxEnt TCSPC lifetime / FRET plugin.

This module provides simple helpers for building lifetime / distance grids
and running the core MaxEnt solvers from :mod:`core` on in-memory arrays or
text files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..api.models import MEMRequest, MEMResult, MEMSettings
from ..core.solver import (
    load_tcspc_two_column,
    solve_lifetime_mem,
    solve_fret_mem,
)

ArrayLike = Sequence[float]


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def build_tau_grid(
    tau_min: float = 0.01,
    tau_max: float = 6.0,
    tau_bins: int = 192,
    tau_step: Optional[float] = None,
) -> np.ndarray:
    """Build a lifetime grid ``tau`` in nanoseconds.

    Parameters
    ----------
    tau_min, tau_max : float
        Minimum and maximum lifetime.
    tau_bins : int
        Number of points in the grid.
    tau_step : float, optional
        Legacy grid spacing. If provided, takes precedence over ``tau_bins``
        for backwards compatibility.
    """

    tmin = float(tau_min)
    tmax = float(tau_max)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        raise ValueError("Invalid tau grid bounds")

    if tau_step is not None:
        step = float(tau_step)
        if step <= 0.0:
            raise ValueError("tau_step must be positive")
        return np.arange(tmin, tmax + 0.5 * step, step, dtype=float)

    n = int(tau_bins)
    if n < 2:
        raise ValueError("tau_bins must be >= 2")
    return np.linspace(tmin, tmax, n, dtype=float)


def build_distance_grid(
    R0: float = 52.0,
    r_min_frac: float = 0.1,
    r_max_frac: float = 3.0,
    r_bins: int = 96,
) -> np.ndarray:
    """Build a distance grid ``R`` in Å for FRET mode.

    The bounds are expressed as fractions of ``R0`` to make the grid portable
    across different Förster radii.

    Parameters
    ----------
    R0 : float
        Förster radius in Å.
    r_min_frac, r_max_frac : float
        Lower/upper bounds as fractions of ``R0``.
    r_bins : int
        Number of points in the grid.
    """

    r0 = float(R0)
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("R0 must be a positive finite float")

    f_min = float(r_min_frac)
    f_max = float(r_max_frac)
    if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min <= 0.0 or f_max <= f_min:
        raise ValueError("Invalid distance grid fraction bounds")

    n = int(r_bins)
    if n < 2:
        raise ValueError("r_bins must be >= 2")

    return np.linspace(r0 * f_min, r0 * f_max, n, dtype=float)


# ---------------------------------------------------------------------------
# Lifetime API
# ---------------------------------------------------------------------------


def run_lifetime_mem_from_arrays(
    *,
    decay: ArrayLike,
    irf: ArrayLike,
    dt: float,
    tau: Optional[ArrayLike] = None,
    timeshift: float = 0.0,
    background: float = 0.0,
    lamp_scatter: float = 0.0,
    fitrange: Optional[Tuple[int, int]] = None,
    irf_background: Optional[float] = None,
    nu: float = 1e-3,
    fit_start_fraction: float = 0.9,
    max_iter: int = 200,
    tol: float = 1e-4,
    optimize_nuisance: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run lifetime MEM on in-memory arrays.

    This is a convenience wrapper around :func:`solve_lifetime_mem` that takes
    1D arrays for the decay and IRF and a lifetime grid.
    """

    tau_grid = np.asarray(tau, dtype=float).ravel() if tau is not None else build_tau_grid()

    return solve_lifetime_mem(
        decay=np.asarray(decay, dtype=float),
        lamp=np.asarray(irf, dtype=float),
        dt=float(dt),
        tau=tau_grid,
        timeshift=float(timeshift),
        background=float(background),
        lamp_scatter=float(lamp_scatter),
        fitrange=fitrange,
        irf_background=irf_background,
        fit_start_fraction=float(fit_start_fraction),
        nu=float(nu),
        max_iter=int(max_iter),
        tol=float(tol),
        optimize_nuisance=bool(optimize_nuisance),
        **kwargs,
    )


def run_lifetime_mem_from_files(
    *,
    decay_path: str,
    irf_path: str,
    dt: float,
    tau: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run lifetime MEM using decay/IRF data loaded from text files."""

    decay_arr = load_tcspc_two_column(str(decay_path))
    irf_arr = load_tcspc_two_column(str(irf_path))
    return run_lifetime_mem_from_arrays(decay=decay_arr, irf=irf_arr, dt=float(dt), tau=tau, **kwargs)


# ---------------------------------------------------------------------------
# FRET distance API
# ---------------------------------------------------------------------------


def run_fret_mem_from_arrays(
    *,
    decay: ArrayLike,
    irf: ArrayLike,
    dt: float,
    R: Optional[ArrayLike] = None,
    tau0: float = 4.1,
    R0: float = 52.0,
    x_donly: float = 0.0,
    timeshift: float = 0.0,
    background: float = 0.0,
    lamp_scatter: float = 0.0,
    fitrange: Optional[Tuple[int, int]] = None,
    irf_background: Optional[float] = None,
    fit_start_fraction: float = 0.9,
    nu: float = 5e-2,
    max_iter: int = 200,
    tol: float = 1e-4,
    use_periodic: bool = False,
    period: Optional[float] = None,
    donly: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run FRET distance MEM on in-memory arrays.

    This is a convenience wrapper around :func:`solve_fret_mem`.
    """

    R_grid = np.asarray(R, dtype=float).ravel() if R is not None else build_distance_grid(R0=float(R0))

    decay_arr = np.asarray(decay, dtype=float)
    irf_arr = np.asarray(irf, dtype=float)

    if period is None and use_periodic:
        # Approximate excitation period by the acquisition window.
        period = float(decay_arr.size * float(dt))

    donly_vec = np.asarray(donly, dtype=float).ravel() if donly is not None else None

    return solve_fret_mem(
        decay=decay_arr,
        lamp=irf_arr,
        dt=float(dt),
        R=R_grid,
        tau0=float(tau0),
        R0=float(R0),
        donly=donly_vec,
        x_donly=float(x_donly),
        timeshift=float(timeshift),
        background=float(background),
        lamp_scatter=float(lamp_scatter),
        fitrange=fitrange,
        irf_background=irf_background,
        fit_start_fraction=float(fit_start_fraction),
        nu=float(nu),
        max_iter=int(max_iter),
        tol=float(tol),
        period=float(period) if period is not None else None,
        **kwargs,
    )


def run_fret_mem_from_files(
    *,
    decay_path: str,
    irf_path: str,
    dt: float,
    R: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run FRET distance MEM using decay/IRF data loaded from text files."""

    decay_arr = load_tcspc_two_column(str(decay_path))
    irf_arr = load_tcspc_two_column(str(irf_path))
    return run_fret_mem_from_arrays(decay=decay_arr, irf=irf_arr, dt=float(dt), R=R, **kwargs)


def request_to_result(request: MEMRequest) -> MEMResult:
    """Run a MaxEnt request and normalize the solver dictionary."""
    settings = request.settings
    if settings.mode == "fret":
        r_axis = build_distance_grid(
            R0=settings.R0,
            r_min_frac=settings.r_min_frac,
            r_max_frac=settings.r_max_frac,
            r_bins=settings.r_bins,
        )
        raw = run_fret_mem_from_arrays(
            decay=request.decay,
            irf=request.irf,
            dt=request.dt,
            R=r_axis,
            tau0=settings.tau0,
            R0=settings.R0,
            x_donly=settings.x_donly,
            timeshift=settings.timeshift,
            background=settings.background,
            lamp_scatter=settings.lamp_scatter,
            fitrange=request.fitrange,
            irf_background=settings.irf_background,
            fit_start_fraction=settings.fit_start_fraction,
            nu=settings.nu,
            max_iter=settings.max_iter,
            tol=settings.tol,
            period=settings.period,
            donly=request.donly,
            optimize_nuisance=settings.optimize_nuisance,
            prior=request.prior,
        )
        axis = np.asarray(raw.get("R", []), dtype=float).tolist()
    else:
        tau_grid = build_tau_grid(
            tau_min=settings.tau_min,
            tau_max=settings.tau_max,
            tau_bins=settings.tau_bins,
        )
        raw = run_lifetime_mem_from_arrays(
            decay=request.decay,
            irf=request.irf,
            dt=request.dt,
            tau=tau_grid,
            timeshift=settings.timeshift,
            background=settings.background,
            lamp_scatter=settings.lamp_scatter,
            fitrange=request.fitrange,
            irf_background=settings.irf_background,
            fit_start_fraction=settings.fit_start_fraction,
            nu=settings.nu,
            max_iter=settings.max_iter,
            tol=settings.tol,
            optimize_nuisance=settings.optimize_nuisance,
            prior=request.prior,
        )
        axis = np.asarray(raw.get("tau", []), dtype=float).tolist()

    p = np.asarray(raw.get("p", []), dtype=float).ravel()
    fit_curve = np.asarray(raw.get("fit_curve", []), dtype=float).ravel()
    if fit_curve.size == 0:
        fit_curve = np.asarray(raw.get("y", []), dtype=float).ravel()
    y = np.asarray(raw.get("y", []), dtype=float).ravel()
    if fit_curve.size == y.size:
        residuals = (y - fit_curve).tolist()
    else:
        residuals = []
    return MEMResult(
        p=p.tolist(),
        axis=axis,
        chisq=float(raw.get("chisq", 0.0)),
        S=float(raw.get("S", 0.0)),
        nu=float(raw.get("nu", raw.get("nu_input", settings.nu))),
        timeshift=float(raw.get("timeshift", settings.timeshift)),
        background=float(raw.get("background", settings.background)),
        fit_curve=fit_curve.tolist(),
        residuals=residuals,
        fitrange=tuple(int(x) for x in raw.get("fitrange", request.fitrange or (0, 0))),
        history=[tuple(float(x) for x in item) for item in raw.get("history", [])],
        mode=settings.mode,
        raw=raw,
    )


__all__ = [
    "build_tau_grid",
    "build_distance_grid",
    "request_to_result",
    "run_lifetime_mem_from_arrays",
    "run_lifetime_mem_from_files",
    "run_fret_mem_from_arrays",
    "run_fret_mem_from_files",
]
