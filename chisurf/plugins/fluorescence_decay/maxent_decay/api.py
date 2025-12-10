"""High-level Python API for the MaxEnt TCSPC lifetime / FRET plugin.

This module provides simple helpers for building lifetime / distance grids
and running the core MaxEnt solvers from :mod:`core` on in-memory arrays or
text files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .core import (
    load_tcspc_two_column,
    mem_vin4_lifetime,
    mem_vin4_fret,
)

ArrayLike = Sequence[float]


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def build_tau_grid(
    tau_min: float = 0.001,
    tau_max: float = 10.0,
    tau_step: float = 0.02,
) -> np.ndarray:
    """Build a lifetime grid ``tau`` in nanoseconds.

    Parameters
    ----------
    tau_min, tau_max : float
        Minimum and maximum lifetime (inclusive upper bound is approximated).
    tau_step : float
        Grid spacing.
    """

    if tau_step <= 0.0:
        raise ValueError("tau_step must be positive")
    return np.arange(float(tau_min), float(tau_max) + 0.5 * float(tau_step), float(tau_step), dtype=float)


def build_distance_grid(
    r_min: float = 18.0,
    r_max: float = 120.0,
    r_step: float = 0.5,
) -> np.ndarray:
    """Build a distance grid ``R`` in Å for FRET mode."""

    if r_step <= 0.0:
        raise ValueError("r_step must be positive")
    return np.arange(float(r_min), float(r_max) + 0.5 * float(r_step), float(r_step), dtype=float)


# ---------------------------------------------------------------------------
# Lifetime API
# ---------------------------------------------------------------------------


def run_lifetime_mem_from_arrays(
    *,
    decay: ArrayLike,
    irf: ArrayLike,
    dt: float,
    tau: Optional[ArrayLike] = None,
    nu: float = 1e-3,
    fit_start_fraction: float = 0.9,
    optimize_nuisance: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run lifetime MEM on in-memory arrays.

    This is a convenience wrapper around :func:`mem_vin4_lifetime` that takes
    1D arrays for the decay and IRF and a lifetime grid.
    """

    tau_grid = np.asarray(tau, dtype=float).ravel() if tau is not None else build_tau_grid()

    return mem_vin4_lifetime(
        decay=np.asarray(decay, dtype=float),
        lamp=np.asarray(irf, dtype=float),
        dt=float(dt),
        tau=tau_grid,
        fit_start_fraction=float(fit_start_fraction),
        nu=float(nu),
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
    lamp_scatter: float = 0.0,
    fit_start_fraction: float = 0.9,
    nu: float = 5e-2,
    use_periodic: bool = False,
    period: Optional[float] = None,
    donly: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run FRET distance MEM on in-memory arrays.

    This is a convenience wrapper around :func:`mem_vin4_fret`.
    """

    R_grid = np.asarray(R, dtype=float).ravel() if R is not None else build_distance_grid()

    decay_arr = np.asarray(decay, dtype=float)
    irf_arr = np.asarray(irf, dtype=float)

    if period is None and use_periodic:
        # Approximate excitation period by the acquisition window.
        period = float(decay_arr.size * float(dt))

    donly_vec = np.asarray(donly, dtype=float).ravel() if donly is not None else None

    return mem_vin4_fret(
        decay=decay_arr,
        lamp=irf_arr,
        dt=float(dt),
        R=R_grid,
        tau0=float(tau0),
        R0=float(R0),
        donly=donly_vec,
        timeshift=0.0,
        background=0.0,
        lamp_scatter=float(lamp_scatter),
        fitrange=None,
        nu=float(nu),
        max_iter=200,
        tol=1e-4,
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


__all__ = [
    "build_tau_grid",
    "build_distance_grid",
    "run_lifetime_mem_from_arrays",
    "run_lifetime_mem_from_files",
    "run_fret_mem_from_arrays",
    "run_fret_mem_from_files",
]
