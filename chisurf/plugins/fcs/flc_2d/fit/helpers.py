"""Small helper ports from the MATLAB 2D-FLC toolbox.

* :func:`histogram_1d` — port of ``TK_Histgram1D`` (fixed-width 1D histogram).
* :func:`create_1d_fdc` — port of ``TK_Create1DFDC_01``: the zero-lag (``dT = 0``)
  same-macro-time fluorescence-decay coincidence, returned as the diagonal of the
  ``dT = 0`` 2D-FDC for both linear and logarithmic micro-time axes.
* :func:`search_rise_irf` — port of ``TK_MyMain_Search_RiseIRF_1DMEM``: scan the IRF rise
  position, run the 1D-MEM at each shift and average the distributions over a window around
  the optimum (the IRF-timing systematic is reduced by averaging).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["histogram_1d", "create_1d_fdc", "search_rise_irf", "RiseIRFResult"]


def histogram_1d(data: np.ndarray, bin_div: float):
    """Fixed-width 1D histogram (port of ``TK_Histgram1D``).

    Bins ``data`` into intervals of width ``bin_div`` spanning ``floor(min)`` to
    ``ceil(max)``. Returns ``(centers, counts)``.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.zeros(0), np.zeros(0)
    lo = np.floor(data.min() / bin_div)
    hi = np.ceil(data.max() / bin_div)
    n_pts = int(hi - lo) + 1
    x = np.linspace(lo * bin_div, hi * bin_div, n_pts)
    idx = (np.floor(data / bin_div) - lo).astype(int)
    idx = np.clip(idx, 0, n_pts - 1)
    counts = np.bincount(idx, minlength=n_pts).astype(float)
    return x, counts


def create_1d_fdc(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    *,
    tMin_ticks: int,
    tMax_ticks: int,
    lint_bin_factor: int = 1,
    logt_imax: int = 100,
    n_chunks: int = 1,
):
    """Build the 1D-FDC (zero-lag same-macro-time decay coincidence).

    Port of ``TK_Create1DFDC_01``: it is the diagonal of the ``dT = 0`` 2D-FDC. Returns
    ``(lin_t, lin_1dfdc, log_t, log_1dfdc)`` where the ``_t`` axes are micro-time ticks.

    Parameters
    ----------
    macro_times, micro_times
        Integer macro-time (ascending) and micro-time (TCSPC channel) ticks.
    tMin_ticks, tMax_ticks
        Micro-time gate in TCSPC channel units.
    lint_bin_factor, logt_imax, n_chunks
        As in :func:`chisurf.plugins.fcs.flc_2d.core.create_2d_fdc_numba_int`.
    """
    from ..core import create_2d_fdc_numba_int

    macro = np.ascontiguousarray(macro_times, dtype=np.int64)
    micro = np.ascontiguousarray(micro_times, dtype=np.int64)
    mat_lin, lin_t, mat_log, log_t = create_2d_fdc_numba_int(
        macro_times=macro,
        micro_times=micro,
        dT_ticks=np.int64(0),
        ddT_ticks=np.int64(1),
        Tstart_ticks=np.int64(macro[0]),
        Tend_ticks=np.int64(macro[-1]),
        tMin_over_tStep=np.int64(tMin_ticks),
        tMax_over_tStep=np.int64(tMax_ticks),
        lint_bin_factor=int(lint_bin_factor),
        logt_imax_in=int(logt_imax),
        build_lin=True,
        n_chunks=int(n_chunks),
    )
    return lin_t, np.diag(mat_lin).copy(), log_t, np.diag(mat_log).copy()


@dataclass
class RiseIRFResult:
    """Result of an IRF-rise scan over the 1D-MEM."""

    best_shift: int  # IRF shift (channels) with the lowest estimator Q
    shifts: np.ndarray  # all tested shifts
    estimator_q: np.ndarray  # Q per shift
    distributions: np.ndarray  # (n_comp, n_shift) per-shift normalized distributions
    averaged: np.ndarray  # window-averaged distribution around the optimum
    tau_grid: np.ndarray  # lifetimes (ns)


def search_rise_irf(
    decay: np.ndarray,
    time_ns: np.ndarray,
    tau_grid: np.ndarray,
    irf: np.ndarray,
    irf_time_ns: np.ndarray,
    *,
    shifts=range(-7, 8),
    average_width: int = 13,
    mem_kwargs: dict | None = None,
):
    """Scan the IRF rise position and average 1D-MEM distributions around the optimum.

    Port of ``TK_MyMain_Search_RiseIRF_1DMEM``. For each integer ``shift`` (channels) the
    IRF is rolled, an exponential basis is rebuilt, the 1D-MEM is solved, and the estimator
    ``Q`` recorded. The distribution is averaged over a window of ``average_width`` shifts
    centered on the minimum-``Q`` shift, which suppresses the IRF-timing systematic.

    Parameters
    ----------
    decay
        Measured 1D fluorescence-decay correlation.
    time_ns, tau_grid
        Decay time axis and lifetime grid (ns).
    irf, irf_time_ns
        Instrument response and its time axis (ns).
    shifts
        Iterable of integer channel shifts to test.
    average_width
        Number of shifts to average around the optimum.
    mem_kwargs
        Extra keyword arguments forwarded to :func:`...fit.mem_1d.solve_mem_1d`.
    """
    from .ilt import build_exp_basis
    from .mem_1d import solve_mem_1d

    shifts = list(shifts)
    tau = np.asarray(tau_grid, dtype=float)
    mem_kwargs = dict(mem_kwargs or {})
    mem_kwargs.setdefault("t_min", float(time_ns[0]))
    mem_kwargs.setdefault("t_max", float(time_ns[-1]))

    dists = np.zeros((tau.size, len(shifts)))
    qvals = np.zeros(len(shifts))
    for j, sh in enumerate(shifts):
        irf_shifted = np.roll(np.asarray(irf, dtype=float), sh)
        basis = build_exp_basis(time_ns, tau, irf=irf_shifted, irf_time_ns=irf_time_ns)
        res = solve_mem_1d(decay, basis, tau, **mem_kwargs)
        norm = res.amplitudes.sum() or 1.0
        dists[:, j] = res.amplitudes / norm
        qvals[j] = res.estimator_q

    best = int(np.argmin(qvals))
    half = average_width // 2
    lo = max(0, best - half)
    hi = min(len(shifts), best + half + 1)
    averaged = dists[:, lo:hi].mean(axis=1)

    return RiseIRFResult(
        best_shift=shifts[best],
        shifts=np.asarray(shifts),
        estimator_q=qvals,
        distributions=dists,
        averaged=averaged,
        tau_grid=tau,
    )
