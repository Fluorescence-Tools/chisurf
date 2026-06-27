"""High-level, Qt-free API for 2D fluorescence lifetime correlation (2D-FLC).

The pipeline has two halves:

* **Lifetime resolution** -- resolve the fluorescence-lifetime species from a decay
  histogram (:func:`lifetime_spectrum`) or from a 2D fluorescence-decay correlation
  matrix built from TTTR photon pairs (:func:`two_d_fdc` + :func:`two_d_spectrum`).
* **Dynamics** -- read out the interconversion of those species as a lifetime-filtered
  (species-resolved) correlation and fit its relaxation time
  (:func:`species_correlation`).

Everything operates on plain NumPy arrays so it is usable from scripts, notebooks, the
RPC backend and the GUI alike. ``tttrlib`` and ``scipy`` are imported lazily.

Example:
-------
>>> data = load_tttr("measurement.ptu")                      # doctest: +SKIP
>>> spec = lifetime_spectrum(data.micro_times,               # doctest: +SKIP
...                          n_microtime_bins=data.n_microtime_channels,
...                          micro_time_resolution_ns=data.micro_time_resolution_ns)
>>> spec.peak_lifetimes(2)                                   # doctest: +SKIP
array([1.0, 3.0])
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import TwoDFDCreator
from .fit.dynamics import (
    SpeciesCorrelation,
    filtered_correlation,
    fit_relaxation,
    species_filters,
)
from .fit.gaussian import GaussianFitResult, fit_gaussian_multi
from .fit.global_mem import GlobalMEMResult, solve_global_mem_2d
from .fit.helpers import RiseIRFResult, create_1d_fdc, histogram_1d, search_rise_irf
from .fit.ilt import (
    ILTResult1D,
    ILTResult2D,
    build_exp_basis,
    ilt_1d,
    ilt_2d,
    lcurve_1d,
    lifetime_grid,
)
from .fit.kinetics import RateMatrixResult, fit_rate_matrix
from .fit.mem_1d import OneDMEMResult, solve_mem_1d

__all__ = [
    "TttrData",
    "load_tttr",
    "lifetime_spectrum",
    "lifetime_spectrum_mem",
    "species_decay_patterns",
    "two_d_fdc",
    "two_d_fdc_scan",
    "one_d_fdc",
    "two_d_spectrum",
    "species_correlation",
    "rate_matrix_kinetics",
    "global_lifetime_mem",
    "correlate_tttr",
    "fit_tikhonov_2d",
    "fit_mem_2d",
    "fit_gaussian_components",
    "search_irf_rise",
    "simulate_stream",
    "histogram_1d",
    "make_synthetic_irf",
    "detect_irf",
    "lifetime_lcurve",
]


# --------------------------------------------------------------------------- loading


@dataclass
class TttrData:
    """A loaded TTTR photon stream and the calibration needed to analyse it."""

    macro_times: np.ndarray  # clock ticks, ascending
    micro_times: np.ndarray  # TCSPC channel indices
    routing_channels: np.ndarray  # detector/routing channel per photon
    macro_time_resolution_s: float  # seconds per macro tick
    micro_time_resolution_ns: float  # ns per micro channel
    n_microtime_channels: int

    @property
    def n_photons(self) -> int:
        """Number of photons in the stream."""
        return int(self.macro_times.shape[0])


def load_tttr(file_path: str | Path, routing_channels: Sequence[int] | None = None) -> TttrData:
    """Load a TTTR file with ``tttrlib`` (format auto-detected).

    Parameters
    ----------
    file_path
        Path to a PTU/HT3/PT3/SPC/HDF5/... file.
    routing_channels
        Optional subset of detector channels to keep (default: all).
    """
    import tttrlib

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TTTR file not found: {path}")
    tttr = tttrlib.TTTR(str(path))
    header = tttr.get_header()

    macro = np.asarray(tttr.macro_times)
    micro = np.asarray(tttr.micro_times)
    routing = np.asarray(tttr.routing_channels)
    if routing_channels is not None:
        keep = np.isin(routing, np.asarray(list(routing_channels)))
        macro, micro, routing = macro[keep], micro[keep], routing[keep]

    micro_res_ns = float(getattr(header, "micro_time_resolution", 0.0)) * 1e9
    n_channels = int(
        getattr(header, "number_of_micro_time_channels", int(micro.max()) + 1 if micro.size else 1)
    )
    return TttrData(
        macro_times=macro,
        micro_times=micro,
        routing_channels=routing,
        macro_time_resolution_s=float(getattr(header, "macro_time_resolution", 1.0)),
        micro_time_resolution_ns=micro_res_ns,
        n_microtime_channels=n_channels,
    )


# ----------------------------------------------------------------- lifetime spectrum


def _decay_histogram(micro_times: np.ndarray, n_bins: int) -> np.ndarray:
    # Drop micro-times outside [0, n_bins): the TCSPC window can be wider than the gate
    # used for the lifetime fit, and clipping would pile tail photons into the edge bin.
    idx = np.asarray(micro_times).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins)[:n_bins].astype(float)


def lifetime_spectrum(
    micro_times: np.ndarray,
    n_microtime_bins: int,
    micro_time_resolution_ns: float,
    *,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 40,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    method: str = "nnls",
    reg: float | None = None,
    gate: tuple[int, int] | None = None,
) -> ILTResult1D:
    """Resolve the fluorescence-lifetime distribution from a micro-time decay.

    Builds the decay histogram, an IRF-convolved exponential basis over a log-spaced
    lifetime grid, and solves the regularized inverse-Laplace problem.

    Parameters
    ----------
    micro_times
        Per-photon TCSPC-channel indices.
    n_microtime_bins
        Number of TCSPC channels.
    micro_time_resolution_ns
        ns per TCSPC channel (sets the decay time axis).
    tau_range, n_components
        Lifetime grid (ns) for the inversion.
    irf, irf_time_ns
        Optional instrument response function and its ns axis.
    method, reg
        Passed to :func:`~chisurf.plugins.fcs.flc_2d.fit.ilt.ilt_1d`.
    gate
        Optional ``(lo, hi)`` micro-time channel window to restrict the fit.
    """
    n_bins = int(n_microtime_bins)
    decay = _decay_histogram(micro_times, n_bins)
    if gate is None:
        # With an IRF the rising edge is modelled, so fit the whole gate. Without one we
        # tail-fit from the decay maximum, otherwise the unmodelled rise drives NNLS to
        # zero.
        lo = int(np.argmax(decay)) if irf is None else 0
        hi = n_bins
    else:
        lo, hi = gate
    sl = slice(int(lo), int(hi))
    decay = decay[sl]
    time_ns = (np.arange(n_bins) * micro_time_resolution_ns)[sl]
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    basis = build_exp_basis(time_ns, tau, irf=irf, irf_time_ns=irf_time_ns)
    return ilt_1d(decay, basis, tau, method=method, reg=reg)


def lifetime_lcurve(
    micro_times: np.ndarray,
    n_microtime_bins: int,
    micro_time_resolution_ns: float,
    *,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 40,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    method: str = "nnls",
):
    """Sample the L-curve of the 1D lifetime inversion (for regularization diagnostics).

    Mirrors :func:`lifetime_spectrum`'s decay/basis construction and returns a general
    :class:`chisurf.core.math.regularization.LCurveData` whose corner is the weight the
    auto-selection would pick.
    """
    n_bins = int(n_microtime_bins)
    decay = _decay_histogram(micro_times, n_bins)
    lo = int(np.argmax(decay)) if irf is None else 0
    sl = slice(lo, n_bins)
    decay = decay[sl]
    time_ns = (np.arange(n_bins) * micro_time_resolution_ns)[sl]
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    basis = build_exp_basis(time_ns, tau, irf=irf, irf_time_ns=irf_time_ns)
    return lcurve_1d(decay, basis, method=method)


def species_decay_patterns(
    lifetimes_ns: Sequence[float],
    n_microtime_bins: int,
    micro_time_resolution_ns: float,
    *,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build per-species IRF-convolved decay patterns for fFCS filtering.

    Given the resolved species lifetimes, returns one decay histogram pattern per species
    suitable as ``species_decays`` input to :func:`species_correlation`.
    """
    time_ns = np.arange(int(n_microtime_bins)) * micro_time_resolution_ns
    patterns = []
    for tau in lifetimes_ns:
        col = build_exp_basis(time_ns, np.array([float(tau)]), irf=irf, irf_time_ns=irf_time_ns)[
            :, 0
        ]
        patterns.append(col)
    return patterns


# -------------------------------------------------------------------------- 2D-FDC


def two_d_fdc(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    *,
    dT: float,
    ddT: float,
    tMin: float = 1,
    tMax: float = 4096,
    logt_imax: int = 100,
    lint_bin_factor: int = 1,
    max_bins: int | None = None,
    build_lin: bool = True,
    n_chunks: int | None = None,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """Build linear and log-binned 2D fluorescence-decay correlation matrices.

    ``macro_times`` must be integer clock ticks, ascending; ``micro_times`` are TCSPC
    channel indices. ``dT``/``ddT`` are the correlation lag and window in macro ticks;
    ``tMin``/``tMax`` gate the micro-time range (channels).

    Parameters
    ----------
    macro_times, micro_times
        Photon stream: macro ticks (ascending) and micro TCSPC channels.
    dT, ddT
        Correlation lag and full window width (macro ticks).
    tMin, tMax
        Micro-time gate (channels).
    logt_imax
        Number of log-spaced bins for the log matrix.
    lint_bin_factor
        Micro-time bin factor for the linear matrix (>=1). Build a coarser matrix
        directly instead of a giant full-resolution one (much faster + less memory).
    max_bins
        If given, ``lint_bin_factor`` is derived so the linear matrix is at most this
        many bins. Overrides ``lint_bin_factor``.
    build_lin
        Build the linear matrix (set ``False`` to build only the log-binned matrix).
    n_chunks
        Number of parallel photon chunks (default: one per CPU thread).
    progress_callback
        Optional callable invoked with a 0..1 progress fraction.

    Returns a dict with ``mat_lin``, ``mat_lin_t``, ``mat_log``, ``mat_log_t``.
    """
    if max_bins is not None:
        span = max(1, int(round(tMax)) - int(round(tMin)))
        lint_bin_factor = max(1, -(-span // int(max_bins)))  # ceil(span / max_bins)
    creator = TwoDFDCreator()
    mat_lin, mat_lin_t, mat_log, mat_log_t = creator.create_2d_fdc(
        macro_times,
        micro_times,
        dT=dT,
        ddT=ddT,
        tMin=tMin,
        tMax=tMax,
        logt_imax=logt_imax,
        lint_bin_factor=lint_bin_factor,
        build_lin=build_lin,
        n_chunks=n_chunks,
        progress_callback=progress_callback,
    )
    return {"mat_lin": mat_lin, "mat_lin_t": mat_lin_t, "mat_log": mat_log, "mat_log_t": mat_log_t}


def two_d_fdc_scan(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    dT_ticks: Sequence[int],
    *,
    ddT: float,
    tMin: float = 1,
    tMax: float = 4096,
    logt_imax: int = 60,
    n_chunks: int | None = None,
) -> dict[str, np.ndarray]:
    """Build a log-binned 2D-FDC matrix at many lags in a single photon pass.

    This is the efficient way to watch the 2D-FLC cross-peaks evolve with macro-time lag
    (e.g. for kinetics): the photon stream is traversed once and every reference photon
    visits all lag windows.

    Parameters
    ----------
    macro_times, micro_times
        Photon stream (macro ticks ascending, micro TCSPC channels).
    dT_ticks
        Lag values (macro ticks) to evaluate.
    ddT
        Lag-window half-width source (macro ticks).
    tMin, tMax
        Micro-time gate (channels).
    logt_imax
        Number of log-spaced bins per matrix.
    n_chunks
        Number of parallel photon chunks (default: one per CPU thread).

    Returns a dict with ``matrices`` (``n_lags x L x L``) and ``dT_ticks``.
    """
    from numba import get_num_threads

    from .core import _fdc_scan_log_kernel

    macro = np.ascontiguousarray(macro_times, dtype=np.int64)
    micro = np.ascontiguousarray(micro_times, dtype=np.int64)
    if macro.shape[0] >= 2 and np.any(macro[1:] < macro[:-1]):
        raise ValueError("macro_times must be sorted ascending")
    lags = np.ascontiguousarray(np.asarray(dT_ticks, dtype=np.int64))
    if n_chunks is None:
        try:
            n_chunks = int(get_num_threads())
        except Exception:  # pragma: no cover
            n_chunks = 1
    mats = _fdc_scan_log_kernel(
        macro,
        micro,
        lags,
        np.int64(round(ddT)),
        np.int64(round(tMin)),
        np.int64(round(tMax)),
        int(logt_imax),
        int(n_chunks),
    )
    return {"matrices": mats, "dT_ticks": lags}


# keep the historical name as a thin alias
def correlate_tttr(
    macro_times,
    micro_times,
    dT=0.1,
    ddT=0.05,
    tMin=1.0,
    tMax=12.0,
    logt_imax=100,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """Build a 2D-FDC matrix (deprecated alias for :func:`two_d_fdc`)."""
    return two_d_fdc(
        macro_times,
        micro_times,
        dT=dT,
        ddT=ddT,
        tMin=tMin,
        tMax=tMax,
        logt_imax=logt_imax,
        progress_callback=progress_callback,
    )


def _rebin_square(matrix: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    """Block-sum a square matrix down to about ``target`` bins; return (matrix, factor)."""
    n = matrix.shape[0]
    k = max(1, n // max(1, target))
    m = (n // k) * k
    if k == 1:
        return matrix[:m, :m], 1
    return matrix[:m, :m].reshape(m // k, k, m // k, k).sum(axis=(1, 3)), k


def two_d_spectrum(
    matrix: np.ndarray,
    time_axis_ns: np.ndarray,
    *,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 24,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    method: str = "tikhonov",
    reg: float | None = None,
    max_bins: int = 80,
) -> ILTResult2D:
    """Invert a 2D-FDC matrix into a 2D lifetime distribution ``P``.

    The matrix is block-rebinned to at most ``max_bins`` for tractability, an
    IRF-convolved basis is built on the (rebinned) time axis, and the regularized 2D
    inverse-Laplace problem ``M = E P E.T`` is solved. ``P``'s diagonal is the marginal
    lifetime spectrum; its off-diagonal encodes lifetime exchange.
    """
    M = np.asarray(matrix, dtype=float)
    t = np.asarray(time_axis_ns, dtype=float)
    M2, k = _rebin_square(M, max_bins)
    if k > 1:
        m = (t.size // k) * k
        t = t[:m].reshape(-1, k).mean(axis=1)
    t = t[: M2.shape[0]]
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    basis = build_exp_basis(t, tau, irf=irf, irf_time_ns=irf_time_ns)
    return ilt_2d(M2, basis, tau, method=method, reg=reg)


# alias requested by the manifest / older callers
def fit_tikhonov_2d(matrix, time_axis_ns, **kw) -> ILTResult2D:
    """Alias for :func:`two_d_spectrum` with ``method='tikhonov'``."""
    kw.setdefault("method", "tikhonov")
    return two_d_spectrum(matrix, time_axis_ns, **kw)


def fit_mem_2d(matrix, time_axis_ns, **kw) -> ILTResult2D:
    """2D lifetime inversion via the maximum-entropy method (faithful, slower).

    Delegates to :mod:`chisurf.plugins.fcs.flc_2d.fit.mem_2d`. See
    :func:`two_d_spectrum` for the fast default.
    """
    from .fit.mem_2d import solve_mem_2d

    M = np.asarray(matrix, dtype=float)
    t = np.asarray(time_axis_ns, dtype=float)
    max_bins = kw.pop("max_bins", 80)
    M2, k = _rebin_square(M, max_bins)
    if k > 1:
        m = (t.size // k) * k
        t = t[:m].reshape(-1, k).mean(axis=1)
    t = t[: M2.shape[0]]
    tau_range = kw.pop("tau_range", (0.3, 8.0))
    n_components = kw.pop("n_components", 24)
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    basis = build_exp_basis(
        t, tau, irf=kw.pop("irf", None), irf_time_ns=kw.pop("irf_time_ns", None)
    )
    return solve_mem_2d(M2, basis, tau, **kw)


# ------------------------------------------------------------------------- dynamics


@dataclass
class DynamicsResult:
    """Species-resolved correlation plus fitted interconversion relaxation."""

    correlation: SpeciesCorrelation
    relaxation: dict[str, float]  # from fit_relaxation on the mean species auto-corr


def species_correlation(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    species_decays: Sequence[np.ndarray],
    total_decay: np.ndarray | None,
    macro_time_resolution_s: float,
    *,
    n_microtime_bins: int | None = None,
    n_bins: int = 8,
    n_casc: int = 25,
    fit: bool = True,
) -> DynamicsResult:
    """Species-resolved (lifetime-filtered) correlation and its relaxation time.

    Constructs fFCS filters from the species decay patterns, computes the species auto-
    and cross-correlations with ``tttrlib``, and (optionally) fits a single-exponential
    relaxation to the mean species auto-correlation. For a two-state exchange the fitted
    rate equals the sum of the interconversion rates.

    Parameters
    ----------
    macro_times, micro_times
        Photon stream (macro ticks ascending, micro TCSPC channels).
    species_decays
        Per-species decay patterns (see :func:`species_decay_patterns`).
    total_decay
        Total decay histogram; if ``None`` it is built from ``micro_times``.
    macro_time_resolution_s
        Seconds per macro tick (lag-axis calibration).
    n_microtime_bins
        Number of TCSPC channels (defaults to the longest species pattern).
    n_bins, n_casc
        Multi-tau correlator settings (channels per cascade, number of cascades).
    fit
        Fit a single-exponential relaxation to the mean species auto-correlation.
    """
    n_mt = n_microtime_bins or max(len(d) for d in species_decays)
    if total_decay is None:
        total_decay = _decay_histogram(micro_times, n_mt)
    filters = species_filters(total_decay, species_decays)
    corr = filtered_correlation(
        macro_times,
        micro_times,
        filters,
        macro_time_resolution_s,
        n_bins=n_bins,
        n_casc=n_casc,
        n_microtime_bins=n_mt,
    )
    relax: dict[str, float] = {}
    if fit and corr.auto:
        mean_auto = np.mean(np.vstack([corr.auto[i] for i in corr.auto]), axis=0)
        try:
            relax = fit_relaxation(corr.lag_s, mean_auto)
        except Exception:  # pragma: no cover - degenerate data
            relax = {}
    return DynamicsResult(correlation=corr, relaxation=relax)


def global_lifetime_mem(
    matrices: Sequence[np.ndarray],
    time_axis_ns: np.ndarray,
    *,
    n_states: int = 2,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 20,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    regulator: float = 1.0,
) -> GlobalMEMResult:
    """Jointly invert several lag matrices with one shared lifetime distribution.

    Builds the IRF-convolved basis once and runs the global multi-lag 2D-MEM
    (:func:`~chisurf.plugins.fcs.flc_2d.fit.global_mem.solve_global_mem_2d`). All matrices
    must share the same shape and time axis (e.g. linear 2D-FDCs built at different ``dT``
    with the same ``tMin``/``tMax``/binning).
    """
    mats = [np.asarray(M, dtype=float) for M in matrices]
    n = mats[0].shape[0]
    t = np.asarray(time_axis_ns, dtype=float)[:n]
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    basis = build_exp_basis(t, tau, irf=irf, irf_time_ns=irf_time_ns)
    return solve_global_mem_2d(mats, basis, tau, n_states=n_states, regulator=regulator)


def rate_matrix_kinetics(
    correlation: SpeciesCorrelation,
    *,
    n_states: int = 2,
    populations: Sequence[float] | None = None,
    t_min: float = 5e-4,
    t_max: float = 0.5,
) -> RateMatrixResult:
    """Fit a rate matrix to species correlation decays (per-state rate constants).

    Builds the ``{(i, j): G_ij}`` curve set from a :class:`SpeciesCorrelation` (auto +
    cross) and runs the shared-rate variable-projection fit
    (:func:`~chisurf.plugins.fcs.flc_2d.fit.kinetics.fit_rate_matrix`). For two states the
    reconstructed rate matrix needs the equilibrium ``populations``.
    """
    curves: dict[tuple[int, int], np.ndarray] = {(i, i): g for i, g in correlation.auto.items()}
    curves.update(correlation.cross)
    pops = None if populations is None else np.asarray(populations, dtype=float)
    return fit_rate_matrix(
        correlation.lag_s, curves, n_states=n_states, populations=pops, t_min=t_min, t_max=t_max
    )


# ------------------------------------------------------------------- 1D-FDC + 1D-MEM


def one_d_fdc(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    *,
    tMin: float = 0,
    tMax: float = 4096,
    lint_bin_factor: int = 1,
    max_bins: int | None = None,
    logt_imax: int = 100,
    n_chunks: int = 1,
) -> dict[str, np.ndarray]:
    """Build the 1D-FDC (zero-lag same-macro-time decay coincidence).

    Port of ``TK_Create1DFDC_01``. Returns a dict with ``lin_t``/``lin`` (linear axis and
    1D-FDC) and ``log_t``/``log`` (log axis and 1D-FDC), in micro-time tick units.

    Parameters
    ----------
    macro_times, micro_times
        Photon stream (macro ticks ascending, micro TCSPC channels).
    tMin, tMax
        Micro-time gate (channels).
    lint_bin_factor
        Linear micro-time bin factor (>=1).
    max_bins
        If given, derive ``lint_bin_factor`` so the linear axis is at most this many bins.
    logt_imax
        Number of log-spaced bins.
    n_chunks
        Number of parallel photon chunks.
    """
    lo, hi = int(round(tMin)), int(round(tMax))
    if max_bins is not None:
        span = max(1, hi - lo)
        lint_bin_factor = max(1, -(-span // int(max_bins)))
    lin_t, lin, log_t, log = create_1d_fdc(
        macro_times,
        micro_times,
        tMin_ticks=lo,
        tMax_ticks=hi,
        lint_bin_factor=max(1, int(lint_bin_factor)),
        logt_imax=int(logt_imax),
        n_chunks=int(n_chunks),
    )
    return {"lin_t": lin_t, "lin": lin, "log_t": log_t, "log": log}


def _fit_1d_decay(
    decay: np.ndarray,
    time_ns: np.ndarray,
    tau: np.ndarray,
    *,
    irf: np.ndarray | None,
    irf_time_ns: np.ndarray | None,
):
    """Drop leading non-positive bins (the MATLAB ``FitStartI``) and build the basis."""
    decay = np.asarray(decay, dtype=float)
    pos = np.flatnonzero(decay > 0)
    start = int(pos[0]) if pos.size else 0
    decay = decay[start:]
    t = np.asarray(time_ns, dtype=float)[start:]
    basis = build_exp_basis(t, tau, irf=irf, irf_time_ns=irf_time_ns)
    return decay, t, basis


def lifetime_spectrum_mem(
    decay: np.ndarray,
    time_ns: np.ndarray,
    *,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 40,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    reg: float = 50.0,
    mi_type: int = 0,
    n_outer: int = 12,
) -> OneDMEMResult:
    """Resolve a 1D lifetime distribution by maximum entropy (faithful MATLAB objective).

    This is the explicit 1D-MEM (port of ``TK_FitF_1DMEM_*``); the fast default for routine
    use remains :func:`lifetime_spectrum` (NNLS/Tikhonov ILT). Leading empty bins are
    dropped automatically (the MATLAB ``FitStartI``).

    Parameters
    ----------
    decay
        1D fluorescence-decay (e.g. ``one_d_fdc(...)["lin"]`` or a micro-time histogram).
    time_ns
        Decay time axis (ns).
    tau_range, n_components
        Lifetime grid (ns).
    irf, irf_time_ns
        Optional instrument response and its ns axis.
    reg, mi_type, n_outer
        Passed to :func:`~chisurf.plugins.fcs.flc_2d.fit.mem_1d.solve_mem_1d`.
    """
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    d, t, basis = _fit_1d_decay(decay, time_ns, tau, irf=irf, irf_time_ns=irf_time_ns)
    return solve_mem_1d(
        d,
        basis,
        tau,
        reg=reg,
        mi_type=mi_type,
        n_outer=n_outer,
        t_min=float(t[0]) if t.size else 0.0,
        t_max=float(t[-1]) if t.size else 1.0,
    )


def fit_gaussian_components(
    tau_grid: np.ndarray,
    distribution: np.ndarray,
    n_components: int = 2,
    *,
    fit_offset: bool = False,
) -> GaussianFitResult:
    """Fit discrete Gaussian peaks to a recovered lifetime distribution.

    Port of ``TK_FitF_GaussianMulti``: extract discrete lifetime components (centre, width,
    amplitude) from a smooth ILT/MEM distribution.
    """
    return fit_gaussian_multi(
        np.asarray(tau_grid, float),
        np.asarray(distribution, float),
        n_components,
        fit_offset=fit_offset,
    )


def search_irf_rise(
    decay: np.ndarray,
    time_ns: np.ndarray,
    irf: np.ndarray,
    irf_time_ns: np.ndarray,
    *,
    tau_range: tuple[float, float] = (0.3, 8.0),
    n_components: int = 40,
    shifts=range(-7, 8),
    average_width: int = 13,
    mem_kwargs: dict | None = None,
) -> RiseIRFResult:
    """Scan the IRF rise position and average the 1D-MEM around the optimum.

    Port of ``TK_MyMain_Search_RiseIRF_1DMEM``. Leading empty bins are dropped first.
    """
    tau = lifetime_grid(tau_range[0], tau_range[1], n_components)
    d, t, _ = _fit_1d_decay(decay, time_ns, tau, irf=None, irf_time_ns=None)
    return search_rise_irf(
        d,
        t,
        tau,
        irf,
        irf_time_ns,
        shifts=shifts,
        average_width=average_width,
        mem_kwargs=mem_kwargs,
    )


def make_synthetic_irf(
    time_ns: np.ndarray,
    center_ns: float,
    fwhm_ns: float,
    *,
    shape: float = 0.0,
) -> np.ndarray:
    """Build a synthetic (skewed-)Gaussian IRF on ``time_ns``.

    Thin re-export of the general
    :func:`chisurf.core.fluorescence.tcspc.irf.synthetic_irf` so the plugin reuses the
    shared chisurf IRF helpers instead of duplicating them.
    """
    from chisurf.core.fluorescence.tcspc.irf import synthetic_irf

    return synthetic_irf(time_ns, center_ns, fwhm_ns, shape=shape)


def detect_irf(
    decay: np.ndarray,
    time_ns: np.ndarray,
    *,
    fwhm_ns: float | None = None,
    shape: float = 0.0,
) -> np.ndarray:
    """Detect a decay's prompt position and return a matching synthetic IRF.

    Thin re-export of
    :func:`chisurf.core.fluorescence.tcspc.irf.estimate_irf_from_decay`.
    """
    from chisurf.core.fluorescence.tcspc.irf import estimate_irf_from_decay

    return estimate_irf_from_decay(decay, time_ns, fwhm_ns=fwhm_ns, shape=shape)


def simulate_stream(
    rate_matrix: np.ndarray,
    lifetimes_ns: Sequence[float],
    intensities_cps: Sequence[float],
    *,
    total_time_s: float = 100.0,
    irf: np.ndarray | None = None,
    irf_time_ns: np.ndarray | None = None,
    macro_time_resolution_s: float = 1e-6,
    tstep_ns: float = 0.004,
    n_microtime_channels: int = 3127,
    seed: int = 0,
):
    """Simulate a single-molecule photon stream from an n-state exchange process.

    Thin wrapper around
    :func:`~chisurf.plugins.fcs.flc_2d.simulate.simulate_photon_stream` (port of
    ``TK_MyMain_Simu_PhotonStream``). Returns a ``SimulatedStream`` with macro/micro ticks
    and ground-truth state labels, closing the loop for validation.
    """
    from .simulate import simulate_photon_stream

    return simulate_photon_stream(
        np.asarray(rate_matrix, dtype=float),
        lifetimes_ns,
        intensities_cps,
        total_time_s=total_time_s,
        irf=irf,
        irf_time_ns=irf_time_ns,
        macro_time_resolution_s=macro_time_resolution_s,
        tstep_ns=tstep_ns,
        n_microtime_channels=n_microtime_channels,
        seed=seed,
    )
