"""
Core 2D-FDC (Fluorescence Decay Correlation) matrix creation functions.

This module implements the 2D-FDC matrix creation algorithm from MATLAB code
TK_Create2DFDC_04.m, adapted for Python/ChiSurf integration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

try:
    from numba import get_num_threads, njit, prange
except ImportError as exc:  # pragma: no cover - hard failure when numba missing
    raise ImportError(
        "chisurf.plugins.fcs.flc_2d.core requires numba. Install numba to use the 2D-FCS module."
    ) from exc


@njit(cache=True)
def _ceil_div_pos(a: int, b: int) -> int:
    """Ceil division for positive integers."""
    return (a + b - 1) // b


@njit(cache=True)
def _ceil_div_signed(a: int, b: int) -> int:
    """Safe ceil division for signed numerator."""
    if a >= 0:
        return (a + b - 1) // b
    return -((-a) // b)


@njit(cache=True)
def _log_bin_int(tau_ticks: int, logt_ticks: np.ndarray) -> int:
    """Find the logarithmic bin index for an integer tau.

    Returns -1 if out of range.
    """
    idx = np.searchsorted(logt_ticks, tau_ticks, side="left")
    if idx <= 0:
        return -1
    if idx >= logt_ticks.shape[0]:
        return -1
    return idx - 1


@njit(cache=True, parallel=True)
def create_2d_fdc_numba_int(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    dT_ticks: int,
    ddT_ticks: int,
    Tstart_ticks: int,
    Tend_ticks: int,
    tMin_over_tStep: int,
    tMax_over_tStep: int,
    lint_bin_factor: int = 2,
    logt_imax_in: int = 100,
    build_lin: bool = True,
    n_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated 2D-FDC creation using integer arithmetic.

    The reference-photon loop is split into ``n_chunks`` independent chunks, each
    accumulating into a private matrix (no write races), run in parallel with
    ``prange`` and reduced at the end. ``lint_bin_factor`` bins the micro-time axis at
    construction time; ``build_lin=False`` skips the (potentially large) linear matrix
    when only the log-binned one is needed.
    """
    n = macro_times.shape[0]
    if n == 0:
        raise ValueError("Input arrays cannot be empty")
    if micro_times.shape[0] != n:
        raise ValueError("macro_times and micro_times must have same length")

    logt_imax = logt_imax_in + 1

    span_ticks = tMax_over_tStep - tMin_over_tStep
    if span_ticks < 0:
        raise ValueError("tMax_over_tStep must be >= tMin_over_tStep")

    t_imax0 = span_ticks + lint_bin_factor
    lint_imax = _ceil_div_pos(t_imax0, lint_bin_factor)
    t_imax = lint_imax * lint_bin_factor

    mat_2dfdc_lint = (lint_bin_factor) * np.arange(lint_imax, dtype=np.int64)

    logt_ticks = np.empty(logt_imax, dtype=np.int64)
    logt_ticks[0] = -1
    for j in range(1, logt_imax):
        x = j / (logt_imax - 1)
        v = (t_imax**x) - 1.0
        if v < -9.22e18:
            logt_ticks[j] = -9223372036854775808
        elif v > 9.22e18:
            logt_ticks[j] = 9223372036854775807
        else:
            logt_ticks[j] = int(np.floor(v + 0.5))

    nc = n_chunks if n_chunks >= 1 else 1
    llin = lint_imax if build_lin else 1
    acc_lin = np.zeros((nc, llin, llin), dtype=np.int64)
    acc_log = np.zeros((nc, logt_imax, logt_imax), dtype=np.int64)
    half = ddT_ticks // 2

    for c in prange(nc):
        i0 = (n * c) // nc
        i1 = (n * (c + 1)) // nc
        for i in range(i0, i1):
            ti = macro_times[i]
            if ti < Tstart_ticks or ti > Tend_ticks:
                continue

            tau_i = micro_times[i] - tMin_over_tStep
            if tau_i <= 0 or tau_i >= t_imax:
                continue

            lint_i = _ceil_div_pos(tau_i, lint_bin_factor)
            logt_i = _log_bin_int(tau_i, logt_ticks)

            dt_start = ti + dT_ticks - half
            dt_end = ti + dT_ticks + half
            if dt_end > macro_times[n - 1] or dt_end > Tend_ticks:
                break  # photons are sorted: the rest of this chunk also overflow

            k_start = np.searchsorted(macro_times, dt_start, side="left")
            k_end = np.searchsorted(macro_times, dt_end, side="right")

            for k in range(k_start, k_end):
                tau_k = micro_times[k] - tMin_over_tStep
                if tau_k <= 0 or tau_k >= t_imax:
                    continue

                if build_lin:
                    lint_k = _ceil_div_pos(tau_k, lint_bin_factor)
                    if lint_i < lint_imax and lint_k < lint_imax:
                        acc_lin[c, lint_i, lint_k] += 1

                logt_k = _log_bin_int(tau_k, logt_ticks)
                if (logt_i > 0) and (logt_k > 0) and (logt_i < logt_imax) and (logt_k < logt_imax):
                    acc_log[c, logt_i, logt_k] += 1

    mat_2dfdc_lin = acc_lin[0].copy()
    mat_2dfdc_log = acc_log[0].copy()
    for c in range(1, nc):
        mat_2dfdc_lin += acc_lin[c]
        mat_2dfdc_log += acc_log[c]

    var_size = mat_2dfdc_lin.shape[0] - 1
    mat_2dfdc_lin = mat_2dfdc_lin[:var_size, :var_size]
    mat_2dfdc_lint = mat_2dfdc_lint[:var_size]

    mat_2dfdc_log = mat_2dfdc_log[: logt_imax - 1, : logt_imax - 1]
    logt_ticks = logt_ticks[: logt_imax - 1]

    return mat_2dfdc_lin, mat_2dfdc_lint, mat_2dfdc_log, logt_ticks


@njit(cache=True, parallel=True)
def _fdc_scan_log_kernel(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    dT_ticks: np.ndarray,
    ddT_ticks: int,
    tMin_over_tStep: int,
    tMax_over_tStep: int,
    logt_imax_in: int,
    n_chunks: int,
) -> np.ndarray:
    """Build one log-binned 2D-FDC matrix per lag in ``dT_ticks`` in a single photon pass.

    Returns an array of shape ``(n_lags, L, L)`` (``L = logt_imax_in``). Each reference
    photon visits every lag window once, so the photon stream is traversed only once.
    """
    n = macro_times.shape[0]
    n_lags = dT_ticks.shape[0]
    logt_imax = logt_imax_in + 1
    span = tMax_over_tStep - tMin_over_tStep
    t_imax = span + 1

    logt_ticks = np.empty(logt_imax, dtype=np.int64)
    logt_ticks[0] = -1
    for j in range(1, logt_imax):
        x = j / (logt_imax - 1)
        v = (t_imax**x) - 1.0
        if v < -9.22e18:
            logt_ticks[j] = -9223372036854775808
        elif v > 9.22e18:
            logt_ticks[j] = 9223372036854775807
        else:
            logt_ticks[j] = int(np.floor(v + 0.5))

    nc = n_chunks if n_chunks >= 1 else 1
    half = ddT_ticks // 2
    last = macro_times[n - 1]
    acc = np.zeros((nc, n_lags, logt_imax, logt_imax), dtype=np.int64)

    for c in prange(nc):
        i0 = (n * c) // nc
        i1 = (n * (c + 1)) // nc
        for i in range(i0, i1):
            ti = macro_times[i]
            tau_i = micro_times[i] - tMin_over_tStep
            if tau_i <= 0 or tau_i >= t_imax:
                continue
            logt_i = _log_bin_int(tau_i, logt_ticks)
            if logt_i <= 0 or logt_i >= logt_imax:
                continue
            for li in range(n_lags):
                dt_start = ti + dT_ticks[li] - half
                dt_end = ti + dT_ticks[li] + half
                if dt_end > last:
                    continue
                k_start = np.searchsorted(macro_times, dt_start, side="left")
                k_end = np.searchsorted(macro_times, dt_end, side="right")
                for k in range(k_start, k_end):
                    tau_k = micro_times[k] - tMin_over_tStep
                    if tau_k <= 0 or tau_k >= t_imax:
                        continue
                    logt_k = _log_bin_int(tau_k, logt_ticks)
                    if 0 < logt_k < logt_imax:
                        acc[c, li, logt_i, logt_k] += 1

    out = acc[0].copy()
    for c in range(1, nc):
        out += acc[c]
    return out[:, : logt_imax - 1, : logt_imax - 1]


class TwoDFDCreatorNumbaInt:
    """Thin wrapper around numba kernel to enforce int64 inputs."""

    def create_2d_fdc(
        self,
        macro_times: np.ndarray,
        micro_times: np.ndarray,
        dT_ticks: int,
        ddT_ticks: int,
        Tstart_ticks: int = 0,
        Tend_ticks: int = 2**63 - 1,
        tMin_over_tStep: int = 0,
        tMax_over_tStep: int = 4096,
        lint_bin_factor: int = 2,
        logt_imax: int = 100,
        build_lin: bool = True,
        n_chunks: int = 1,
    ):
        """Build the linear and log 2D-FDC matrices via the numba kernel."""
        macro_times = np.ascontiguousarray(macro_times, dtype=np.int64)
        micro_times = np.ascontiguousarray(micro_times, dtype=np.int64)

        if macro_times.shape[0] >= 2 and np.any(macro_times[1:] < macro_times[:-1]):
            raise ValueError("macro_times must be sorted ascending for numba implementation")

        return create_2d_fdc_numba_int(
            macro_times=macro_times,
            micro_times=micro_times,
            dT_ticks=np.int64(dT_ticks),
            ddT_ticks=np.int64(ddT_ticks),
            Tstart_ticks=np.int64(Tstart_ticks),
            Tend_ticks=np.int64(Tend_ticks),
            tMin_over_tStep=np.int64(tMin_over_tStep),
            tMax_over_tStep=np.int64(tMax_over_tStep),
            lint_bin_factor=int(lint_bin_factor),
            logt_imax_in=int(logt_imax),
            build_lin=bool(build_lin),
            n_chunks=int(n_chunks),
        )


class TwoDFDCreator:
    """
    Creates 2D-Fluorescence Decay Correlation (FDC) matrices from TTTR data.

    This class provides a high-level interface for generating 2D-FDC matrices,
    which represent the correlation between photon microtimes (arrival times within
    a laser cycle) separated by a specific macro-time delay (dT).

    The implementation uses Numba for high-performance correlation calculation.
    """

    def __init__(self, prefer_numba: bool = True):
        """
        Initialize the 2D-FDC creator.

        Args:
            prefer_numba: Whether to use the Numba-accelerated backend (default: True).
        """
        self.logger = logging.getLogger(__name__)
        self._numba_creator = TwoDFDCreatorNumbaInt()

    def create_2d_fdc(
        self,
        macro_times: np.ndarray,
        micro_times: np.ndarray,
        dT: float = 0.1,
        ddT: float = 0.05,
        tMin: float = 1.0,
        tMax: float = 12.0,
        logt_imax: int = 100,
        lint_bin_factor: int = 1,
        build_lin: bool = True,
        n_chunks: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 2D-FDC matrices from photon arrival times.

        This method computes linear and logarithmically binned 2D-FDC matrices.
        The input parameters dT, ddT, tMin, and tMax are expected to be in tick units
        (integers) if raw TTTR data is used, or in physical units if normalized.

        Args:
            macro_times: Array of macro-time arrival ticks.
            micro_times: Array of micro-time arrival ticks.
            dT: Macro-time delay (ticks or seconds).
            ddT: Macro-time window width (ticks or seconds).
            tMin: Lower micro-time gate (ticks or seconds).
            tMax: Upper micro-time gate (ticks or seconds).
            logt_imax: Number of points for the logarithmic time axis.
            lint_bin_factor: Micro-time bin factor for the linear matrix (>=1). Larger
                factors build a smaller matrix directly, avoiding a giant full-resolution
                matrix and speeding up both construction and the downstream fit.
            build_lin: Build the linear matrix (set False to build only the log matrix).
            n_chunks: Number of parallel photon chunks (default: one per thread). Memory
                scales with ``n_chunks * matrix_size``; capped automatically.
            progress_callback: Optional function called with progress (0.0 to 1.0).

        Returns:
            Tuple containing:
            - mat_lin: Linear 2D-FDC matrix (n_tau x n_tau).
            - mat_lin_t: Time axis for linear matrix (ticks or seconds).
            - mat_log: Logarithmic 2D-FDC matrix (log_points x log_points).
            - mat_log_t: Time axis for logarithmic matrix.
        """
        params = self._normalize_inputs(
            macro_times=macro_times,
            micro_times=micro_times,
            dT=dT,
            ddT=ddT,
            tMin=tMin,
            tMax=tMax,
            logt_imax=logt_imax,
        )
        params["lint_bin_factor"] = max(1, int(lint_bin_factor))
        params["build_lin"] = bool(build_lin)
        params["n_chunks"] = self._pick_n_chunks(params, n_chunks)

        self.logger.info(
            "2D-FDC create_2d_fdc called | backend=numba photons=%d dT=%d ddT=%d "
            "tMin=%d tMax=%d logt_imax=%d bin=%d chunks=%d",
            params["n_photons"],
            params["dT_ticks"],
            params["ddT_ticks"],
            params["tMin_ticks"],
            params["tMax_ticks"],
            params["logt_imax"],
            params["lint_bin_factor"],
            params["n_chunks"],
        )
        return self._create_with_numba(params, progress_callback)

    def _pick_n_chunks(self, params: dict, n_chunks: int | None) -> int:
        """Choose a parallel chunk count, capping accumulator memory to ~512 MB."""
        if n_chunks is not None:
            return max(1, int(n_chunks))
        try:
            threads = int(get_num_threads())
        except Exception:  # pragma: no cover - numba threading layer unavailable
            threads = 1
        span = max(1, params["tMax_over_tStep"] - params["tMin_over_tStep"])
        llin = (span // params["lint_bin_factor"] + 2) if params["build_lin"] else 1
        llog = params["logt_imax"] + 1
        bytes_per_chunk = 8 * (llin * llin + llog * llog)
        max_chunks = max(1, int(512 * 1024 * 1024 / max(1, bytes_per_chunk)))
        return max(1, min(threads, max_chunks))

    def _create_with_numba(self, params: dict, progress_callback: callable | None):
        if progress_callback:
            progress_callback(0.0)

        mat_lin, mat_lin_t, mat_log, mat_log_t = self._numba_creator.create_2d_fdc(
            macro_times=params["macro_times"],
            micro_times=params["micro_times"],
            dT_ticks=params["dT_ticks"],
            ddT_ticks=params["ddT_ticks"],
            Tstart_ticks=params["Tstart_ticks"],
            Tend_ticks=params["Tend_ticks"],
            tMin_over_tStep=params["tMin_over_tStep"],
            tMax_over_tStep=params["tMax_over_tStep"],
            lint_bin_factor=params["lint_bin_factor"],
            logt_imax=params["logt_imax"],
            build_lin=params["build_lin"],
            n_chunks=params["n_chunks"],
        )

        step = 1
        mat_lin_t = (mat_lin_t * step).astype(np.int64, copy=False)
        mat_log_t = (mat_log_t * step).astype(np.int64, copy=False)

        if progress_callback:
            progress_callback(1.0)

        self.logger.info(
            "2D-FDC creation complete via Numba. Processed %d photons.", params["n_photons"]
        )
        return mat_lin, mat_lin_t, mat_log, mat_log_t

    def _normalize_inputs(
        self,
        macro_times: np.ndarray,
        micro_times: np.ndarray,
        dT: float,
        ddT: float,
        tMin: float,
        tMax: float,
        logt_imax: int,
    ) -> dict:
        macro_times_int = self._ensure_int_array(macro_times, "macro_times")
        micro_times_int = self._ensure_int_array(micro_times, "micro_times")

        if macro_times_int.shape[0] != micro_times_int.shape[0]:
            raise ValueError("macro_times and micro_times must have same length")
        if macro_times_int.shape[0] == 0:
            raise ValueError("Input arrays cannot be empty")

        if np.any(macro_times_int[1:] < macro_times_int[:-1]):
            raise ValueError("macro_times must be sorted ascending for 2D-FDC creation")

        tMin_ticks = self._ensure_int_value(tMin, "tMin")
        tMax_ticks = self._ensure_int_value(tMax, "tMax")
        if tMax_ticks <= tMin_ticks:
            raise ValueError("tMax must be greater than tMin after conversion to ticks")

        if macro_times_int.shape[0] < 2:
            raise ValueError("macro_times must contain at least 2 photons for 2D-FDC")

        params = {
            "macro_times": macro_times_int,
            "micro_times": micro_times_int,
            "n_photons": macro_times_int.shape[0],
            "dT_ticks": self._ensure_int_value(dT, "dT", positive=True),
            "ddT_ticks": self._ensure_int_value(ddT, "ddT", positive=True),
            "Tstart_ticks": int(macro_times_int[0]),
            "Tend_ticks": int(macro_times_int[-1]),
            "tMin_ticks": tMin_ticks,
            "tMax_ticks": tMax_ticks,
            "logt_imax": int(max(2, logt_imax)),
        }

        params["tMin_over_tStep"] = tMin_ticks
        params["tMax_over_tStep"] = tMax_ticks

        return params

    def _ensure_int_array(self, array: np.ndarray, name: str) -> np.ndarray:
        arr = np.asanyarray(array)
        if not np.issubdtype(arr.dtype, np.integer):
            self.logger.warning("2D-FDC: %s provided as float; rounding to nearest tick.", name)
            arr = np.rint(arr).astype(np.int64)
        else:
            arr = arr.astype(np.int64, copy=False)
        return np.ascontiguousarray(arr)

    def _ensure_int_value(
        self,
        value: float,
        name: str,
        positive: bool = False,
        allow_zero: bool = False,
        minimum: int | None = None,
    ) -> int:
        if not isinstance(value, (int, np.integer)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            self.logger.debug("2D-FDC: %s=%s rounded to nearest integer tick", name, value)
            value = int(round(value))
        else:
            value = int(value)

        if minimum is not None:
            value = max(value, minimum)

        if positive and value <= 0:
            raise ValueError(f"{name} must be positive in tick units")
        if not allow_zero and value == 0:
            value = 1 if positive else 0

        return value

    def create_short_delay_fdc(
        self,
        macro_times: np.ndarray,
        micro_times: np.ndarray,
        tMin: float = 1.0,
        tMax: float = 12.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a short-delay 2D-FDC matrix (dT = minimum macro difference)."""
        macro_arr = self._ensure_int_array(macro_times, "macro_times")
        min_delay = np.min(np.diff(macro_arr))
        return self.create_2d_fdc(
            macro_arr,
            self._ensure_int_array(micro_times, "micro_times"),
            dT=min_delay,
            ddT=max(1, min_delay // 2),
            tMin=tMin,
            tMax=tMax,
        )[:2]
