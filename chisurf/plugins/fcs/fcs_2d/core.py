"""
Core 2D-FDC (Fluorescence Decay Correlation) matrix creation functions.

This module implements the 2D-FDC matrix creation algorithm from MATLAB code
TK_Create2DFDC_04.m, adapted for Python/ChiSurf integration.
"""

import numpy as np
from typing import Tuple, Optional, Callable
import logging

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - hard failure when numba missing
    raise ImportError(
        "chisurf.plugins.fcs.fcs_2d.core requires numba. "
        "Install numba to use the 2D-FCS module."
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
    """
    Find logarithmic bin index for integer tau.
    Returns -1 if out of range.
    """
    idx = np.searchsorted(logt_ticks, tau_ticks, side="left")
    if idx <= 0:
        return -1
    if idx >= logt_ticks.shape[0]:
        return -1
    return idx - 1


@njit(cache=True)
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated 2D-FDC creation using integer arithmetic."""
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

        mat_2dfdc_lin = np.zeros((lint_imax, lint_imax), dtype=np.int64)
        mat_2dfdc_lint = (lint_bin_factor) * np.arange(lint_imax, dtype=np.int64)
        mat_2dfdc_log = np.zeros((logt_imax, logt_imax), dtype=np.int64)

        logt_ticks = np.empty(logt_imax, dtype=np.int64)
        logt_ticks[0] = -1
        for j in range(1, logt_imax):
            x = j / (logt_imax - 1)
            v = (t_imax ** x) - 1.0
            if v < -9.22e18:
                logt_ticks[j] = -9223372036854775808
            elif v > 9.22e18:
                logt_ticks[j] = 9223372036854775807
            else:
                logt_ticks[j] = int(np.floor(v + 0.5))

        for i in range(n):
            ti = macro_times[i]
            if ti < Tstart_ticks or ti > Tend_ticks:
                continue

            tau_i = micro_times[i] - tMin_over_tStep
            if tau_i <= 0 or tau_i >= t_imax:
                continue

            lint_i = _ceil_div_pos(tau_i, lint_bin_factor)
            logt_i = _log_bin_int(tau_i, logt_ticks)

            half = ddT_ticks // 2
            dt_start = ti + dT_ticks - half
            dt_end = ti + dT_ticks + half

            if dt_end > macro_times[n - 1] or dt_end > Tend_ticks:
                break

            k_start = np.searchsorted(macro_times, dt_start, side="left")
            k_end = np.searchsorted(macro_times, dt_end, side="right")

            for k in range(k_start, k_end):
                tau_k = micro_times[k] - tMin_over_tStep
                if tau_k <= 0 or tau_k >= t_imax:
                    continue

                lint_k = _ceil_div_pos(tau_k, lint_bin_factor)
                if lint_i < lint_imax and lint_k < lint_imax:
                    mat_2dfdc_lin[lint_i, lint_k] += 1

                logt_k = _log_bin_int(tau_k, logt_ticks)
                if (logt_i > 0) and (logt_k > 0) and (logt_i < logt_imax) and (logt_k < logt_imax):
                    mat_2dfdc_log[logt_i, logt_k] += 1

        var_size = mat_2dfdc_lin.shape[0] - 1
        mat_2dfdc_lin = mat_2dfdc_lin[:var_size, :var_size]
        mat_2dfdc_lint = mat_2dfdc_lint[:var_size]

        mat_2dfdc_log = mat_2dfdc_log[: logt_imax - 1, : logt_imax - 1]
        logt_ticks = logt_ticks[: logt_imax - 1]

        return mat_2dfdc_lin, mat_2dfdc_lint, mat_2dfdc_log, logt_ticks


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
        ):
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
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 2D-FDC matrices from photon arrival times.
        
        This method computes both linear and logarithmically binned 2D-FDC matrices.
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
            logt_imax=logt_imax
        )
        
        self.logger.info(
            "2D-FDC create_2d_fdc called | backend=numba photons=%d dT=%d ddT=%d "
            "tMin=%d tMax=%d logt_imax=%d",
            params['n_photons'],
            params['dT_ticks'],
            params['ddT_ticks'],
            params['tMin_ticks'],
            params['tMax_ticks'],
            params['logt_imax']
        )
        print(
            "[2D-FLCS][Core] Starting 2D-FDC (numba) | "
            f"photons={params['n_photons']} dT={params['dT_ticks']} "
            f"ddT={params['ddT_ticks']} tMin={params['tMin_ticks']} "
            f"tMax={params['tMax_ticks']}"
        )
        
        return self._create_with_numba(params, progress_callback)
    
    def _create_with_numba(self, params: dict, progress_callback: Optional[callable]):
        if progress_callback:
            progress_callback(0.0)
        
        mat_lin, mat_lin_t, mat_log, mat_log_t = self._numba_creator.create_2d_fdc(
            macro_times=params['macro_times'],
            micro_times=params['micro_times'],
            dT_ticks=params['dT_ticks'],
            ddT_ticks=params['ddT_ticks'],
            Tstart_ticks=params['Tstart_ticks'],
            Tend_ticks=params['Tend_ticks'],
            tMin_over_tStep=params['tMin_over_tStep'],
            tMax_over_tStep=params['tMax_over_tStep'],
            lint_bin_factor=1,
            logt_imax=params['logt_imax']
        )
        
        step = 1
        mat_lin_t = (mat_lin_t * step).astype(np.int64, copy=False)
        mat_log_t = (mat_log_t * step).astype(np.int64, copy=False)
        
        if progress_callback:
            progress_callback(1.0)
        
        self.logger.info("2D-FDC creation complete via Numba. Processed %d photons.", params['n_photons'])
        print(f"[2D-FLCS][Core] Completed 2D-FDC creation (numba). Processed photons: {params['n_photons']}")
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
            'macro_times': macro_times_int,
            'micro_times': micro_times_int,
            'n_photons': macro_times_int.shape[0],
            'dT_ticks': self._ensure_int_value(dT, "dT", positive=True),
            'ddT_ticks': self._ensure_int_value(ddT, "ddT", positive=True),
            'Tstart_ticks': int(macro_times_int[0]),
            'Tend_ticks': int(macro_times_int[-1]),
            'tMin_ticks': tMin_ticks,
            'tMax_ticks': tMax_ticks,
            'logt_imax': int(max(2, logt_imax)),
        }
        
        params['tMin_over_tStep'] = tMin_ticks
        params['tMax_over_tStep'] = tMax_ticks
        
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
        minimum: Optional[int] = None
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
        tMax: float = 12.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create short-delay 2D-FDC matrix (dT = minimum macro difference).
        """
        macro_arr = self._ensure_int_array(macro_times, "macro_times")
        min_delay = np.min(np.diff(macro_arr))
        return self.create_2d_fdc(
            macro_arr,
            self._ensure_int_array(micro_times, "micro_times"),
            dT=min_delay,
            ddT=max(1, min_delay // 2),
            tMin=tMin,
            tMax=tMax
        )[:2]
