"""Core mathematical functions for Jordi G-Factor calculations."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def shift_interp_on_axis(t: np.ndarray, y: np.ndarray, shift: float) -> np.ndarray:
    """Interpolate ``y`` onto the time axis ``t`` after shifting by ``shift``.

    Parameters
    ----------
    t : np.ndarray
        Reference time axis.
    y : np.ndarray
        Intensity values aligned with ``t``.
    shift : float
        Shift to apply to ``t`` before interpolation.

    Returns
    -------
    np.ndarray
        Interpolated values aligned with ``t`` (NaN outside the range).
    """
    if shift == 0.0:
        return np.asarray(y, dtype=float).copy()
    xq = t - shift
    out = np.full_like(y, np.nan, dtype=float)
    mask = (xq >= t[0]) & (xq <= t[-1])
    if np.any(mask):
        out[mask] = np.interp(xq[mask], t, y)
    return out


def compute_rt(par: np.ndarray, perp: np.ndarray, g_factor: float, l1: float = 0.0, l2: float = 0.0) -> np.ndarray:
    """Compute the anisotropy r(t) for given parallel/perpendicular traces.

    Parameters
    ----------
    par : array-like
        Parallel (VV) intensities.
    perp : array-like
        Perpendicular (VH) intensities.
    g_factor : float
        G-factor.
    l1 : float, optional
        L1 correction factor.
    l2 : float, optional
        L2 correction factor.

    Returns
    -------
    np.ndarray
        Anisotropy values (NaN where invalid).
    """
    g = float(g_factor)
    if not np.isfinite(g) or g <= 0.0:
        return np.full_like(np.asarray(par, dtype=float), np.nan, dtype=float)
    p = np.asarray(par, dtype=float)
    s = np.asarray(perp, dtype=float)
    num = g * p - s
    den = (1.0 - 3.0 * float(l2)) * g * p + (2.0 - 3.0 * float(l1)) * s
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=(np.isfinite(den) & (den != 0.0)))


def compute_background_levels(
    par_raw: np.ndarray,
    perp_raw: np.ndarray,
    time_axis: np.ndarray,
    shifted_time_axis: np.ndarray,
    bg_region_bounds: list[float] | tuple[float, float],
) -> tuple[float, float]:
    """Compute the mean background level of parallel and perpendicular channels.

    Parameters
    ----------
    par_raw : np.ndarray
        Raw parallel (VV) intensities.
    perp_raw : np.ndarray
        Raw perpendicular (VH) intensities.
    time_axis : np.ndarray
        Time axis for the parallel channel.
    shifted_time_axis : np.ndarray
        Time axis shifted for the perpendicular channel.
    bg_region_bounds : tuple/list of float
        The [bg_min, bg_max] bounds.

    Returns
    -------
    tuple of float
        ``(bg_parallel_avg, bg_perpendicular_avg)``.
    """
    bg_min_time, bg_max_time = bg_region_bounds
    bg_min_idx_parallel = np.argmin(np.abs(time_axis - bg_min_time))
    bg_max_idx_parallel = np.argmin(np.abs(time_axis - bg_max_time))
    bg_min_idx_perp = np.argmin(np.abs(shifted_time_axis - bg_min_time))
    bg_max_idx_perp = np.argmin(np.abs(shifted_time_axis - bg_max_time))
    bg_parallel = par_raw[bg_min_idx_parallel:bg_max_idx_parallel]
    bg_perpendicular = perp_raw[bg_min_idx_perp:bg_max_idx_perp]
    bg_parallel_avg = float(np.mean(bg_parallel)) if bg_parallel.size else 0.0
    bg_perpendicular_avg = float(np.mean(bg_perpendicular)) if bg_perpendicular.size else 0.0
    return bg_parallel_avg, bg_perpendicular_avg


def calculate_g_factor_core(
    parallel_data: list[float] | np.ndarray,
    perpendicular_data: list[float] | np.ndarray,
    region_bounds: list[float] | tuple[float, float],
    decay_shift: float = 0.0,
    use_bg: bool = False,
    bg_region_bounds: list[float] | tuple[float, float] | None = None,
    flip: bool = False,
) -> dict:
    """Calculate G-factor based on tail matching for Jordi decays.

    Parameters
    ----------
    parallel_data : list of float or np.ndarray
        Parallel (VV) decay curve.
    perpendicular_data : list of float or np.ndarray
        Perpendicular (VH) decay curve.
    region_bounds : list or tuple of float
        The tail-matching [min_time, max_time] bounds.
    decay_shift : float, optional
        The time shift applied to perpendicular trace.
    use_bg : bool, optional
        Whether background correction is enabled.
    bg_region_bounds : list or tuple of float, optional
        The background subtraction [min_time, max_time] bounds.
    flip : bool, optional
        Whether to swap parallel and perpendicular arrays.

    Returns
    -------
    dict
        Dictionary containing calculated G-factors, stddevs, background averages, etc.
    """
    logger.debug(
        "calculate_g_factor_core: region_bounds=%s, decay_shift=%f, use_bg=%s, bg_region_bounds=%s, flip=%s",
        region_bounds, decay_shift, use_bg, bg_region_bounds, flip
    )
    if flip:
        parallel_data, perpendicular_data = perpendicular_data, parallel_data

    n = min(len(parallel_data), len(perpendicular_data))
    par_full = np.asarray(parallel_data[:n], dtype=float)
    perp_full = np.asarray(perpendicular_data[:n], dtype=float)
    time_axis = np.arange(n, dtype=float)

    # Get region bounds for tail matching
    min_time, max_time = region_bounds

    # Find indices corresponding to the region for parallel data
    min_idx_parallel = int(np.argmin(np.abs(time_axis - min_time)))
    max_idx_parallel = int(np.argmin(np.abs(time_axis - max_time)))
    if max_idx_parallel < min_idx_parallel:
        min_idx_parallel, max_idx_parallel = max_idx_parallel, min_idx_parallel
    if max_idx_parallel == min_idx_parallel:
        max_idx_parallel = min(n, min_idx_parallel + 1)

    # Create shifted time axis for perpendicular data
    shifted_time_axis = time_axis + decay_shift

    # Find indices corresponding to the region for perpendicular data (accounting for shift)
    min_idx_perp = int(np.argmin(np.abs(shifted_time_axis - min_time)))
    max_idx_perp = int(np.argmin(np.abs(shifted_time_axis - max_time)))
    if max_idx_perp < min_idx_perp:
        min_idx_perp, max_idx_perp = max_idx_perp, min_idx_perp
    if max_idx_perp == min_idx_perp:
        max_idx_perp = min(n, min_idx_perp + 1)

    # Extract data in the region
    parallel_region = par_full[min_idx_parallel:max_idx_parallel]
    perpendicular_region = perp_full[min_idx_perp:max_idx_perp]
    time_region = time_axis[min_idx_parallel:max_idx_parallel]

    # If the shifted indices result in different array lengths, interpolate to match
    if len(parallel_region) != len(perpendicular_region):
        src_t = shifted_time_axis[min_idx_perp:max_idx_perp]
        if len(src_t) >= 2:
            perpendicular_region = np.interp(
                time_region,
                src_t,
                perpendicular_region,
                left=perpendicular_region[0],
                right=perpendicular_region[-1],
            )
        else:
            perpendicular_region = np.resize(perpendicular_region, len(parallel_region))

    # Calculate uncorrected g-factor
    with np.errstate(divide='ignore', invalid='ignore'):
        g_factors_uncorrected = np.divide(
            parallel_region,
            perpendicular_region,
            out=np.full_like(parallel_region, np.nan, dtype=float),
            where=(np.isfinite(perpendicular_region) & (perpendicular_region != 0))
        )
    valid_indices_uncorrected = ~np.isnan(g_factors_uncorrected) & ~np.isinf(g_factors_uncorrected) & (g_factors_uncorrected > 0)
    valid_g_factors_uncorrected = g_factors_uncorrected[valid_indices_uncorrected]

    if len(valid_g_factors_uncorrected) > 0:
        g_factor_uncorrected = float(np.mean(valid_g_factors_uncorrected))
        g_factor_stddev_uncorrected = float(np.std(valid_g_factors_uncorrected))
    else:
        g_factor_uncorrected = None
        g_factor_stddev_uncorrected = None

    logger.debug(
        "calculate_g_factor_core: uncorrected G-factor=%s (stddev=%s, valid points=%d/%d)",
        g_factor_uncorrected, g_factor_stddev_uncorrected, len(valid_g_factors_uncorrected), len(parallel_region)
    )

    bg_parallel_avg = 0.0
    bg_perpendicular_avg = 0.0
    g_factor_corrected = None
    g_factor_stddev_corrected = None

    if use_bg and bg_region_bounds is not None:
        bg_min_time, bg_max_time = bg_region_bounds
        bg_min_idx = int(np.argmin(np.abs(time_axis - bg_min_time)))
        bg_max_idx = int(np.argmin(np.abs(time_axis - bg_max_time)))
        if bg_max_idx < bg_min_idx:
            bg_min_idx, bg_max_idx = bg_max_idx, bg_min_idx
        if bg_max_idx == bg_min_idx:
            bg_max_idx = min(n, bg_min_idx + 1)

        bg_min_idx_perp = int(np.argmin(np.abs(shifted_time_axis - bg_min_time)))
        bg_max_idx_perp = int(np.argmin(np.abs(shifted_time_axis - bg_max_time)))
        if bg_max_idx_perp < bg_min_idx_perp:
            bg_min_idx_perp, bg_max_idx_perp = bg_max_idx_perp, bg_min_idx_perp
        if bg_max_idx_perp == bg_min_idx_perp:
            bg_max_idx_perp = min(n, bg_min_idx_perp + 1)

        bg_parallel = par_full[bg_min_idx:bg_max_idx]
        bg_perpendicular = perp_full[bg_min_idx_perp:bg_max_idx_perp]

        bg_parallel_avg = float(np.mean(bg_parallel)) if bg_parallel.size else 0.0
        bg_perpendicular_avg = float(np.mean(bg_perpendicular)) if bg_perpendicular.size else 0.0

        parallel_region_corrected = np.maximum(parallel_region - bg_parallel_avg, 0.0)
        perpendicular_region_corrected = np.maximum(perpendicular_region - bg_perpendicular_avg, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            g_factors_corrected = np.divide(
                parallel_region_corrected,
                perpendicular_region_corrected,
                out=np.full_like(parallel_region_corrected, np.nan, dtype=float),
                where=(np.isfinite(perpendicular_region_corrected) & (perpendicular_region_corrected != 0))
            )

        valid_indices_corrected = ~np.isnan(g_factors_corrected) & ~np.isinf(g_factors_corrected) & (g_factors_corrected > 0)
        valid_g_factors_corrected = g_factors_corrected[valid_indices_corrected]

        if len(valid_g_factors_corrected) > 0:
            g_factor_corrected = float(np.mean(valid_g_factors_corrected))
            g_factor_stddev_corrected = float(np.std(valid_g_factors_corrected))

        logger.debug(
            "calculate_g_factor_core: bg_parallel_avg=%f, bg_perpendicular_avg=%f",
            bg_parallel_avg, bg_perpendicular_avg
        )
        logger.debug(
            "calculate_g_factor_core: corrected G-factor=%s (stddev=%s, valid points=%d/%d)",
            g_factor_corrected, g_factor_stddev_corrected, len(valid_g_factors_corrected), len(parallel_region)
        )

    # Determine final g-factor
    if use_bg and g_factor_corrected is not None:
        g_factor = g_factor_corrected
    else:
        g_factor = g_factor_uncorrected

    logger.info("calculate_g_factor_core: final G-factor calculated: %s", g_factor)

    return {
        "g_factor": g_factor,
        "g_factor_uncorrected": g_factor_uncorrected,
        "g_factor_stddev_uncorrected": g_factor_stddev_uncorrected,
        "g_factor_corrected": g_factor_corrected,
        "g_factor_stddev_corrected": g_factor_stddev_corrected,
        "bg_parallel_avg": bg_parallel_avg,
        "bg_perpendicular_avg": bg_perpendicular_avg,
    }


def perrin_steady_state_anisotropy(tau_ns: float, rho_ns: float, r0: float = 0.38) -> float:
    """Perrin steady-state anisotropy for a sphere with rotational correlation time ``rho``.

    Parameters
    ----------
    tau_ns : float
        Fluorescence lifetime in nanoseconds.
    rho_ns : float
        Rotational correlation time in nanoseconds.
    r0 : float, optional
        Fundamental anisotropy (default 0.38).

    Returns
    -------
    float
        Steady-state anisotropy, or NaN if ``rho`` is non-positive.
    """
    tau = float(tau_ns)
    rho = float(rho_ns)
    if rho <= 0.0:
        return np.nan
    return float(r0 / (1.0 + tau / rho))


def estimate_lifetime_first_moment(time_axis: np.ndarray, intensity: np.ndarray) -> float:
    """Estimate the intensity-weighted first moment of the time axis (proxy lifetime).

    Parameters
    ----------
    time_axis : array-like
        Channel time axis.
    intensity : array-like
        Intensity values aligned with ``time_axis``.

    Returns
    -------
    float
        First moment in channel units, or NaN if the weighted sum is non-positive.
    """
    t = np.asarray(time_axis, dtype=float)
    i = np.asarray(intensity, dtype=float)
    valid = np.isfinite(t) & np.isfinite(i) & (i > 0.0)
    if not np.any(valid):
        return np.nan
    tv = t[valid]
    iv = i[valid]
    w = float(np.sum(iv))
    if w <= 0.0:
        return np.nan
    t0 = float(tv[0])
    return float(np.sum((tv - t0) * iv) / w)


def solve_linked_l_from_steady_state(sp: float, ss: float, g_factor: float, r_target: float) -> float:
    """Solve for the linked l1=l2 mixing parameter from a target steady-state r.

    Parameters
    ----------
    sp : float
        Sum of parallel channel counts (background corrected).
    ss : float
        Sum of perpendicular channel counts (background corrected).
    g_factor : float
        Detector G-factor.
    r_target : float
        Target steady-state anisotropy (e.g. from Perrin).

    Returns
    -------
    float
        Estimated l1=l2, or NaN on degenerate input.
    """
    sp = float(sp)
    ss = float(ss)
    g = float(g_factor)
    r = float(r_target)
    if not np.isfinite(sp) or not np.isfinite(ss) or not np.isfinite(g) or not np.isfinite(r):
        return np.nan
    if abs(r) < 1e-15:
        if abs(g * sp - ss) < 1e-15:
            return 0.0
        return np.nan
    den_ss = g * sp + ss
    if abs(den_ss) < 1e-15:
        return np.nan
    d = g * sp + 2.0 * ss
    num_val = g * sp - ss
    val = d - num_val / r
    return float(val / (3.0 * den_ss))
