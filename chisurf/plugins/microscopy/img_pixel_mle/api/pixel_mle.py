"""Pure computation helpers extracted from imgmle.py — no Qt at import time.

These functions are thin wrappers around the logic already implemented in
:class:`~chisurf.plugins.microscopy.img_pixel_mle.imgmle.LifetimeMleAnalysisWizard`.
They exist so that the CLI and backend services can invoke the analysis without
starting Qt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import PixelMleRequest, PixelMleResult


def interpolate_shift(arr: np.ndarray, shift: float) -> np.ndarray:
    """Shift a 1-D array by *shift* bins, supporting fractional shifts.

    Parameters
    ----------
    arr:
        1-D input array (e.g. an IRF histogram).
    shift:
        Number of bins to shift.  The integer part is applied with
        :func:`numpy.roll`; the fractional part is handled by linear
        interpolation.

    Returns
    -------
    numpy.ndarray
        Shifted array with the same dtype as the input.
    """
    result = arr.astype(np.float64).copy()
    if shift == 0:
        return result
    int_shift = int(np.trunc(shift))
    if int_shift != 0:
        result = np.roll(result, int_shift)
        if int_shift > 0:
            result[:int_shift] = 0.0
        else:
            result[int_shift:] = 0.0
    frac_shift = shift - int_shift
    if frac_shift != 0:
        x = np.arange(result.size)
        result = np.interp(x - frac_shift, x, result, left=0.0, right=0.0)
    return result


def prepare_irf(
    irf_p: np.ndarray,
    irf_s: np.ndarray,
    *,
    threshold: float = -1.0,
    shift: int = 0,
    shift_sp: float = 0.0,
    shift_ss: float = 0.0,
    threshold_vv: float | None = None,
    threshold_vh: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare parallel and perpendicular IRF arrays.

    Applies per-channel thresholding, normalisation and sub-bin shifts.

    Parameters
    ----------
    irf_p, irf_s:
        Raw parallel / perpendicular IRF histograms.
    threshold:
        Default fraction of the maximum below which bins are zeroed.
    shift:
        Integer relative shift of the perpendicular channel.
    shift_sp, shift_ss:
        Sub-bin (fractional) shifts applied to each channel independently.
    threshold_vv, threshold_vh:
        Per-channel overrides for the threshold fraction.

    Returns
    -------
    tuple[ndarray, ndarray]
        Prepared (irf_p, irf_s) arrays.
    """
    irf_p = irf_p.astype(np.float64).copy()
    irf_s = irf_s.astype(np.float64).copy()

    t_p = threshold_vv if threshold_vv is not None else threshold
    t_s = threshold_vh if threshold_vh is not None else threshold
    if t_p is not None and t_p > 0 and irf_p.size:
        irf_p[irf_p < t_p * irf_p.max()] = 0.0
    if t_s is not None and t_s > 0 and irf_s.size:
        irf_s[irf_s < t_s * irf_s.max()] = 0.0

    sp = irf_p.sum()
    if sp > 0:
        irf_p /= sp
    ss = irf_s.sum()
    if ss > 0:
        irf_s /= ss

    irf_p = interpolate_shift(irf_p, shift_sp)
    irf_s = interpolate_shift(irf_s, shift_ss)

    if shift != 0:
        irf_s = np.roll(irf_s, shift)
        if shift > 0:
            irf_s[:shift] = 0.0
        else:
            irf_s[shift:] = 0.0

    return irf_p, irf_s
