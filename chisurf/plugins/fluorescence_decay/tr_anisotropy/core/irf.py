"""Pure IRF background correction for the anisotropy wizard.

Time-resolved anisotropy needs the VV and VH instrument-response functions (IRFs)
background-subtracted (using a signal-free region of the decay) and then rescaled
so the two channels carry the same total intensity. These operations are plain
NumPy and are unit-tested without Qt; the GUI's interactive region selector calls
:func:`correct_irfs` on every drag.
"""

from __future__ import annotations

import numpy as np


def initial_region(n: int, lo: float = 0.3, hi: float = 0.8) -> tuple[int, int]:
    """Return a default background region as ``(lower, upper)`` channel indices.

    Parameters
    ----------
    n : int
        Number of channels in the decay.
    lo, hi : float, optional
        Fractional bounds of the region (defaults 30 %–80 % of the range).

    Returns
    -------
    (int, int)
        The lower and upper channel indices.
    """
    return int(n * lo), int(n * hi)


def channel_background(y: np.ndarray, lb: int, ub: int) -> float:
    """Return the mean background of *y* over channels ``[lb, ub)`` (0.0 if empty)."""
    y = np.asarray(y, dtype=float)
    lb = max(0, min(int(lb), len(y)))
    ub = max(0, min(int(ub), len(y)))
    if ub <= lb:
        return 0.0
    sl = y[lb:ub]
    return float(np.nanmean(sl)) if sl.size else 0.0


def correct_irfs(
    y_vv: np.ndarray,
    y_vh: np.ndarray,
    lb: int,
    ub: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Background-subtract and intensity-normalise the VV / VH IRFs.

    The background is the per-channel mean over ``[lb, ub)``; it is subtracted and
    the result clipped to non-negative. Both channels are then rescaled so each
    carries the same total intensity (the mean of the two integrals), which is the
    normalisation the anisotropy model expects.

    Parameters
    ----------
    y_vv, y_vh : array_like
        Raw VV / VH IRF intensities. May differ in length (truncated to the
        shorter of the two).
    lb, ub : int
        Background region (channel indices); an empty/invalid region skips the
        subtraction (background 0).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The corrected VV and VH IRFs. Empty arrays are returned when either input
        is empty.
    """
    y_vv = np.asarray(y_vv, dtype=float)
    y_vh = np.asarray(y_vh, dtype=float)
    n = min(len(y_vv), len(y_vh))
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    y_vv = y_vv[:n]
    y_vh = y_vh[:n]

    bg_vv = channel_background(y_vv, lb, ub)
    bg_vh = channel_background(y_vh, lb, ub)
    vv = np.clip(y_vv - bg_vv, 0, None)
    vh = np.clip(y_vh - bg_vh, 0, None)

    s = (vv.sum() + vh.sum()) / 2.0
    if s > 0:
        if vv.sum() > 0:
            vv = vv * (s / vv.sum())
        if vh.sum() > 0:
            vh = vh * (s / vh.sum())
    return vv, vh
