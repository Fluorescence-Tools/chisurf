"""Core algorithm: compute time-window BIDs from a TTTR object."""

from __future__ import annotations

import numpy as np


def compute_bids_from_tttr(tttr: "tttrlib.TTTR", time_window_s: float) -> np.ndarray:
    """Compute start/stop photon indices for fixed-duration time windows.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR data object.
    time_window_s : float
        Duration of each time window in seconds.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_windows, 2)`` with ``[start_idx, stop_idx)``
        per row.  Returns an empty array when no windows can be created.

    Raises
    ------
    ValueError
        If *time_window_s* is not positive.
    RuntimeError
        If the macro-time resolution cannot be determined from the TTTR
        header.
    """
    if time_window_s <= 0:
        raise ValueError("time_window_s must be positive")
    mt = tttr.macro_times
    if mt is None or len(mt) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    try:
        res = float(tttr.header.macro_time_resolution)
    except Exception:
        res = float(getattr(tttr, "macro_time_resolution", 0.0))
    if res <= 0:
        raise RuntimeError("Macro time resolution unavailable from TTTR header")

    clocks_per_bin = int(np.floor(time_window_s / res))
    if clocks_per_bin < 1:
        clocks_per_bin = 1

    max_clock = int(mt.max())
    if max_clock < 0:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.arange(0, max_clock + 1, clocks_per_bin, dtype=np.int64)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    starts = np.searchsorted(mt, edges, side="left")
    stops = np.searchsorted(mt, edges + clocks_per_bin, side="left")
    return np.stack([starts, stops], axis=1)
