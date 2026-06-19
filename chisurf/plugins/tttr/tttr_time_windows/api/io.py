"""Input/output helpers for the Time Window Bins API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .selection import compute_bids_from_tttr


def load_tttr(path: str | Path) -> "tttrlib.TTTR":
    """Load a TTTR file using tttrlib.

    Parameters
    ----------
    path : str or Path
        TTTR file path.

    Returns
    -------
    tttrlib.TTTR
        Loaded TTTR object.
    """
    import tttrlib

    return tttrlib.TTTR(str(path))


def save_bst(bids: np.ndarray, path: str | Path) -> None:
    """Save a BID array as a tab-separated ``.bst`` file.

    Parameters
    ----------
    bids : numpy.ndarray
        Array of shape ``(n_windows, 2)`` with ``[start_idx, stop_idx)``.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), bids.astype(np.int64), fmt="%d\t%d")


def compute_and_save(
    path: str | Path,
    time_window_s: float,
    output_dir: str | Path,
) -> tuple[int, str]:
    """Load a TTTR file, compute BIDs, and save as ``.bst``.

    Parameters
    ----------
    path : str or Path
        Input TTTR file.
    time_window_s : float
        Time window duration in seconds.
    output_dir : str or Path
        Output directory for the ``.bst`` file.

    Returns
    -------
    n_windows : int
        Number of time windows produced.
    output_path : str
        Path to the written ``.bst`` file.

    Raises
    ------
    RuntimeError
        If no data or computation fails.
    """
    import tttrlib

    tttr = tttrlib.TTTR(str(path))
    bids = compute_bids_from_tttr(tttr, time_window_s)
    if bids.size == 0:
        raise RuntimeError(f"No windows produced for {path}")
    out = Path(output_dir) / f"{Path(path).stem}.bst"
    save_bst(bids, out)
    return int(len(bids)), str(out.resolve())
