"""CUSUM/SPRT-based photon filtering for TTTR data.

This module provides functions for filtering photons based on CUSUM/SPRT
burst search criteria using the TTTR object's burst_search method.
"""

import numpy as np
import tttrlib
from chisurf.core.fluorescence.burst.utils import create_array_with_ones


def cusum_filter(
    tttr: tttrlib.TTTR,
    min_ph: int,
    background_rate: int,
    sb_ratio: float,
    alpha: float = 0.05,
    beta: float = 0.05
) -> np.ndarray:
    """Filter photons based on CUSUM/SPRT burst search criteria.

    This function uses the TTTR object's burst_search method to identify
    bursts of photons using the CUSUM/SPRT algorithm.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR object containing the photon data.
    min_ph : int
        Minimum number of photons for a burst (L).
    background_rate : int
        Background count rate in counts/second (m).
    sb_ratio : float
        Signal-to-background ratio (T). If 0, auto-estimate.
    alpha : float, default=0.05
        False alarm probability.
    beta : float, default=0.05
        Missed detection probability.

    Returns
    -------
    np.ndarray
        Boolean mask of selected photons.
    """
    # Call the TTTR object's burst_search method in CUSUM mode
    start_stop = tttr.burst_search(
        L=min_ph,
        m=background_rate,
        T=sb_ratio,
        mode="cusum_sprt",
        alpha=alpha,
        beta=beta
    )

    # Reshape the result into a 2D array of start-stop pairs
    start_stop = np.array(start_stop).reshape((-1, 2))

    # Create a boolean mask of selected photons
    n = len(tttr)
    mask = create_array_with_ones(start_stop, n)

    return mask
