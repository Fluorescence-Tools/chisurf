"""
Burst search-based photon filtering for TTTR data.

This module provides functions for filtering photons based on burst search
criteria using the TTTR object's burst_search method.
"""

import numpy as np
import tttrlib
from typing import Dict, Any, Optional, Union, List, Tuple

from chisurf.fluorescence.burst.utils import create_array_with_ones


def burst_filter(
    tttr: tttrlib.TTTR,
    min_ph: int,
    ph_window: int,
    time_window: float
) -> np.ndarray:
    """
    Filter photons based on burst search criteria.
    
    This function uses the TTTR object's burst_search method to identify
    bursts of photons and returns a boolean mask of selected photons.
    
    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR object containing the photon data.
    min_ph : int
        Minimum number of photons for a burst.
    ph_window : int
        Number of photons to compute a count rate.
    time_window : float
        Maximum time window in seconds for a burst.
        
    Returns
    -------
    np.ndarray
        Boolean mask of selected photons.
    """
    # Call the TTTR object's burst_search method
    start_stop = tttr.burst_search(min_ph, ph_window, time_window)
    
    # Reshape the result into a 2D array of start-stop pairs
    start_stop = np.array(start_stop).reshape((-1, 2))
    
    # Create a boolean mask of selected photons
    n = len(tttr)
    mask = create_array_with_ones(start_stop, n)
    
    return mask
