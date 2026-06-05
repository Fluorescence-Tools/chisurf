"""
Count rate-based photon filtering for TTTR data.

This module provides functions for filtering photons based on count rate
criteria using the TTTR object's get_selection_by_count_rate method.
"""

import numpy as np
import tttrlib
from typing import Dict, Any, Optional, Union, List, Tuple


def count_rate_filter(
    tttr: tttrlib.TTTR,
    n_ph_max: int,
    time_window: float,
    invert: bool = False,
    make_mask: bool = True
) -> np.ndarray:
    """
    Filter photons based on count rate criteria.
    
    This function uses the TTTR object's get_selection_by_count_rate method
    to select photons where the count rate is below a specified threshold.
    
    Parameters
    ----------
    tttr : tttrlib.TTTR
        TTTR object containing the photon data.
    n_ph_max : int
        Maximum number of photons within the time window.
    time_window : float
        Length of the time window in seconds.
    invert : bool, optional
        If True, invert the selection criteria. Default is False.
    make_mask : bool, optional
        If True, return a boolean mask. If False, return indices. Default is True.
        
    Returns
    -------
    np.ndarray
        If make_mask is True, returns a boolean mask of selected photons.
        If make_mask is False, returns indices of selected photons.
    """

    # Create filter options dictionary
    filter_options = {
        'n_ph_max': n_ph_max,
        'time_window': time_window,
        'invert': invert,
        'make_mask': make_mask
    }
    
    # Call the TTTR object's get_selection_by_count_rate method
    selection = tttr.get_selection_by_count_rate(**filter_options)
    
    return selection
