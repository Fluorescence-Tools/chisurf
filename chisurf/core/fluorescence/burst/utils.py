"""
Utility functions for burst detection and analysis.

This module contains common utility functions used by various burst detection
algorithms in the chisurf package.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Any
from numba import njit


@njit(cache=True)
def _fill_intervals(arr: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> None:
    for i in range(len(starts)):
        arr[starts[i]:stops[i]] = True


def create_array_with_ones(start_stop_pairs: np.ndarray, length: int) -> np.ndarray:
    """
    Create a boolean array of the given length, set to True (1)
    in the intervals [start, stop) defined by start_stop_pairs.
    
    Parameters
    ----------
    start_stop_pairs : np.ndarray
        Array of shape (n, 2) containing start and stop indices.
        Each row is a pair [start, stop] defining an interval.
    length : int
        Length of the output array.
        
    Returns
    -------
    np.ndarray
        Boolean array of length `length` with True values in the
        intervals defined by `start_stop_pairs`.
    """
    arr = np.zeros(length, dtype=bool)
    if len(start_stop_pairs) == 0:
        return arr
    pairs = np.asarray(start_stop_pairs, dtype=np.intp)
    _fill_intervals(arr, pairs[:, 0], pairs[:, 1])
    return arr
