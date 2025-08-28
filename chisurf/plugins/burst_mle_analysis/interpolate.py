import numpy as np
from typing import Union


def interpolate_shift(arr: np.ndarray, shift: Union[int, float]) -> np.ndarray:
    """
    Shift a 1D array by a given number of bins, supporting fractional shifts.

    Parameters
    ----------
    arr : np.ndarray
        Input array to shift.
    shift : int or float
        Number of bins to shift (positive rightwards, negative leftwards).

    Returns
    -------
    np.ndarray
        Shifted array with zeros filled.
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
