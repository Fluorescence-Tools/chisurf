import numpy as np
from typing import Union


def interpolate_shift(arr: np.ndarray, shift: Union[int, float]) -> np.ndarray:
    """
    Shift a 1D array by a given number of bins, supporting fractional shifts.

    Parameters
    ----------
    arr : np.ndarray
        Input array to shift. If None or empty, returns an empty float64 array.
    shift : int or float
        Number of bins to shift (positive rightwards, negative leftwards).

    Returns
    -------
    np.ndarray
        Shifted array with zeros filled. For arrays of length < 2, only the
        integer shift (zero-padded) is applied and fractional interpolation is
        skipped to avoid numpy.interp errors.
    """
    if arr is None:
        return np.array([], dtype=np.float64)

    result = np.asarray(arr, dtype=np.float64).copy()
    n = result.size
    if n == 0:
        return result

    if shift == 0:
        return result

    # Integer shift (zero-padded)
    int_shift = int(np.trunc(shift))
    if int_shift != 0:
        result = np.roll(result, int_shift)
        if int_shift > 0:
            result[:int_shift] = 0.0
        else:
            result[int_shift:] = 0.0

    # Fractional shift via interpolation, only if we have at least 2 points
    frac_shift = shift - int_shift
    if frac_shift != 0 and n >= 2:
        x = np.arange(result.size)
        result = np.interp(x - frac_shift, x, result, left=0.0, right=0.0)

    return result
