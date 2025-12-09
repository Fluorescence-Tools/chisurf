import numpy as np
from chisurf.plugins.burst_mle_analysis.interpolate import interpolate_shift


def test_interpolate_shift_empty_array_returns_empty():
    arr = np.array([], dtype=np.float64)
    out = interpolate_shift(arr, 0.5)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.size == 0


def test_interpolate_shift_len1_skips_fractional_interp():
    arr = np.array([3.0], dtype=np.float64)
    out = interpolate_shift(arr, 0.5)
    # With length 1, only integer shift is applicable (here 0), so unchanged
    assert out.shape == (1,)
    assert out[0] == 3.0


def test_interpolate_shift_len1_with_integer_shift_zero_padded():
    arr = np.array([3.0], dtype=np.float64)
    out = interpolate_shift(arr, 1.0)
    # Integer shift on length-1 array zero-pads the single element
    assert out.shape == (1,)
    assert out[0] == 0.0
