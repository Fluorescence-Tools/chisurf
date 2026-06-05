import pytest
import numpy as np
import chisurf.core.math.signal

@pytest.mark.parametrize("args, expected", [
    ((np.arange(10), 0,), np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])),
    ((np.arange(10), 1,), np.array([0., 0., 1., 2., 3., 4., 5., 6., 7., 8.])),
    ((np.arange(10), 1.5,), np.array([0., 0., 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])),
    ((np.arange(10), 2.0,), np.array([0., 0., 0., 1., 2., 3., 4., 5., 6., 7.])),
    ((np.arange(10), -1.0,), np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 0.])),
    ((np.arange(10), -1.5,), np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.0, 0.0])),
    ((np.arange(10), -2.0,), np.array([2., 3., 4., 5., 6., 7., 8., 9., 0., 0.])),
    ((np.arange(10), -2.0, True, 33.), np.array([2., 3., 4., 5., 6., 7., 8., 9., 33., 33.])),
    ((np.arange(10), 2.0, True, 33.), np.array([33., 33., 0., 1., 2., 3., 4., 5., 6., 7.])),
    ((np.arange(10), -1.5, True, 33.), np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 33.0, 33.0])),
    ((np.arange(10), -2.0, False, 33.), np.array([2., 3., 4., 5., 6., 7., 8., 9., 0., 1.])),
    ((np.arange(10), -1.5, False, 33.), np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 4.5])),
    # Edge cases
    ((np.array([]), 1.0), np.array([])),
    ((np.array([1, 2, 3]), 10.0), np.array([0., 0., 0.])),
    ((np.array([1, 2, 3]), -10.0), np.array([0., 0., 0.])),
])
def test_math_signal_shift_array(args, expected):
    result = chisurf.core.math.signal.shift_array(*args)
    assert np.allclose(result, expected)

@pytest.mark.parametrize("window_type", chisurf.core.math.signal.window_function_types)
def test_window_smoothing(window_type):
    x = np.linspace(0, 2*np.pi, 100)
    data = np.sin(x)
    smoothed = chisurf.core.math.signal.window(data, window_len=11, window_function_type=window_type)
    assert smoothed.shape == data.shape
    assert np.all(np.isfinite(smoothed))

def test_window_errors():
    with pytest.raises(ValueError, match="smooth only accepts 1 dimension arrays"):
        chisurf.core.math.signal.window(np.zeros((5, 5)), 3)
    with pytest.raises(ValueError, match="Input vector needs to be bigger than window size"):
        chisurf.core.math.signal.window(np.arange(2), 5)
    with pytest.raises(ValueError, match="Window must be one of"):
        chisurf.core.math.signal.window(np.arange(10), 5, window_function_type="invalid")

@pytest.mark.parametrize("background", [0.0, 100.0])
def test_calculate_fwhm(background):
    # Create a centered peak
    x = np.linspace(-5, 5, 101)
    y = np.exp(-x**2 / 0.5)
    fwhm, (lb_i, ub_i), (x_left, x_right) = chisurf.core.math.signal.calculate_fwhm(x, y, background=background)
    if not np.all(y <= background):
        assert fwhm > 0
        assert lb_i <= ub_i
        assert x_left <= x_right
    else:
        assert fwhm == 0.0

def test_find_bursts():
    arr = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
    bursts = chisurf.core.math.signal.find_bursts(arr)
    expected = np.array([[1, 2], [5, 7]])
    assert np.array_equal(bursts, expected)
    
    # Merged gaps
    bursts_merged = chisurf.core.math.signal.find_bursts(arr, max_gap=2)
    expected_merged = np.array([[1, 7]])
    assert np.array_equal(bursts_merged, expected_merged)

    # Empty/Zero
    assert chisurf.core.math.signal.find_bursts(np.array([])).size == 0
    assert chisurf.core.math.signal.find_bursts(np.zeros(10)).size == 0
