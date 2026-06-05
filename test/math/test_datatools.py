import pytest
import numpy as np
import chisurf.core.math.datatools as dt

@pytest.mark.parametrize("distances, separation, sigma, normalize, expected_sum", [
    (np.linspace(0, 10, 100), 5.0, 1.0, True, 1.0),
    (np.linspace(0, 10, 100), 5.0, 1.0, False, None), # Sum depends on spacing
    (np.array([5.0]), 5.0, 1.0, True, 1.0),
    (np.array([]), 5.0, 1.0, True, 0.0), # Empty input
    (np.array([np.nan, 5.0]), 5.0, 1.0, True, 1.0), # NaN handling
])
def test_distance_between_gaussian(distances, separation, sigma, normalize, expected_sum):
    result = dt.distance_between_gaussian(distances, separation, sigma, normalize=normalize)
    assert result.shape == distances.shape
    assert np.all(result[~np.isnan(result)] >= 0)
    if normalize and distances.size > 0:
        assert np.isclose(np.sum(result), expected_sum)

def test_histogram_rebin():
    counts = np.array([0, 2, 1])
    bin_edges = np.array([0, 5, 10, 15])
    new_bin_edges = np.array([-5, 2.5, 7.5, 12.5, 20])
    
    rebinded = dt.histogram_rebin(bin_edges, counts, new_bin_edges)
    # new_bin_edges: -5 (out -> 0), 2.5 (bin 0-5 -> counts[0]=0), 7.5 (bin 5-10 -> counts[1]=2), 
    #                12.5 (bin 10-15 -> counts[2]=1), 20 (out -> 0)
    expected = [0.0, 0, 2, 1, 0.0]
    assert rebinded == expected

def test_bin_count():
    data = np.array([0, 10, 20, 30, 4000])
    bins, counts = dt.bin_count(data, bin_width=100, bin_min=0, bin_max=4000)
    assert bins.shape == counts.shape
    assert counts[0] == 4 # 0, 10, 20, 30 are in first bin [0, 100)
    assert counts[-1] == 0 # 4000 is bin_max, bin_index = 40. n_bins = 40. 40 < 40 is false.

def test_bin_count_empty():
    bins, counts = dt.bin_count(np.array([]), bin_width=100)
    assert counts.sum() == 0


def test_minmax():
    x = np.array([0, 1, 2, 3, 4, 5])
    assert dt.minmax(x) == (0, 5)
    assert dt.minmax(x, ignore_zero=True) == (1, 5)
    
    x_neg = np.array([-10, 0, 10])
    assert dt.minmax(x_neg) == (-10, 10)

def test_overlapping_region():
    x1 = np.linspace(0, 10, 11)
    y1 = x1 * 2
    x2 = np.linspace(5, 15, 11)
    y2 = x2 * 3
    
    (rx1, ry1), (rx2, ry2) = dt.overlapping_region((x1, y1), (x2, y2))
    # Overlap should be [5, 10]
    assert np.all(rx1 >= 5) and np.all(rx1 <= 10)
    assert np.all(rx2 >= 5) and np.all(rx2 <= 10)
    assert np.array_equal(ry1, rx1 * 2)
    assert np.array_equal(ry2, rx2 * 3)

def test_overlapping_region_no_overlap():
    x1 = np.linspace(0, 5, 6)
    y1 = x1
    x2 = np.linspace(10, 15, 6)
    y2 = x2
    (rx1, ry1), (rx2, ry2) = dt.overlapping_region((x1, y1), (x2, y2))
    assert rx1.size == 0
    assert rx2.size == 0


def test_interleaved_conversions():
    spectrum = np.array([0.1, 1.0, 0.2, 2.0]) # amp, lifetime, amp, lifetime
    amps, lifes = dt.interleaved_to_two_columns(spectrum)
    assert np.array_equal(amps, [0.1, 0.2])
    assert np.array_equal(lifes, [1.0, 2.0])
    
    interleaved = dt.two_column_to_interleaved(amps, lifes)
    assert np.array_equal(interleaved, spectrum)

def test_invert_interleaved():
    spectrum = np.array([0.1, 2.0, 0.2, 4.0])
    inverted = dt.invert_interleaved(spectrum)
    # Amps same, lifetimes inverted (converted to rates)
    expected = np.array([0.1, 0.5, 0.2, 0.25])
    assert np.allclose(inverted, expected)

def test_smooth_edge_cases():
    x = np.ones(10)
    # l < m case
    smoothed = dt.smooth(x, 2, 5)
    assert np.all(smoothed[2:] == 0) # Only first l-m elements are processed
    # m = 0 case
    smoothed_m0 = dt.smooth(x, 10, 0)
    # When m=0, inner loop i-m to i+m is range(i, i), so xz[i] remains 0
    # or takes one value if it was range(i-m, i+m+1). 
    # Current implementation: range(i-m, i+m) which is i-0, i+0 -> empty.
    assert np.all(smoothed_m0 == 0) 

