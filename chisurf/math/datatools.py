from __future__ import annotations

from itertools import tee
from chisurf import typing

import copy
from math import floor

import numba as nb
import numpy as np


def histogram_rebin(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        new_bin_edges: np.ndarray
):
    """Interpolates a histogram to a new set of bin edges.

    This function returns the histogram values corresponding to a new set of bin edges.
    If a new bin edge is outside the range of the original histogram's bin edges, the value
    is set to 0. This is useful for changing the bin spacing of a histogram.

    :param bin_edges: array
        The bin edges of the original histogram.
    :param counts: array
        Histogram values corresponding to the bins defined by bin_edges.
    :param new_bin_edges: array
        The new bin edges for which to interpolate the histogram.
    :return: list or float
        A list of histogram values corresponding to new_bin_edges if its length > 1,
        otherwise a single value.

    Examples
    --------
    >>> counts = np.array([0, 2, 1])
    >>> bin_edges = np.array([0, 5, 10, 15])
    >>> new_bin_edges = np.linspace(-5, 20, 17)
    >>> histogram_rebin(bin_edges, counts, new_bin_edges)
    [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0.0, 0.0, 0.0, 0.0]
    """
    re = list()
    for xi in new_bin_edges.flatten():
        if xi > max(bin_edges) or xi < min(bin_edges):
            re.append(0.0)
        else:
            sel = np.where(xi < bin_edges)
            re.append(counts[sel[0][0] - 1])
    if len(re) == 1:
        return re[0]
    else:
        return re


def overlapping_region(
        dataset1,
        dataset2
):
    """Find the overlapping region between two datasets based on their x-values.

    Each dataset is a tuple of (x, y) arrays. This function computes the common x-range
    where both datasets overlap and masks out data points outside this range. The resulting
    overlapping segments of the datasets are returned.

    :param dataset1: tuple
        A tuple containing two numpy arrays (x and y values) for the first dataset.
    :param dataset2: tuple
        A tuple containing two numpy arrays (x and y values) for the second dataset.
    :return: tuple of tuples
        Two tuples corresponding to the overlapping regions of dataset1 and dataset2.
        Each tuple contains the x and y values within the overlapping range.

    Examples
    --------
    >>> import matplotlib.pylab as p
    >>> x1 = np.linspace(-1, 12, 10)
    >>> y1 = np.cos(x1)
    >>> a1 = (x1, y1)
    >>> x2 = np.linspace(-1, 12, 11)
    >>> y2 = np.sin(x2)
    >>> a2 = (x2, y2)
    >>> (rx1, ry1), (rx2, ry2) = overlapping_region(a1, a2)
    >>> p.plot(x1, y1, 'r')
    >>> p.plot(rx1, ry1, 'k')
    >>> p.plot(x2, y2, 'g')
    >>> p.plot(rx2, ry2, 'b')
    >>> p.show()
    """
    x1, y1 = dataset1
    x2, y2 = dataset2

    # Sort the arrays in ascending order in x
    x1 = np.array(x1)
    y1 = np.array(y1)
    inds = x1.argsort()
    x1 = x1[inds]
    y1 = y1[inds]
    x1 = np.ma.array(x1)
    y1 = np.ma.array(y1)

    x2 = np.array(x2)
    y2 = np.array(y2)
    inds = x2.argsort()
    x2 = x2[inds]
    y2 = y2[inds]
    x2 = np.ma.array(x2)
    y2 = np.ma.array(y2)

    # Calculate range for adjustment of spacing
    rng = (max(min(x1), min(x2)), min(max(x1), max(x2)))

    # Mask the irrelevant parts
    m1l = x1.data < rng[0]
    m1u = x1.data > rng[1]
    m1 = m1l + m1u
    x1.mask = m1
    y1.mask = m1

    m2l = x2.data < rng[0]
    m2u = x2.data > rng[1]
    m2 = m2l + m2u
    x2.mask = m2
    y2.mask = m2

    # Create new lists for the overlapping regions
    ox1 = x1.compressed()
    oy1 = y1.compressed()
    ox2 = x2.compressed()
    oy2 = y2.compressed()

    return (ox1, oy1), (ox2, oy2)


def align_x_spacing(
        dataset1: typing.Tuple[np.ndarray, np.ndarray],
        dataset2: typing.Tuple[np.ndarray, np.ndarray],
        method: str = 'linear-close'
) -> typing.Tuple[
    typing.Tuple[np.ndarray, np.ndarray],
    typing.Tuple[np.ndarray, np.ndarray]
]:
    """Align the x-spacing of two datasets using a template and rescaling method.

    This function takes two datasets (each a tuple of (x, y) arrays) and aligns the x-values of the dataset
    with more points to match the x-spacing of the dataset with fewer points. The default method ('linear-close')
    uses linear interpolation between nearby data points.

    :param dataset1: tuple of np.ndarray
        The first dataset as a tuple (x, y).
    :param dataset2: tuple of np.ndarray
        The second dataset as a tuple (x, y).
    :param method: str, optional
        The method used for alignment. Currently supports 'linear-close'.
    :return: tuple of tuples
        Two tuples corresponding to the aligned datasets (x, y) for dataset1 and dataset2.

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.linspace(-5, 5, 10)
    >>> y1 = np.sin(x1)
    >>> a1 = (x1, y1)
    >>> x2 = np.linspace(0, 10, 10)
    >>> y2 = np.sin(x2)
    >>> a2 = (x2, y2)
    >>> (rx1, ry1), (rx2, ry2) = align_x_spacing(a1, a2)
    """
    (ox1, oy1), (ox2, oy2) = dataset1, dataset2
    # Assume that data is more or less equally spaced
    # t - template array, r - rescale array
    if len(ox1) < len(ox2):
        tx = ox1
        ty = oy1
        rx = ox2
        ry = oy2
        cm = "r2"
    else:
        tx = ox2
        ty = oy2
        rx = ox1
        ry = oy1
        cm = "r1"

    # Create template arrays as new arrays - n
    nx = copy.deepcopy(tx)
    ny = copy.deepcopy(ty)

    if method == 'linear-close':
        j = 0  # counter for template array
        ry1 = ry[0]
        rx1 = rx[0]
        for i, rxi in enumerate(rx):
            if j < len(tx) - 1:
                if rxi > tx[j]:
                    rx2 = rxi
                    ry2 = ry[i]
                    if ry2 - ry1 != 0 and rx1 - rx2 != 0:
                        m = (ry2 - ry1) / (rx1 - rx2)
                        ny[j] = m * tx[j] + ry1 - m * rx1
                    else:
                        ny[j] = ry1
                    j += 1
                elif rxi == tx[j]:
                    ny[j] = ry[i]
                    j += 1
                else:
                    ry1 = ry[i]
                    rx1 = rx[i]
            else:
                ny[j] = ry[i:].mean()
    if cm == "r1":
        rx1 = nx
        ry1 = ny
        rx2 = tx
        ry2 = ty
    elif cm == "r2":
        rx2 = nx
        ry2 = ny
        rx1 = tx
        ry1 = ty
    return (rx1, ry1), (rx2, ry2)


def bin_count(
        data: np.ndarray,
        bin_width: int = 16,
        bin_min: int = 0,
        bin_max: int = 4095
):
    """
    Count the number of occurrences of each value in an array of non-negative integers.

    The data is binned into intervals of width bin_width, starting from bin_min up to bin_max.
    The function returns the bin values and the counts for each bin.

    :param data: array_like
        1-dimensional input array of non-negative integers.
    :param bin_width: int, optional
        The width of each bin. Default is 16.
    :param bin_min: int, optional
        The minimum value to consider for binning. Default is 0.
    :param bin_max: int, optional
        The maximum value to consider for binning. Default is 4095.
    :return: tuple (bins, count)
        bins: numpy array of bin starting values.
        count: numpy array of counts for each bin.
    """
    n_min = np.rint(bin_min / bin_width)
    n_max = np.rint(bin_max / bin_width)
    n_bins = n_max - n_min
    count = np.zeros(n_bins, dtype=np.float32)
    bins = np.arange(n_min, n_max, dtype=np.float32)
    bins *= bin_width
    for i in range(data.shape[0]):
        bin_index = np.rint((data[i] / bin_width)) - n_min
        if bin_index < n_bins:
            count[bin_index] += 1
    return bins, count


@nb.jit(nopython=True, nogil=True)
def minmax(
        x: np.ndarray,
        ignore_zero: bool = False
):
    """Compute the minimum and maximum values of an array.

    If ignore_zero is True, zeros in the array will be ignored when computing the minimum value.

    :param x: array or list
        The input array.
    :param ignore_zero: bool, optional
        Whether to ignore zero values when computing the minimum. Default is False.
    :return: tuple (min, max)
        The minimum and maximum values in the array.
    """
    max_v = -np.inf
    min_v = np.inf
    for i in x:
        if i > max_v:
            max_v = i
        if ignore_zero and i == 0:
            continue
        if i < min_v:
            min_v = i
    return min_v, max_v


@nb.jit(nopython=True, nogil=True)
def histogram1D(
        values,
        weights,
        n_bins: int = 101,
        tth_max: float = -1e12,
        tth_min: float = 1e12
):
    """Compute a 1D histogram with linear binning between specified limits.

    The histogram is computed by linearly binning the input values between tth_min and tth_max
    into n_bins bins. The weights associated with each value are summed in the corresponding bin.

    :param values: array_like
        The values to be binned.
    :param weights: array_like
        The weights for each value.
    :param n_bins: int, optional
        Number of bins in the histogram. Default is 101.
    :param tth_max: float, optional
        The maximum value for binning. Default is -1e12.
    :param tth_min: float, optional
        The minimum value for binning. Default is 1e12.
    :return: tuple (axis, hist)
        axis: numpy array representing the bin centers.
        hist: numpy array containing the weighted counts for each bin.
    """
    n_values = values.size

    bin_width = (n_bins - 1.0) / (tth_max - tth_min)
    hist = np.zeros(n_bins, dtype=np.float64)
    axis = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        axis[i] = i / bin_width + tth_min

    for i in range(n_values):
        v = values[i]
        if (v > tth_max) or (v < tth_min):
            continue
        bin = int(floor((v - tth_min) * bin_width))
        hist[bin] += weights[i]
    return axis, hist


@nb.jit(nopython=True, nogil=True)
def discriminate(
        values: np.ndarray,
        weights: np.ndarray,
        discriminator: float
):
    """Filter values and weights based on a discriminator threshold.

    This function selects elements from the input arrays where the corresponding weight is greater than
    the given discriminator value.

    :param values: array_like
        Array of values.
    :param weights: array_like
        Array of weights corresponding to the values.
    :param discriminator: float
        The threshold value for weights.
    :return: tuple (filtered_values, filtered_weights)
        Arrays containing the values and weights that passed the discriminator threshold.
    """
    v_r = np.zeros_like(values)
    w_r = np.zeros_like(weights)

    n_v = 0
    for i, w in enumerate(weights):
        if w > discriminator:
            w_r[n_v] = w
            v_r[n_v] = values[i]
            n_v += 1
    return v_r[:n_v], w_r[:n_v]


def smooth(x, l, m):
    """Smooth an array using a moving average filter.

    This function applies a simple moving average smoothing to the input array `x`. The smoothing is performed
    over a window of size (2*m + 1) for the first (l - m) elements of the array.

    Note: The implementation divides by (2*m + 1) inside the inner loop, which may not produce the intended
    averaging effect if multiple accumulations occur before division. Ensure that the parameters `l` and `m`
    are chosen appropriately relative to the size of `x`.

    :param x: numpy array
        Input array to be smoothed.
    :param l: int
        Number of elements from `x` to process.
    :param m: int
        Half-window size for the moving average. The full window size is (2*m + 1).
    :return: numpy array
        The smoothed array.
    """
    xz = np.empty(x.shape[0], dtype=np.float64)
    for i in range(l - m):
        xz[i] = 0
        for j in range(i - m, i + m):
            xz[i] += x[j]
            xz[i] /= (2 * m + 1)
    return xz


def interleaved_to_two_columns(
        ls: np.ndarray,
        sort: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Convert an interleaved spectrum into two-column data.

    The interleaved spectrum is assumed to alternate between amplitude and lifetime values.
    Optionally, the resulting data can be sorted by the lifetime values.

    :param ls: numpy array
        The interleaved spectrum (alternating amplitude and lifetime values).
    :param sort: bool, optional
        If True, the resulting columns are sorted by lifetime values. Default is False.
    :return: tuple (amplitudes, lifetimes)
        Two numpy arrays representing amplitudes and lifetimes respectively.

    Examples
    --------
    >>> import numpy as np
    >>> lifetime_spectrum = np.array([0.25, 1, 0.75, 4])
    >>> amplitudes, lifetimes = interleaved_to_two_columns(lifetime_spectrum)
    """
    lt = ls.reshape((ls.shape[0] // 2, 2))
    if sort:
        s = lt[np.argsort(lt[:, 1])]
        y = s[:, 0]
        x = s[:, 1]
        return y, x
    else:
        return lt[:, 0], lt[:, 1]


def two_column_to_interleaved(
        x: np.ndarray,
        t: np.ndarray
) -> np.ndarray:
    """Convert two-column lifetime spectra into an interleaved format.

    The two input arrays (amplitudes and lifetimes) are interleaved to form a single array.

    :param x: numpy array
        Array of amplitudes.
    :param t: numpy array
        Array of lifetimes (or rate constants).
    :return: numpy array
        Interleaved array containing amplitude and lifetime values.
    """
    c = np.vstack((x, t)).reshape(-1, order='F')
    return c


@nb.jit(nopython=True, nogil=True)
def elte2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Combine two interleaved lifetime spectra into a new spectrum.

    For each pair of corresponding elements in the two spectra, the resulting amplitude is the product
    of the amplitudes, and the resulting lifetime is computed as the harmonic mean:
    1 / (1/l1 + 1/l2).

    :param e1: array_like
        First interleaved lifetime spectrum in the format (a1, l1, a2, l2, ...).
    :param e2: array_like
        Second interleaved lifetime spectrum in the same format as e1.
    :return: numpy array
        A new interleaved lifetime spectrum of the form (a1*a1, harmonic_mean(l1, l2), ...).

    Examples
    --------
    >>> import numpy as np
    >>> e1 = np.array([1, 2, 3, 4])
    >>> e2 = np.array([5, 6, 7, 8])
    >>> elte2(e1, e2)
    array([ 5.        ,  1.5       ,  7.        ,  1.6       , 15.        ,
            2.4       , 21.        ,  2.66666667])
    """
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.empty(n1 * n2 * 2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = 1. / (1. / e1[i * 2 + 1] + 1. / e2[j * 2 + 1])
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def ere2(
        e1: np.ndarray,
        e2: np.ndarray
) -> np.array:
    """
    Combine two interleaved rate spectra into a new spectrum.

    For each pair of corresponding elements in the two spectra, the resulting amplitude is the product
    of the amplitudes, and the resulting rate is the sum of the rates.

    :param e1: array_like
        First interleaved rate spectrum in the format (a1, r1, a2, r2, ...).
    :param e2: array_like
        Second interleaved rate spectrum in the same format as e1.
    :return: numpy array
        A new interleaved rate spectrum of the form (a1*a1, r1+r1, ...).

    Examples
    --------
    >>> import numpy as np
    >>> e1 = np.array([0.5, 1, 0.5, 2])
    >>> e2 = np.array([0.5, 3, 0.5, 4])
    >>> ere2(e1, e2)
    array([0.25, 4.  , 0.25, 5.  , 0.25, 5.  , 0.25, 6.  ])
    """
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.empty(n1 * n2 * 2, dtype=np.float64)
    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = e1[i * 2 + 1] + e2[j * 2 + 1]
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def invert_interleaved(
        interleaved_spectrum: np.ndarray
) -> np.ndarray:
    """Convert an interleaved lifetime spectrum to a rate spectrum and vice versa.

    For each pair in the interleaved spectrum, the amplitude remains unchanged while the second value
    (lifetime or rate) is inverted (i.e., replaced by its reciprocal).

    :param interleaved_spectrum: array_like
        Interleaved lifetime or rate spectrum.
    :return: numpy array
        The spectrum with each second component inverted.

    Examples
    --------
    >>> import numpy as np
    >>> e1 = np.array([1, 2, 3, 4])
    >>> invert_interleaved(e1)
    array([1.  , 0.5 , 3.  , 0.25])
    """
    n1 = interleaved_spectrum.shape[0] // 2
    r = np.empty(n1 * 2, dtype=np.float64)

    for i in range(n1):
        r[i * 2 + 0] = interleaved_spectrum[i * 2 + 0]
        r[i * 2 + 1] = 1. / (interleaved_spectrum[i * 2 + 1])
    return r


@nb.jit(nopython=True, nogil=True)
def e1tn(
        e1: np.array,
        n: float
) -> np.array:
    """
    Multiply the amplitude components of an interleaved spectrum by a constant factor.

    Only the amplitude values (every other element starting from the first) are multiplied by the factor n.
    The rate or lifetime values remain unchanged.

    :param e1: array_like
        An interleaved spectrum in the format (amplitude, value, amplitude, value, ...).
    :param n: float
        The multiplication factor for the amplitude components.
    :return: numpy array
        The interleaved spectrum with scaled amplitude values.

    Examples
    --------
    >>> e1 = np.array([1, 2, 3, 4])
    >>> e1tn(e1, 2.0)
    array([2, 2, 6, 4])
    """
    n2 = e1.shape[0]
    for i in range(0, n2, 2):
        e1[i] *= n
    return e1


@nb.jit(nopython=True, nogil=True)
def e1ti2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Combine two interleaved spectra by multiplying corresponding amplitude and rate/lifetime components.

    This function takes two interleaved spectra (each formatted as alternating amplitude and rate/lifetime values)
    and computes a new interleaved spectrum. For each combination of a pair from the first spectrum and a pair from
    the second spectrum, the resulting amplitude is the product of the amplitude values, and the resulting rate/lifetime
    is the product of the corresponding second values.

    :param e1: array_like
        First interleaved spectrum.
    :param e2: array_like
        Second interleaved spectrum.
    :return: numpy array
        A new interleaved spectrum resulting from the element-wise multiplications.

    Examples
    --------
    >>> import numpy as np
    >>> e1 = np.array([1, 2, 3, 4])
    >>> e2 = np.array([5, 6, 7, 8])
    >>> e1ti2(e1, e2)
    array([ 5., 12.,  7., 16., 15., 24., 21., 32.])
    """
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.zeros(n1 * n2 * 2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = e1[i * 2 + 1] * e2[j * 2 + 1]
            k += 1
    return r


def pairwise(iterable):
    """Generate successive overlapping pairs from an iterable.

    Example: s -> (s0, s1), (s1, s2), (s2, s3), ...

    :param iterable: iterable
        An iterable sequence.
    :return: iterator
        An iterator over pairs of consecutive elements.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
