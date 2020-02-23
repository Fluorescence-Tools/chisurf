from __future__ import annotations

from itertools import tee
import typing

import copy
from math import floor

import numba as nb
import numpy as np


def histogram_rebin(
        bin_edges: np.ndarray,
        counts: np.ndarray,
        new_bin_edges: np.ndarray
):
    """
    Interpolates a histogram to a new x-axis. Here the parameter x can be any
    numpy array as return value another array is obtained containing the values
    of the histogram at the given x-values. This function may be useful if the spacing
    of a histogram has to be changed.

    :param new_bin_edges: array
    :param counts: array
        Histogram values
    :param bin_edges: array
        The bin edges of the histogram
    :return: A list of the same size as the parameter x

    Examples
    --------

    >>> counts = np.array([0,2,1])
    >>> bin_edges = np.array([0,5,10,15])

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
    """

    :param dataset1: tuple
        The tuple should consist of the x and y values. Whereas the x and
        y values have to be a numpy array.
    :param dataset2: tuple
        The tuple should consist of the x and y values. Whereas the x and
        y values have to be a numpy array.
    :return: Two tuples

    Examples
    --------

    >>> import matplotlib.pylab as p
    >>> x1 = np.linspace(-1, 12, 10)
    >>> y1 = np.cos(x1)
    >>> a1 = (x1, y1)
    >>> x2 = np.linspace(-1, 12, 11)
    >>> y2 = np.sin(x2)
    >>> a2 = (x2, y2)
    >>> (rx1, ry1), (rx2, ry2) = align_x_spacing(a1, a2)

    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()

    Test overlay

    >>> x1 = np.linspace(-5, 5, 10)
    >>> y1 = np.sin(x1)
    >>> a1 = (x1, y1)
    >>> x2 = np.linspace(0, 10, 10)
    >>> y2 = np.sin(x2)
    >>> a2 = (x2, y2)
    >>> (rx1, ry1), (rx2, ry2) = overlapping_region(a1, a2)

    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()

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
    
    # mask the irrelevant parts
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
    
    # create new lists: overlap - o and rest -r
    # overlap
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
    """

    :param dataset1:
    :param dataset2:
    :param method:
    :return:
    """
    (ox1, oy1), (ox2, oy2) = dataset1, dataset2
    #Assume that data is more or less equaliy spaced
    # t- template array, r - rescale array
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

    # Create of template-arrays as new arrays - n
    nx = copy.deepcopy(tx)
    ny = copy.deepcopy(ty)
 
    if method == 'linear-close':
        j = 0 # counter for tx
        ry1 = ry[0]
        rx1 = rx[0]
        for i, rxi in enumerate(rx):
            if j < len(tx)-1:
                if rxi > tx[j]:
                    rx2 = rxi
                    ry2 = ry[i]
                    if ry2-ry1 != 0 and rx1-rx2 != 0:
                        m = (ry2-ry1)/(rx1-rx2)
                        ny[j] = m*tx[j] + ry1 - m * rx1
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
    Count number of occurrences of each value in array of non-negative ints.

    :param data: array_like
        1-dimensional input array
    :param bin_width: the width of the bins
    :param bin_min: The value of the smallest bin
    :param bin_max: The value of the largest bin
    :return: As return values a numpy array with the bins and a array containing the counts is obtained.
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
    """Minimum and maximum value of an array

    :param x: array or list
    :return: minimum and maximum value
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
        n_bins: int = 101
):
    """ Creates a histogram of the values the histogram is linear between the
    minium and the maximum value

    :param values: the values to be binned
    :param weights: the weights of the values
    :param n_bins: number of bins
    :return:
    """
    n_values = values.size

    tth_max = -1e12
    tth_min = 1e12
    for v in values:
        if v > tth_max:
            tth_max = v
        if v < tth_min:
            tth_min = v
    bin_width = (n_bins - 1.0) / (tth_max - tth_min)

    hist = np.zeros(n_bins, dtype=np.float64)
    axis = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        axis[i] = i / bin_width + tth_min

    for i in range(n_values):
        bin = int(floor(((values[i]) - tth_min) * bin_width))
        hist[bin] += weights[i]
    return axis, hist


@nb.jit(nopython=True, nogil=True)
def discriminate(
        values: np.ndarray,
        weights: np.ndarray,
        discriminator: float
):
    """Discriminates values where the weights are below a certain threshold

    :param values: array
    :param weights: array
    :param discriminator: float
    :return: two arrays of discriminated values and weights
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

#
# def histogram1D(
#         pos,
#         data=None,
#         nbPt: int = 100
# ):
#     """
#     Calculates histogram of pos weighted by weights.
#
#     :param pos: 2Theta array
#     :param weights: array with intensities
#     :param bins: number of output bins
#     :param pixelSize_in_Pos: size of a pixels in 2theta
#     :param nthread: maximum number of thread to use. By default: maximum available.
#         One can also limit this with OMP_NUM_THREADS environment variable
#
#     :return 2theta, I, weighted histogram, raw histogram
#
#     https://github.com/kif/pyFAI/blob/master/src/histogram.pyx
#     """
#     if data is None:
#         data = np.ones(pos.shape[0], dtype=np.float64)
#     else:
#         assert pos.size == data.size
#     ctth = pos.astype("float64").flatten()
#     cdata = data.astype("float64").flatten()
#     outData = np.zeros(nbPt, dtype=np.float64)
#     outCount = np.zeros(nbPt, dtype=np.int64)
#     size = pos.size
#     tth_min = pos.min()
#     tth_max = pos.max()
#     idtth = (float(nbPt - 1)) / (tth_max - tth_min)
#     bin = 0
#     i = 0
#     for i in range(size):
#         bin = int(floor((float(ctth[i]) - tth_min) * idtth))
#         outCount[bin] += 1
#         outData[bin] += cdata[i]
#     return outData, outCount


def smooth(x, l, m):
    xz = np.empty(x.shape[0], dtype=np.float64)
    for i in range(l-m):
        xz[i] = 0
        for j in range(i-m, i+m):
            xz[i] += x[j]
            xz[i] /= (2 * m + 1)
    return xz


def interleaved_to_two_columns(
        ls: np.ndarray,
        sort: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Converts an interleaved spectrum to two column-data
    :param ls: numpy array
        The interleaved spectrum (amplitude, lifetime)
    :param sort: bool
        if True sort by the size of the lifetimes
    :return: two arrays (amplitudes), (lifetimes)

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
    """Converts two column lifetime spectra to interleaved lifetime spectra
    :param ls: The
    :return:
    """
    c = np.vstack((x, t)).reshape(-1, order='F')
    return c


@nb.jit(nopython=True, nogil=True)
def elte2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Takes two interleaved spectrum of lifetimes (a11, l11, a12, l12,...) and
    (a21, l21, a22, l22,...) and return a new spectrum of lifetimes of the
    form (a11*a21, 1/(1/l11+1/l21), a12*a22, 1/(1/l22+1/l22), ...)

    :param e1: array-like
        Lifetime spectrum 1
    :param e2: array-like
        Lifetime spectrum 2
    :return: array-like
        Lifetime-spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1,2,3,4])
    >>> e2 = np.array([5,6,7,8])
    >>> elte2(e1, e2)
    array([ 5.        ,  1.5       ,  7.        ,  1.6       , 15.        ,
            2.4       , 21.        ,  2.66666667])
    """
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.empty(n1*n2*2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = 1. / (1. / e1[i * 2 + 1] + 1. / e2[j * 2 + 1])
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def ere2(
        e1: np.array,
        e2: np.array
) -> np.array:
    """
    Takes two interleaved spectrum of rates (a11, r11, a12, r12,...) and
    (a21, r21, a22, r22,...) and return a new spectrum of lifetimes of the
    form (a11*a21, r11+r21), a12*a22, r22+r22), ...)

    :param e1: array-like
        Lifetime spectrum 1
    :param e2: array-like
        Lifetime spectrum 2
    :return: array-like
        Lifetime-spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([0.5,1,0.5,2])
    >>> e2 = np.array([0.5,3,0.5,4])
    >>> ere2(e1, e2)
    array([0.25, 4.  , 0.25, 5.  , 0.25, 5.  , 0.25, 6.  ])
    """
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.zeros(n1*n2*2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i * 2 + 0] * e2[j * 2 + 0]
            r[k * 2 + 1] = e1[i * 2 + 1] + e2[j * 2 + 1]
            k += 1
    return r


@nb.jit(nopython=True, nogil=True)
def invert_interleaved(
        interleaved_spectrum: np.array
) -> np.array:
    """Converts interleaved lifetime to rate spectra and vice versa

    :param interleaved_spectrum: array-like
        Lifetime/rate spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1, 2, 3, 4])
    >>> invert_interleaved(e1)
    array([1.  , 0.5 , 3.  , 0.25])
    """
    n1 = interleaved_spectrum.shape[0] // 2
    r = np.empty(n1*2, dtype=np.float64)

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
    Multiply amplitudes of interleaved rate/lifetime spectrum by float

    :param e1: array-like
        Rate spectrum
    :param n: float

    Examples
    --------

    >>> e1 = np.array([1,2,3,4])
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
    n1 = e1.shape[0] // 2
    n2 = e2.shape[0] // 2
    r = np.zeros(n1 * n2 * 2, dtype=np.float64)

    k = 0
    for i in range(n1):
        for j in range(n2):
            r[k * 2 + 0] = e1[i] * e2[j]
            r[k * 2 + 1] = e1[i + 1] * e2[j + 1]
            k += 1
    return r


def pairwise(iterable):
    """Iterate in pairs of 2 over iterable

    :param iterable:
    :return:
    """
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


