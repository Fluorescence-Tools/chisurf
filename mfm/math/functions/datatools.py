from __future__ import annotations
from typing import Tuple

import copy
from math import floor

import numba as nb
import numpy as np


def histogram_rebin(
        bin_edges: np.array,
        counts: np.array,
        new_bin_edges: np.array
):
    """
    Extrapolation of a histogram to a new x-axis. Here the parameter x can be any
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
    >>> new_bin_edges
    array([ -5.    ,  -3.4375,  -1.875 ,  -0.3125,   1.25  ,   2.8125,
             4.375 ,   5.9375,   7.5   ,   9.0625,  10.625 ,  12.1875,
            13.75  ,  15.3125,  16.875 ,  18.4375,  20.    ])
    >>> y = histogram_rebin(bin_edges, counts, new_bin_edges)
    [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0.0, 0.0, 0.0, 0.0]
    """
    re = []
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
        The tuple should consist of the x and y values. Whereas the x and y values have to be a numpy array.
    :param dataset2: tuple
        The tuple should consist of the x and y values. Whereas the x and y values have to be a numpy array.
    :return: Two tuples
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
        dataset1: Tuple[np.array, np.array],
        dataset2: Tuple[np.array, np.array],
        method: str = 'linear-close'
) -> Tuple[
    Tuple[np.array, np.array],
    Tuple[np.array, np.array]
]:
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
                ny[j] = ry[i:].mean_xyz()
                                        
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


if __name__ == "__main__":
    print("Test align_x_spacing")
    import matplotlib.pylab as p
    x1 = np.linspace(-1, 12, 10)
    y1 = np.cos(x1)
    a1 = (x1, y1)
    
    x2 = np.linspace(-1, 12, 11)
    y2 = np.sin(x2)
    a2 = (x2, y2)
    
    (rx1, ry1), (rx2, ry2) = align_x_spacing(a1, a2)
    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()
    
    print("Test overlay")
    import matplotlib.pylab as p
    x1 = np.linspace(-5, 5, 10)
    y1 = np.sin(x1)
    a1 = (x1, y1)
    
    x2 = np.linspace(0, 10, 10)
    y2 = np.sin(x2)
    a2 = (x2, y2)
    
    (rx1, ry1), (rx2, ry2) = overlapping_region(a1, a2)
    print(ry1)
    print(ry2)

    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()    


def binCount(
        data,
        binWidth: int = 16,
        binMin: int = 0,
        binMax: int = 4095
):
    """
    Count number of occurrences of each value in array of non-negative ints.

    :param data: array_like
        1-dimensional input array
    :param binWidth:
    :param binMin:
    :param binMax:
    :return: As return values a numpy array with the bins and a array containing the counts is obtained.
    """
    nMin = np.rint(binMin / binWidth)
    nMax = np.rint(binMax / binWidth)
    nBins = nMax - nMin
    count = np.zeros(nBins, dtype=np.float32)
    bins = np.arange(nMin, nMax, dtype=np.float32)
    bins *= binWidth
    for i in range(data.shape[0]):
        bin = np.rint((data[i] / binWidth)) - nMin
        if bin < nBins:
            count[bin] += 1
    return bins, count


@nb.jit(nopython=True, nogil=True)
def minmax(
        x,
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
    """ Creates a histogram of the values the histogram is linear between the minium and the maximum value

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
def discriminate(values, weights, discriminator):
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


def histogram1D(
        pos,
        data=None,
        nbPt: int = 100
):
    """
    Calculates histogram of pos weighted by weights.

    :param pos: 2Theta array
    :param weights: array with intensities
    :param bins: number of output bins
    :param pixelSize_in_Pos: size of a pixels in 2theta
    :param nthread: maximum number of thread to use. By default: maximum available.
        One can also limit this with OMP_NUM_THREADS environment variable

    :return 2theta, I, weighted histogram, raw histogram

    https://github.com/kif/pyFAI/blob/master/src/histogram.pyx
    """
    if data is None:
        data = np.ones(pos.shape[0], dtype=np.float64)
    else:
        assert pos.size == data.size
    ctth = pos.astype("float64").flatten()
    cdata = data.astype("float64").flatten()
    outData = np.zeros(nbPt, dtype=np.float64)
    outCount = np.zeros(nbPt, dtype=np.int64)
    size = pos.size
    tth_min = pos.min()
    tth_max = pos.max()
    idtth = (float(nbPt - 1)) / (tth_max - tth_min)
    bin = 0
    i = 0
    for i in range(size):
        bin = int(floor((float(ctth[i]) - tth_min) * idtth))
        outCount[bin] += 1
        outData[bin] += cdata[i]
    return outData, outCount


def smooth(x, l, m):
    xz = np.empty(x.shape[0], dtype=np.float64)
    for i in range(l-m):
        xz[i] = 0
        for j in range(i-m, i+m):
            xz[i] += x[j]
            xz[i] /= (2 * m + 1)
    return xz

