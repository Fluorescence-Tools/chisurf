import cython
import numpy as np
cimport numpy as np
cdef double eps = 1e-9
from libc.math cimport floor, ceil, round


def binCount(data, binWidth=16, binMin=0, binMax=4095):
    """
    Count number of occurrences of each value in array of non-negative ints.

    :param data: array_like
        1-dimensional input array
    :param binWidth:
    :param binMin:
    :param binMax:
    :return: As return values a numpy array with the bins and a array containing the counts is obtained.
    """
    # OK - rounding in C: rint, round
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


@cython.boundscheck(False)
def histogram1D(np.ndarray pos not None, np.ndarray data=None, long nbPt=100):
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
    cdef np.ndarray[np.float64_t, ndim = 1] ctth = pos.astype("float64").flatten()
    cdef np.ndarray[np.float64_t, ndim = 1] cdata = data.astype("float64").flatten()
    cdef np.ndarray[np.float64_t, ndim = 1] outData = np.zeros(nbPt, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] outCount = np.zeros(nbPt, dtype=np.int64)
    cdef long size = pos.size
    cdef double tth_min = pos.min()
    cdef double tth_max = pos.max()
    cdef double idtth = (< double > (nbPt - 1)) / (tth_max - tth_min)
    cdef long bin = 0
    cdef long i = 0
    # with nogil:
    for i in range(size):
        bin = <long> (floor(((<double> ctth[i]) - tth_min) * idtth))
        outCount[bin] += 1
        outData[bin] += cdata[i]
    return outData, outCount


def smooth(np.ndarray[np.float64_t, ndim=1] x, int l, int m):
    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim = 1] xz = np.empty(x.shape[0], dtype=np.float64)
    for i in range(l-m):
        xz[i] = 0
        for j in range(i-m, i+m):
            xz[i] += x[j]
            xz[i] /= (2 * m + 1)
    return xz

