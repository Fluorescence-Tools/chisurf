from tcspc import *
from . import convolve


def weights(y):
    """
    Calculated the weights used in TCSPC. Poissonian noise (sqrt of counts)
    :param y: numpy-array
        the photon counts
    :return: numpy-array
        an weighting array for fittin
    """
    if min(y) >= 0:
        w = np.array(y, dtype=np.float64)
        w[w == 0.0] = 1e30
        w = 1. / np.sqrt(w)
    else:
        w = np.ones_like(y)
    return w


def weights_ps(p, s, fs):
    """Calculated the weights used in TCSPC. weight = 1/err(P+fs*S)

    :param p: parallel decay
    :param s: perpendicular decay
    :param fs: weight of perpendicular decay
    :return: err(p + fs * fs)
    """
    if min(p) >= 0 and min(s) >= 0:
        vp = np.sqrt(p)
        vs = np.sqrt(s)
        vt = np.maximum(np.sqrt(vp**2 + fs**2 * vs**2), 1.0)
        w = 1. / vt
    else:
        w = np.ones_like(p)
    return w


def fitrange(y, threshold=10.0, area=0.999):
    """
    Determines the fitting range based on the total number of photons to be fitted (fitting area).

    :param y: numpy-array
        a numpy array containing the photon counts
    :param threshold: float
        a threshold value. Lowest index of the fitting range is the first encountered bin with a photon count
        higher than the threshold.
    :param area: float
        The area which should be considered for fitting. Here 1.0 corresponds to all measured photons. 0.9 to 90%
        of the total measured photons.
    :return:
    """
    xmin = np.where(y > threshold)[0][0]
    cumsum = np.cumsum(y, dtype=np.float64)
    s = np.sum(y, dtype=np.float64)
    xmax = np.where(cumsum >= s * area)[0][0]
    if y[xmax] < threshold:
        xmax = len(y) - np.searchsorted(y[::-1], [threshold])[-1]
    return xmin, min(xmax, len(y) - 1)
