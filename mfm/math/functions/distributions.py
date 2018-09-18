import numpy as np
from numba import jit
from math import exp


@jit(nopython=True, nogil=True)
def poisson_0toN(lam, N):
    """Poisson-distribution for the parameter lambda up to N

    :param lam: float
    :param N: integer value
    :return: numpy array

    Examples
    --------

    >>> import mfm
    >>> r = mfm.math.functions.distributions.poisson_0toN(0.2, 5)
    >>> r
    array([  8.18730753e-01,   3.27492301e-02,   1.30996920e-03,
         5.23987682e-05,   2.09595073e-06])

    """
    p = np.empty(N, dtype=np.float64)
    p[0] = exp(-lam)
    for i in range(1, N):
        p[i] = p[i-1] * lam / i
    return p


@jit(nopython=True, nogil=True)
def normal_distribution(x, loc=0.0, scale=1.0, norm=True):
    """Probability density function of a generalized normal distribution

    :param x:
    :param loc: mean (location)
    :param scale:
    :param norm: Boolean if true the returned array is normalized to one
    :return:
    """
    y = 1.0 / (np.sqrt(2.0 * np.pi) * scale) * np.exp(- (x - loc)**2 / (2. * scale**2))
    if norm:
        y /= y.sum()
    return y


@jit(nopython=True, nogil=True)
def generalized_normal_distribution(x, loc=0.0, scale=1.0, shape=0.0, norm=True):
    """ Probability density function of a generalized normal distribution in which a shape parameter
    introduces a skew. If the shape parameter is zero, the normal distribution
    results. Positive values of the shape parameter yield left-skewed distributions bounded to the right,
    and negative values of the shape parameter yield right-skewed distributions bounded to the left. Only
    when the shape parameter is zero is the density function for this distribution positive over the
    whole real line: in this case the distribution is a normal distribution, otherwise the distributions
    are shifted and possibly reversed log-normal distributions.

    :param x:
    :param loc: mean
    :param scale: the width of the normal distribution
    :param shape: a parameter of the "skewness"
    :param norm: Boolean if true the returned array is normalized to one
    :return:
    """
    if shape == 0.0:
        z = (x - loc) / scale
    else:
        t = 1.0 - shape * (x - loc) / scale
        t[t < 0.] = np.spacing(1)
        z = -1.0 / shape * np.log(t)

    y = normal_distribution(z, norm=False)

    if norm:
        y /= y.sum()
    return y


@jit(nopython=True, nogil=True)
def linear_dist(x, px, py, normalize=True):
    """ Creates a distribution for a number of points and linearly interpolates between the points.

    :param x: linear x-axis
    :param px: x-coordinates of points
    :param py: y-coordinates of points
    :return:
    """
    y = np.zeros(x.shape)
    dx = x[1] - x[0]
    offset = int(x[0] / dx)
    x_n_max = len(x) - 1

    idx = np.argsort(px)
    px = np.array(px)[idx]
    py = np.array(py)[idx]

    i = 0
    while i < len(px) - 1:
        i1 = max(0, int(px[i] / dx) - offset)
        x1 = px[i]
        y1 = py[i]

        i2 = max(0, int(px[i + 1] / dx) - offset)
        x2 = px[i + 1]
        y2 = py[i + 1]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - (y2 - y1) / (x2 - x1) * x1
        for j in range(i1, min(x_n_max, i2)):
            y[j] = m * x[j] + b
        i += 1
    if normalize:
        y /= y.sum()
    return y
