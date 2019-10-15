"""
Functions related to random numbers
"""

from math import sqrt

import numba as nb
import numpy as np
from scipy.stats import norm


@nb.jit
def weighted_choice(weights, n=1):
    """
    A weighted random number generator. The random number generator generates
    random numbers between zero and the length of the provided weight-array. The
    elements of the weight-arrays are proportional to the probability that the corresponding
    integer random number is generated.

    :param weights: array-like
    :param n: int
        number of random values to be generated
    :return: Returns an array containing the random values

    Examples
    --------

    >>> import numpy as np
    >>> weighted_choice(np.array([0.1, 0.5, 3]), 10)
    array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=uint32)

    http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
    """
    running_total = 0.0

    nWeights = weights.shape[0]

    r = np.empty(n, dtype=np.uint32)
    totals = np.empty(nWeights, dtype=np.float64)

    for i in range(nWeights):
        running_total += weights[i]
        totals[i] = running_total

    for j in range(n):
        rnd = np.random.ranf() * running_total
        for i in range(nWeights):
            if rnd <= totals[i]:
                r[j] = i
                break
    return r


def brownian(
        x0: np.ndarray,
        n: int,
        dt: float,
        delta: float,
        out=None
):
    """\
    Generate an instance of Brownian motion (i.e. the Wiener process):

    .. math::
        X(t) = X(0) + N(0, \delta^2 \cdot t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

    .. math::
        X(t + dt) = X(t) + N(0, \delta^2 \cdot dt; t, t+dt)


    If :math:`x_0` is an array (or array-like), each value in :math:`x_0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than :math:`x_0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


@nb.jit()
def mc(
        e0: float,
        e1: float,
        kT: float
) -> bool:
    """Accept or reject a candidate in Metropolis-Algorithm by its energy
    and the energy of its predecessors using the Bennett acceptance ratio.

    (1) calculates the acceptance probability p_A, a candidate is accepted if it
    (2) generate a uniform random number u on [0,1],
    (3) and accepts a candidate if u <= alpha, rejects a candidate if u > alpha.

    .. math::
        p_A = min(1, exp( (E_1 - E_0) / kT))

    (see: https://de.wikipedia.org/wiki/Metropolis-Algorithmus)

    :param e0: float
        Previous energy
    :param e1: float
        Candidate energy
    :param kT: float
        Temperature
    :return: True if a candidate is accepted, False is a candidate is
    rejected.

    """
    if e1 < e0:
        return True
    else:
        return np.random.ranf() < np.exp((e0 - e1) / kT)


@nb.jit(nopython=True)
def random_numbers(
        cdf_axis,
        cdf_values: np.ndarray,
        n: int,
        norm_cdf=True, dtype=np.float64
) -> np.ndarray:
    """Generates an array of n random numbers according to an cumulative
    distribution function (CDF)

    :param cdf_axis: x-axis of cdf
    :param cdf_values: CDF according to which random numbers are generated
    :param n: the number of random numbers to be generated
    :param norm_cdf: if True the array passed as cdf is normalized so that
    its last point is one
    :return:

    Examples
    --------

    >>> import chisurf.math
    >>> import pylab as p
    >>> x = np.linspace(0, 8, num=1000)
    >>> y = chisurf.math.functions.distributions.normal_distribution(x, loc=4)
    >>> rn = chisurf.math.rand.random_numbers(axis=x, cdf=np.cumsum(y), n=10000000, norm_cdf=True)
    >>> hy, hx = np.histogram(rn, bins=4096, range=(0, 10))
    >>> p.plot(hx[1:], hy)
    """
    if norm_cdf:
        # use the last point of the CDF for normalization
        # at the latest time a sample is taken for sure...
        cdf_values /= cdf_values[-1]
    tr = np.zeros(n, dtype=dtype)
    for j, r in enumerate(
            np.random.random_sample(n)
    ):
        for i, xi in enumerate(cdf_values):
            if xi >= r:
                tr[j] = cdf_axis[i]
                break
    return tr
