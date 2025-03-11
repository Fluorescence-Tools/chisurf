from __future__ import annotations
from chisurf import typing

import math
import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True)
def poisson_0toN(lam: float, N: int):
    """
    Compute Poisson distribution probabilities for k = 0 to N-1 for a given lambda.

    This function calculates the probability mass function of the Poisson distribution
    for k values from 0 up to N-1 using the recursive relation:
      p(0) = exp(-lam)
      p(k) = p(k-1) * lam / k, for k >= 1

    Parameters
    ----------
    lam : float
        The lambda (rate) parameter of the Poisson distribution.
    N : int
        The number of terms (from 0 to N-1) to compute.

    Returns
    -------
    numpy.ndarray
        An array of probabilities corresponding to k = 0, 1, ..., N-1.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=8)
    >>> poisson_0toN(0.2, 5)
    array([8.18730753e-01, 1.63746151e-01, 1.63746151e-02, 1.09164100e-03,
           5.45820502e-05])
    """
    p = np.empty(N, dtype=np.float64)
    p[0] = math.exp(-lam)
    for i in range(1, N):
        p[i] = p[i - 1] * lam / i
    return p


@nb.jit(nopython=True, nogil=True)
def normal_distribution(x: np.ndarray, loc: float = 0.0, scale: float = 1.0, norm: bool = True):
    """
    Compute the normal (Gaussian) probability density function (PDF) for given x values.

    The function evaluates the PDF of a normal distribution with mean `loc` and standard
    deviation `scale` at each element in the array `x`. If `norm` is True, the resulting
    array is normalized such that its sum equals 1, which is useful when dealing with
    discretized distributions.

    Parameters
    ----------
    x : numpy.ndarray
        Array of values where the PDF is evaluated.
    loc : float, optional
        The mean (location) of the normal distribution (default is 0.0).
    scale : float, optional
        The standard deviation (scale) of the normal distribution (default is 1.0).
    norm : bool, optional
        If True, the returned PDF values are normalized to sum to 1 (default is True).

    Returns
    -------
    numpy.ndarray
        An array of PDF values for the normal distribution evaluated at `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([-1, 0, 1])
    >>> normal_distribution(x, loc=0, scale=1, norm=False)
    array([0.24197072, 0.39894228, 0.24197072])
    >>> np.round(normal_distribution(x, loc=0, scale=1, norm=True), 6)
    array([0.274069, 0.451863, 0.274069])
    """
    y = 1.0 / (np.sqrt(2.0 * np.pi) * scale) * np.exp(- (x - loc)**2 / (2.0 * scale**2))
    if norm:
        y /= y.sum()
    return y


@nb.jit(nopython=True, nogil=True)
def generalized_normal_distribution(x: np.ndarray, loc: float = 0.0, scale: float = 1.0,
                                      shape: float = 0.0, norm: bool = True):
    """
    Compute the probability density function (PDF) of a generalized normal distribution
    with an added skew parameter.

    When `shape` is zero, the function returns the standard normal distribution. For
    nonzero `shape`, the distribution is modified to introduce skewness:
      - Positive `shape` values yield left-skewed distributions (bounded to the right).
      - Negative `shape` values yield right-skewed distributions (bounded to the left).

    Note that for nonzero `shape`, the PDF is positive only over a limited domain and may
    represent a shifted (and possibly reversed) log-normal distribution.

    Parameters
    ----------
    x : numpy.ndarray
        Array of values where the PDF is evaluated.
    loc : float, optional
        The mean (location) of the distribution (default is 0.0).
    scale : float, optional
        The width (scale) of the distribution (default is 1.0).
    shape : float, optional
        The skewness parameter. When zero, the distribution is normal (default is 0.0).
    norm : bool, optional
        If True, the returned PDF values are normalized to sum to 1 (default is True).

    Returns
    -------
    numpy.ndarray
        An array of PDF values for the generalized normal distribution evaluated at `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-3, 3, 5)
    >>> # When shape is 0, it reduces to the standard normal PDF.
    >>> np.round(generalized_normal_distribution(x, loc=0, scale=1, shape=0, norm=False), 6)
    array([0.004432, 0.129518, 0.398942, 0.129518, 0.004432])
    """
    if shape == 0.0:
        z = (x - loc) / scale
    else:
        if scale == 0.0:
            return np.zeros_like(x)
        else:
            t = 1.0 - shape * (x - loc) / scale
            t[t < 0.0] = np.spacing(1)
            z = -1.0 / shape * np.log(t)

    y = normal_distribution(z, norm=False)

    if norm:
        y /= y.sum()
    return y


def linear_dist(x: np.ndarray, px: np.ndarray, py: np.ndarray, normalize: bool = True):
    """
    Create a distribution by linearly interpolating between specified control points.

    Given an x-axis array `x` and control points defined by (`px`, `py`), the function
    linearly interpolates between each adjacent pair of control points. The resulting array
    represents a distribution over `x`. If `normalize` is True, the output is normalized
    so that its sum equals 1.

    Parameters
    ----------
    x : numpy.ndarray
        A 1D array representing the x-axis over which the distribution is defined.
    px : array-like
        The x-coordinates of the control points.
    py : array-like
        The y-coordinates corresponding to the control points.
    normalize : bool, optional
        Whether to normalize the resulting distribution (default is True).

    Returns
    -------
    numpy.ndarray
        An array representing the interpolated distribution over `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2])
    >>> px = np.array([0, 2])
    >>> py = np.array([0, 2])
    >>> linear_dist(x, px, py, False)
    array([0., 1., 0.])
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
        b = y1 - m * x1
        for j in range(i1, min(x_n_max, i2)):
            y[j] = m * x[j] + b
        i += 1
    if normalize:
        y /= y.sum()
    return y


def sum_distribution(x_axis, dist_function, dist_args, weights: typing.List[float] = None,
                     accumulate: bool = True, normalize: bool = False):
    """
    Generate a combined distribution by summing multiple individual distributions.

    This function evaluates a given distribution function (`dist_function`) on an x-axis (`x_axis`)
    for several sets of parameters specified in `dist_args`. Each distribution can be weighted by an
    optional list of `weights`. When `accumulate` is True, the function returns a single distribution
    that is the sum of all weighted distributions. Otherwise, it returns a list of individual
    weighted distributions.

    Parameters
    ----------
    x_axis : array-like
        The x-axis values at which to evaluate the distributions.
    dist_function : callable
        The distribution function to evaluate.
    dist_args : list of lists
        A list where each sublist contains the arguments to be passed to `dist_function`.
    weights : list of float, optional
        Weights for each distribution. If None, all distributions have equal weight.
    accumulate : bool, optional
        If True, returns the sum of the weighted distributions; if False, returns a list of them.
        (default is True)
    normalize : bool, optional
        If True, the resulting distribution(s) are normalized to sum to 1 (default is False).

    Returns
    -------
    numpy.ndarray or list
        A single numpy array if `accumulate` is True, or a list of numpy arrays otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> x_axis = np.linspace(0, 10, 11)
    >>> def dummy_pdf(x, a, b):
    ...     return a * x + b
    >>> # Two distributions: one increasing (dummy_pdf(x,1,0)=x) and one constant (dummy_pdf(x,0,1)=1).
    >>> result = sum_distribution(x_axis, dummy_pdf, [[1, 0], [0, 1]], weights=[1, 1], accumulate=True)
    >>> np.allclose(result, (x_axis + 1))
    True
    """
    if weights is None:
        weights = [1.] * len(dist_args)
    if accumulate:
        y_values = np.zeros_like(x_axis)
        for i, arg in enumerate(dist_args):
            y_values += weights[i] * dist_function(x_axis, *arg)
    else:
        y_values = []
        for i, arg in enumerate(dist_args):
            y_values.append(weights[i] * dist_function(x_axis, *arg))
    if normalize:
        y_values /= y_values.sum()
    return y_values


def combine_distributions(x_axis, dist_function, dist_args, weights: typing.List[float] = None,
                          accumulate: bool = True, normalize: bool = False):
    """
    Combine multiple distribution functions by summing them.

    This function is functionally equivalent to `sum_distribution` and evaluates the provided
    `dist_function` on the given `x_axis` for each set of parameters in `dist_args`. Each
    distribution is optionally weighted by the corresponding value in `weights`. The resulting
    distributions are either accumulated into a single distribution or returned as a list, based on
    the `accumulate` flag.

    Parameters
    ----------
    x_axis : array-like
        The x-axis values at which to evaluate the distributions.
    dist_function : callable
        The distribution function to evaluate.
    dist_args : list of lists
        A list where each sublist contains the arguments to be passed to `dist_function`.
    weights : list of float, optional
        Weights for each distribution. If None, all distributions have equal weight.
    accumulate : bool, optional
        If True, returns the sum of all weighted distributions; if False, returns a list of them.
        (default is True)
    normalize : bool, optional
        If True, the resulting distribution(s) are normalized to sum to 1 (default is False).

    Returns
    -------
    numpy.ndarray or list
        A combined numpy array if `accumulate` is True, or a list of numpy arrays otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> x_axis = np.linspace(0, 10, 11)
    >>> def dummy_pdf(x, a, b):
    ...     return a * x + b
    >>> result = combine_distributions(x_axis, dummy_pdf, [[1, 0], [0, 1]], weights=[1, 1], accumulate=True)
    >>> np.allclose(result, (x_axis + 1))
    True
    """
    if weights is None:
        weights = [1.] * len(dist_args)
    if accumulate:
        y_values = np.zeros_like(x_axis)
        for i, arg in enumerate(dist_args):
            y_values += weights[i] * dist_function(x_axis, *arg)
    else:
        y_values = []
        for i, arg in enumerate(dist_args):
            y_values.append(weights[i] * dist_function(x_axis, *arg))
    if normalize:
        y_values /= y_values.sum()
    return y_values


@nb.jit(nopython=True)
def distance_between_gaussian(distances: np.array, separation_distance: float, sigma: float,
                              normalize: bool = False) -> np.array:
    """
    Calculate the distance distribution between two Gaussian distributions.

    For a positive `separation_distance`, the function computes a weighted difference of two
    normal distributions (centered at `separation_distance` and `-separation_distance`) scaled
    by the ratio of the distance value to the separation distance. When `separation_distance`
    is zero, the distribution corresponds to a single Gaussian modified by a factor of
    2 * (distances^2 / sigma^2).

    Parameters
    ----------
    distances : numpy.array
        Array of distance values at which to evaluate the distribution.
    separation_distance : float
        The separation between the centers of the two Gaussian distributions.
    sigma : float
        The standard deviation of the Gaussians.
    normalize : bool, optional
        If True, the resulting distribution is normalized so that its sum equals 1
        (default is False).

    Returns
    -------
    numpy.array
        An array representing the computed distance distribution.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> # Without normalization
    >>> np.round(distance_between_gaussian(x, 2, 1, normalize=False), 6)
    array([0.      , 0.118769, 0.398808, 0.362954])
    >>> # With normalization, the sum equals 1
    >>> np.round(distance_between_gaussian(x, 2, 1, normalize=True).sum(), 6)
    1.0
    """
    if separation_distance > 0:
        pr = distances / separation_distance * (
            normal_distribution(x=distances, loc=separation_distance, scale=sigma, norm=False) -
            normal_distribution(x=distances, loc=-separation_distance, scale=sigma, norm=False)
        )
    else:
        pr = 2.0 * distances**2 / sigma**2 * normal_distribution(x=distances, loc=0.0, scale=sigma, norm=False)
    if normalize:
        s = pr.sum()
        if s > 0:
            pr /= s
    return pr
