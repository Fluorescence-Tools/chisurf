from __future__ import annotations
from chisurf import typing

import math
import numpy as np
import numba as nb




@nb.jit(nopython=True, nogil=True)
def poisson_0toN(
        lam: float,
        N: int
):
    """Poisson-distribution for the parameter lambda up to N

    :param lam: float
    :param N: integer value
    :return: numpy array

    Examples
    --------

    >>> import chisurf.math
    >>> r = chisurf.math.functions.distributions.poisson_0toN(0.2, 5)
    >>> r
    array([  8.18730753e-01,   3.27492301e-02,   1.30996920e-03,
         5.23987682e-05,   2.09595073e-06])

    """
    p = np.empty(N, dtype=np.float64)
    p[0] = math.exp(-lam)
    for i in range(1, N):
        p[i] = p[i-1] * lam / i
    return p


@nb.jit(nopython=True, nogil=True)
def normal_distribution(
        x: np.ndarray,
        loc: float = 0.0,
        scale: float = 1.0,
        norm: bool = True
):
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


@nb.jit(nopython=True, nogil=True)
def generalized_normal_distribution(
        x: np.ndarray,
        loc: float = 0.0,
        scale: float = 1.0,
        shape: float = 0.0,
        norm: bool = True
):
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
        if scale == 0.0:
            return np.zeros_like(x)
        else:
            t = 1.0 - shape * (x - loc) / scale
            t[t < 0.] = np.spacing(1)
            z = -1.0 / shape * np.log(t)

    y = normal_distribution(z, norm=False)

    if norm:
        y /= y.sum()
    return y


@nb.jit(nopython=True, nogil=True)
def linear_dist(
        x: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        normalize: bool = True
):
    """ Creates a distribution for a number of points and linearly interpolates
    between the points.

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


def sum_distribution(
        x_axis,
        dist_function,
        dist_args,
        weights: typing.List[float] = None,
        accumulate: bool = True,
        normalize: bool = False
):
    """Generates a sum of distribution functions, e.g. the sum of two normal distributions
    evaluated for a given x_axis. The arguments dist_args should be a list of lists, which
    is passed to the dist_function. 
    
    :param x_axis: 
    :param dist_function: function to be evaluated
    :param dist_args: arguments for the dist_function
    :param accumulate: If True the distributions are summed up and only a single joint distribution is returned
    :return: 
    
    Example
    -------
    >>> import chisurf.math
    >>> import pylab as p
    >>> pdf = chisurf.math.functions.distributions.normal_distribution
    >>> x_axis = np.linspace(0, 10, 100)
    >>> dist_args = [[3, 2], [5, 1]]
    >>> y_values = sum_distribution(x_axis, pdf, dist_args)
    >>> p.plot(x_axis, y_values)
    """
    if weights is None:
        weights = [1.] * len(dist_args)
    if accumulate:
        y_values = np.zeros_like(x_axis)
        for i, arg in enumerate(dist_args):
            y_values += weights[i] * dist_function(x_axis, *arg)
    else:
        y_values = list()
        for i, arg in enumerate(dist_args):
            y_values.append(weights[i] * dist_function(x_axis, *arg))
    if normalize:
        y_values /= y_values.sum()
    return y_values


def combine_distributions(
        x_axis,
        dist_function,
        dist_args,
        weights: typing.List[float] = None,
        accumulate: bool = True,
        normalize: bool = False
):
    """Generates a sum of distribution functions, e.g. the sum of two normal distributions
    evaluated for a given x_axis. The arguments dist_args should be a list of lists, which
    is passed to the dist_function.

    :param x_axis:
    :param dist_function: function to be evaluated
    :param dist_args: arguments for the dist_function
    :param accumulate: If True the distributions are summed up and only a single joint distribution is returned
    :return:

    Example
    -------
    >>> import scikit_fluorescence.math
    >>> import pylab as p
    >>> pdf = scikit_fluorescence.math.functions.distributions.normal_distribution
    >>> x_axis = np.linspace(0, 10, 100)
    >>> dist_args = [[3, 2], [5, 1]]
    >>> y_values = combine_distributions(x_axis, pdf, dist_args)
    >>> p.plot(x_axis, y_values)
    """
    if weights is None:
        weights = [1.] * len(dist_args)
    if accumulate:
        y_values = np.zeros_like(x_axis)
        for i, arg in enumerate(dist_args):
            y_values += weights[i] * dist_function(x_axis, *arg)
    else:
        y_values = list()
        for i, arg in enumerate(dist_args):
            y_values.append(weights[i] * dist_function(x_axis, *arg))
    if normalize:
        y_values /= y_values.sum()
    return y_values


def combine_distributions(
        x_axis,
        dist_function,
        dist_args,
        weights: typing.List[float] = None,
        accumulate: bool = True,
        normalize: bool = False
):
    """Generates a sum of distribution functions, e.g. the sum of two normal distributions
    evaluated for a given x_axis. The arguments dist_args should be a list of lists, which
    is passed to the dist_function.

    :param x_axis:
    :param dist_function: function to be evaluated
    :param dist_args: arguments for the dist_function
    :param accumulate: If True the distributions are summed up and only a single joint distribution is returned
    :return:

    Example
    -------
    >>> import scikit_fluorescence.math
    >>> import pylab as p
    >>> pdf = scikit_fluorescence.math.functions.distributions.normal_distribution
    >>> x_axis = np.linspace(0, 10, 100)
    >>> dist_args = [[3, 2], [5, 1]]
    >>> y_values = combine_distributions(x_axis, pdf, dist_args)
    >>> p.plot(x_axis, y_values)
    """
    if weights is None:
        weights = [1.] * len(dist_args)
    if accumulate:
        y_values = np.zeros_like(x_axis)
        for i, arg in enumerate(dist_args):
            y_values += weights[i] * dist_function(x_axis, *arg)
    else:
        y_values = list()
        for i, arg in enumerate(dist_args):
            y_values.append(weights[i] * dist_function(x_axis, *arg))
    if normalize:
        y_values /= y_values.sum()
    return y_values


@nb.jit(nopython=True)
def distance_between_gaussian(
        distances: np.array,
        separation_distance: float,
        sigma: float,
        normalize: bool = False
) -> np.array:
    """Calculates the distance distribution between two Gaussians

    :param distances:
    :param separation_distance:
    :param sigma:
    :param normalize:
    :return:
    """
    if separation_distance > 0:
        pr = distances / separation_distance * (
                normal_distribution(
                    x=distances,
                    loc=separation_distance,
                    scale=sigma,
                    norm=False
                ) -
                normal_distribution(
                    x=distances,
                    loc=-separation_distance,
                    scale=sigma,
                    norm=False
                )
        )
    else:
        pr = 2. * distances ** 2 / sigma ** 2 * normal_distribution(
            x=distances,
            loc=0.0,
            scale=sigma,
            norm=False
        )
    if normalize:
        s = pr.sum()
        if s > 0:
            pr /= s
    return pr
