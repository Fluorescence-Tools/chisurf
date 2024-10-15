from __future__ import annotations
from chisurf import typing

import scipy.stats
import numpy as np
import numba as nb
import scipy.special


def incremental_average(
        next_sample,
        previous_average = None,
        number_of_samples: int = 1,
        fractions: typing.Tupel[float, float] = None
):
    """Compute the average online

    Parameters
    ----------
    previous_average
        The previous average over the sample
    next_sample
        The next sample that should be considered in the average
    number_of_samples : int
        The total number of samples
    fractions : (float, float)
        An optional parameter that overrides the fraction used to mix the
        previous sample average and the new sample that should be considered
        in the new average

    See Also
    --------
    http://datagenetics.com/blog/november22017/index.html and
    http://www.heikohoffmann.de/htmlthesis/node134.html

    Returns
    -------
    The average over the sample considering a newly added sample

    """
    if previous_average is None:
        number_of_samples = 1
    try:
        xp, xn = fractions
    except TypeError or ValueError:
        xp, xn = (number_of_samples - 1) / number_of_samples, 1. / number_of_samples
    next_average = xp * previous_average + xn * next_sample
    return next_average


def incremental_variance():
    pass


def random_point_in_sphere(
        radius: float = 1.0,
        center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64),
        n_per_sphere: int = 1
) -> np.ndarray:
    """Get random points uniformly sample in a sphere

    Parameters
    ----------
    radius : float
        Radius of the sphere
    center : numpy-array
        Center of the sphere
    n_per_sphere : int
        Number of random samples

    Returns
    -------
    array with random points in the sphere

    See Also
    --------
    https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume/23785326

    Examples
    --------
    >>> import pylab as plt
    >>> fig1 = plt.figure(1)
    >>> ax1 = fig1.gca()
    >>> center = np.array([0, 0])
    >>> radius = 1
    >>> p = sample(center, radius, 10000)
    >>> ax1.scatter(p[:, 0], p[:, 1], s=0.5)
    >>> ax1.add_artist(plt.Circle(center, radius, fill=False, color='0.5'))
    >>> ax1.set_xlim(-1.5, 1.5)
    >>> ax1.set_ylim(-1.5, 1.5)
    >>> ax1.set_aspect('equal')
    >>> plt.show()

    """
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * scipy.special.gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    if n_per_sphere == 1:
        return p[0]
    return p


def bayesian_information_criterion(
        k: int,
        n: int,
        value: float,
        case = 'gaussian'
):
    """NOT TESTED

    Parameters
    ----------
    n: number of data points
    k: number of parameters estimated by the model
    value: either the maximized value of the likelihood function of the model or
    the reduced chi2 = 1/n * chi2

    Returns
    -------
    Bayesian information criterion (BIC) or Schwarz information criterion

    """
    if case == 'gaussian':
        return n * value + k * np.log(n)
    else:
        return k * np.log(n) - 2.0 * np.log(value)


@nb.jit(nopython=True)
def durbin_watson(
        residuals: np.array
) -> float:
    """Durbin-Watson parameter (1950,1951)

    :param residuals:  array
    :return:
    """
    n_res = len(residuals)
    nom = 0.0
    denomminator = float(np.sum(residuals ** 2))
    for i in range(1, n_res):
        nom += (residuals[i] - residuals[i - 1]) ** 2
    return nom / max(1.0, denomminator)


@nb.jit(nopython=True)
def kl(
        p: np.ndarray,
        q: np.ndarray
):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    n_min = min(p.shape[0], q.shape[0])
    s = 0.0
    for i in range(n_min):
        pi, qi = p[i], q[i]
        s += qi
        if pi > 0 and qi > 0:
            s += pi * np.log(pi / qi) - pi
    return s



def chi2_max(
        chi2_value: float = 1.0,
        number_of_parameters: int = 1,
        nu: int = 1,
        conf_level: float = 0.95
) -> float:
    """Calculate the maximum chi2r of a fit given a certain confidence level

    :param chi2_value: the chi2 value
    :param number_of_parameters: the number of parameters of the models
    :param conf_level: the confidence level that is used to calculate the
    maximum chi2
    :param nu: the number of free degrees of freedom (number of observations
    - number of models parameters)
    """
    return chi2_value * (
            1.0 + float(
        number_of_parameters
        ) / nu *
            scipy.stats.f.isf(
                1. - conf_level, number_of_parameters, nu
            )
    )

