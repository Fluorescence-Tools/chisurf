from __future__ import annotations
from chisurf import typing

import scipy.stats
import numpy as np
import numba as nb
import scipy.special


def incremental_average(
        next_sample,
        previous_average=None,
        number_of_samples: int = 1,
        fractions: typing.Tupel[float, float] = None
):
    """
    Compute the running (online) average when a new sample is added.

    This function updates the average based on the previous average and the new sample.
    Optionally, the mixing fractions for the previous average and new sample can be specified;
    otherwise, the fractions are determined by the number of samples.

    Parameters
    ----------
    next_sample : float or array-like
        The new sample to be included in the average.
    previous_average : float or array-like, optional
        The previously computed average. If None, the new sample is returned as the average.
    number_of_samples : int, optional
        The total number of samples (including the new one). Default is 1.
    fractions : (float, float), optional
        A tuple (xp, xn) specifying the weights for the previous average and the new sample.
        If not provided, the weights are calculated as:
            xp = (number_of_samples - 1) / number_of_samples
            xn = 1 / number_of_samples

    Returns
    -------
    float or array-like
        The updated average after including the new sample.

    See Also
    --------
    http://datagenetics.com/blog/november22017/index.html
    http://www.heikohoffmann.de/htmlthesis/node134.html

    Examples
    --------
    >>> avg = incremental_average(5.0)  # With no previous average, returns 5.0
    >>> avg = incremental_average(7.0, previous_average=5.0, number_of_samples=2)
    >>> print(avg)
    6.0
    """
    if previous_average is None:
        number_of_samples = 1
    try:
        xp, xn = fractions
    except (TypeError, ValueError):
        xp, xn = (number_of_samples - 1) / number_of_samples, 1.0 / number_of_samples
    next_average = xp * previous_average + xn * next_sample
    return next_average


def incremental_variance(
        next_sample,
        previous_average=None,
        previous_variance=None,
        number_of_samples: int = 1
):
    """
    Update the running (online) average and variance with a new sample using Welford's algorithm.

    This function updates the running average and variance when a new sample is added. If there is no
    previous average (i.e. on the first sample), the new sample is taken as the initial average and
    the variance is set to 0. For subsequent samples, the running average is updated and the variance is
    updated based on the accumulated sum of squared differences.

    Parameters
    ----------
    next_sample : float or array-like
        The new sample to include in the running statistics.
    previous_average : float or array-like, optional
        The previous running average. If None, the new sample becomes the average.
    previous_variance : float or array-like, optional
        The previous running variance. If None, it is assumed to be 0.
    number_of_samples : int, optional
        The total number of samples after including the new sample. Default is 1.

    Returns
    -------
    tuple
        A tuple (new_average, new_variance) representing the updated running average and variance.

    Examples
    --------
    >>> # First sample: average equals the sample, variance is 0.
    >>> avg, var = incremental_variance(5.0)
    >>> print(avg, var)
    5.0 0.0
    >>> # Second sample update:
    >>> avg, var = incremental_variance(7.0, previous_average=5.0, previous_variance=0.0, number_of_samples=2)
    >>> print(avg, var)
    6.0 2.0
    """
    if previous_average is None:
        # No previous sample; initialize average and variance.
        return next_sample, 0.0

    # If previous_variance is not provided, assume it is 0.
    if previous_variance is None:
        previous_variance = 0.0

    # Update the average using the new sample.
    new_average = previous_average + (next_sample - previous_average) / number_of_samples

    # Calculate the cumulative sum of squares (S) from the previous variance.
    # For n samples, variance = S / (n - 1), so S = variance * (n - 1)
    if number_of_samples > 1:
        S = previous_variance * (number_of_samples - 1)
        # Update S with the contribution of the new sample.
        S_new = S + (next_sample - previous_average) * (next_sample - new_average)
        new_variance = S_new / (number_of_samples - 1)
    else:
        new_variance = 0.0

    return new_average, new_variance


def random_point_in_sphere(
        radius: float = 1.0,
        center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64),
        n_per_sphere: int = 1
) -> np.ndarray:
    """
    Generate random point(s) uniformly distributed within a sphere.

    Random points are generated using normally distributed coordinates that are then
    scaled so that their distribution is uniform within the sphere. The scaling makes use
    of the incomplete gamma function.

    Parameters
    ----------
    radius : float, optional
        The radius of the sphere. Default is 1.0.
    center : numpy.ndarray, optional
        The center of the sphere. Default is a 3D point at the origin.
    n_per_sphere : int, optional
        The number of random points to generate. If 1, a single point is returned; otherwise,
        an array of shape (n_per_sphere, ndim) is returned.

    Returns
    -------
    numpy.ndarray
        A single point (if n_per_sphere is 1) or an array of random points within the sphere.

    See Also
    --------
    https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume/23785326

    Examples
    --------
    >>> point = random_point_in_sphere(radius=2.0)
    >>> print(point)  # e.g., array([0.532, -1.234, 0.678])
    >>> points = random_point_in_sphere(radius=2.0, n_per_sphere=5)
    >>> print(points.shape)
    (5, 3)
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
        case='gaussian'
):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given model.

    For the 'gaussian' case, the BIC is computed using the maximized likelihood value.
    For other cases, the function computes a BIC-like score based on the log of the likelihood.

    Parameters
    ----------
    k : int
        The number of parameters estimated by the model.
    n : int
        The number of data points.
    value : float
        Either the maximized value of the likelihood function (for 'gaussian') or
        the reduced chi-squared (chi2/n) value.
    case : str, optional
        The case for which to compute the BIC. Default is 'gaussian'.

    Returns
    -------
    float
        The computed Bayesian Information Criterion (BIC).

    Examples
    --------
    >>> bic = bayesian_information_criterion(k=3, n=100, value=2.5, case='gaussian')
    >>> print(bic)
    100 * 2.5 + 3 * np.log(100)
    """
    if case == 'gaussian':
        return n * value + k * np.log(n)
    else:
        return k * np.log(n) - 2.0 * np.log(value)


@nb.jit(nopython=True)
def durbin_watson(
        residuals: np.array
) -> float:
    """
    Compute the Durbin-Watson statistic for a series of residuals.

    The Durbin-Watson statistic tests for the presence of autocorrelation at lag 1 in the residuals
    of a regression analysis. The value ranges from 0 to 4, where 2 indicates no autocorrelation.

    Parameters
    ----------
    residuals : numpy.array
        Array of residuals from a regression model.

    Returns
    -------
    float
        The Durbin-Watson statistic.

    References
    ----------
    Durbin, J. and Watson, G. S. (1950, 1951).
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
    """
    Compute the Kullback-Leibler divergence D(P || Q) for discrete probability distributions.

    The KL divergence measures the difference between two probability distributions P and Q.
    This implementation iterates over the minimum length of the two input distributions.

    Parameters
    ----------
    p : numpy.ndarray
        The first discrete probability distribution.
    q : numpy.ndarray
        The second discrete probability distribution.

    Returns
    -------
    float
        The computed Kullback-Leibler divergence.

    Notes
    -----
    If either p[i] or q[i] is zero, that term is skipped in the summation.
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
    """
    Calculate the maximum chi-squared value for a fit at a given confidence level.

    This function adjusts the provided chi-squared value based on the number of parameters,
    the degrees of freedom (nu), and the desired confidence level using the F-distribution.

    Parameters
    ----------
    chi2_value : float, optional
        The base chi-squared value. Default is 1.0.
    number_of_parameters : int, optional
        The number of model parameters. Default is 1.
    nu : int, optional
        The number of degrees of freedom (observations minus model parameters). Default is 1.
    conf_level : float, optional
        The confidence level used to determine the maximum chi-squared value. Default is 0.95.

    Returns
    -------
    float
        The maximum chi-squared value adjusted for the specified confidence level.

    Examples
    --------
    >>> chi2_threshold = chi2_max(chi2_value=1.0, number_of_parameters=3, nu=20, conf_level=0.95)
    >>> print(chi2_threshold)
    """
    return chi2_value * (
        1.0 + float(number_of_parameters) / nu *
        scipy.stats.f.isf(1. - conf_level, number_of_parameters, nu)
    )
