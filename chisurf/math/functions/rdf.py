from __future__ import annotations

from math import exp, log

import numpy as np
import numba as nb

from chisurf.math.functions.special import i0
from . import distributions


@nb.jit(nopython=True)
def gaussian_chain_ree(
        segment_length: float,
        number_of_segments: int
) -> float:
    """Calculates the root mean square end-to-end distance of a Gaussian chain

    :param segment_length: float
        The length of a segment
    :param number_of_segments: int
        The number of segments
    :return:
    """
    return segment_length * np.sqrt(number_of_segments)


@nb.jit(nopython=True)
def gaussian_chain(
        r,
        segment_length: float,
        number_of_segments: int
) -> float:
    """Calculates the radial distribution function of a Gaussian chain in three dimensions

    :param number_of_segments: int
        The number of segments
    :param segment_length: float
        The segment length
    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1

    ..plot:: plots/rdf-gauss.py

    """
    r2_mean = gaussian_chain_ree(segment_length, number_of_segments) ** 2
    return 4*np.pi*r**2/(2./3. * np.pi*r2_mean)**(3./2.) * np.exp(-3./2. * r**2 / r2_mean)


@nb.jit(nopython=True)
def Qd(
        r,
        kappa
) -> float:
    return pow((3.0 / (4.0 * 3.14159265359 * kappa)), (3.0 / 2.0)) * \
           exp(-3.0 / 4.0 * r * r / kappa) * \
           (1.0 - 5.0 / 4.0 * kappa + 2.0 * r * r - 33.0 / 80.0 * r * r * r * r / kappa)


@nb.jit(nopython=True)
def worm_like_chain(
        distances: np.array,
        kappa: float,
        chain_length: float = 0.0,
        normalize: bool = True,
        distance=True
):
    """Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution
    according to:

    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    Parameters
    ----------
    distances: a vector at which the pdf is evaluated.
    kappa: a parameter describing the stiffness (details see publication)
    chain_length: the total length of the chain.
    normalize: If this is True the sum of the returned pdf vector is normalized to one.
    distance: If this is False, the end-to-end vector distribution is calculated. If True the distribution 
    the pdf is integrated over a sphere, i.e., the pdf of the end-to-end distribution function 
    is multiplied with 4*pi*r**2.

    Returns
    -------
    An array of the pdf

    Examples
    --------

    >>> import chisurf.math.functions.rdf as rdf
    >>> import numpy as np
    >>> r = np.linspace(0, 0.99, 50)
    >>> kappa = 1.0
    >>> rdf.worm_like_chain(r, kappa)
    array([  4.36400392e-06,   4.54198260e-06,   4.95588702e-06,
             5.64882576e-06,   6.67141240e-06,   8.09427111e-06,
             1.00134432e-05,   1.25565315e-05,   1.58904681e-05,
             2.02314725e-05,   2.58578047e-05,   3.31260228e-05,
             4.24918528e-05,   5.45365051e-05,   7.00005025e-05,
             8.98266752e-05,   1.15215138e-04,   1.47693673e-04,
             1.89208054e-04,   2.42238267e-04,   3.09948546e-04,
             3.96381668e-04,   5.06711496e-04,   6.47572477e-04,
             8.27491272e-04,   1.05745452e-03,   1.35165891e-03,
             1.72850634e-03,   2.21192991e-03,   2.83316807e-03,
             3.63314697e-03,   4.66568936e-03,   6.00184475e-03,
             7.73573198e-03,   9.99239683e-03,   1.29382877e-02,
             1.67949663e-02,   2.18563930e-02,   2.85090497e-02,
             3.72510109e-02,   4.86977611e-02,   6.35415230e-02,
             8.23790455e-02,   1.05199154e-01,   1.30049143e-01,
             1.49953168e-01,   1.47519190e-01,   9.57787954e-02,
             1.45297018e-02,   1.53180248e-08])

    References
    ----------

    .. [1] Becker NB, Rosa A, Everaers R, Eur Phys J E Soft Matter, 2010 May;32(1):53-69,
       The radial distribution function of worm-like chains.

    """
    if chain_length == 0.0:
        chain_length = np.max(distances)

    k = kappa
    a = 14.054
    b = 0.473
    c = 1.0 - (1.0+(0.38*k**(-0.95))**(-5.))**(-1./5.)
    pr = np.zeros_like(distances, dtype=np.float64)

    if k < 0.125:
        d = k + 1.0
    else:
        d = 1.0 - 1.0/(0.177/(k-0.111)+6.4 * exp(0.783 * log(k-0.111)))

    for i in range(len(distances)):
        r = distances[i]
        if r < chain_length:
            r /= chain_length

            pri = ((1.0 - c * r**2.0) / (1.0 - r**2.0))**(5.0 / 2.0)
            pri *= np.exp(-d * k * a * b * (1.0 + b) / (1.0 - (b*r)**2.0) * r**2.0)

            g = (((-3./4.) / k - 1./2.) * r**2. + ((-23./64.) / k + 17./16.) * r**4. + ((-7./64.) / k - 9./16.) * r**6.)
            pri *= exp(g / (1.0 - r**2.0))
            pri *= i0(-d*k*a*(1+b)*r/(1-(b*r)**2))
            pr[i] = pri
        else:
            break

    if normalize:
        pr /= pr.sum()

    return pr


@nb.jit(nopython=True)
def distance_between_gaussian(
        distances: np.array,
        separation_distance: float,
        sigma: float,
        normalize: bool = False
) -> np.array:
    """Calculates the distance distribution between two separated Gaussians a distance

    :param distances:
    :param separation_distance:
    :param sigma:
    :param normalize:
    :return:
    """
    if separation_distance > 0:
        pr = distances / separation_distance * (
                distributions.normal_distribution(
                    x=distances,
                    loc=separation_distance,
                    scale=sigma,
                    norm=False
                ) -
                distributions.normal_distribution(
                    x=distances,
                    loc=-separation_distance,
                    scale=sigma,
                    norm=False
                )
        )
    else:
        pr = 2. * distances ** 2 / sigma ** 2 * distributions.normal_distribution(
            x=distances,
            loc=0.0,
            scale=sigma,
            norm=False
        )
    if normalize:
        pr /= pr.sum()
    return pr


@nb.jit(nopython=True)
def worm_like_chain_linker(
        distances: np.array,
        kappa: float,
        chain_length: float = 0.0,
        sigma: float = 6.0,
        normalize: bool = True
) -> np.array:
    """
    Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution
    according to:

    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    Additionally the broadening by the dye-linkers is considered

    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1
    :param kappa: float

    Examples
    --------

    .. [1] Becker NB, Rosa A, Everaers R, Eur Phys J E Soft Matter, 2010 May;32(1):53-69,
       The radial distribution function of worm-like chains.

    """
    pr = worm_like_chain(
        distances=distances,
        kappa=kappa,
        chain_length=chain_length
    )
    pn = np.zeros_like(pr)
    for r, p in zip(distances, pr):
        pn += p * distance_between_gaussian(
            distances=distances,
            separation_distance=r,
            sigma=sigma
        )

    if normalize:
        pn /= pn.sum()
    return pn

