"""

"""
from __future__ import annotations

from chisurf import typing
import numpy as np

import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.tcspc.corrections
from .tcspc import rescale_w_bg


def counting_noise(
        decay: np.ndarray,
        treat_zeros: bool = True,
        zero_value: float = 1e12
) -> np.array:
    """Calculated Poisson noise (sqrt of counts) for TCSPC fluorescence decays

    Parameters
    ----------
    decay : numpy-array
        the photon counts
    treat_zeros: bool
        If treat zeros is True (default) a number specified by `zero_value`
        will be assigned for cases the decay is zero.
    zero_value : float
        The number that will be assigned cases where the value of the decay is
        zero. 

    Returns
    -------
    numpy-array
        an array containing noise estimates of the decay that can ve used in
        the data analysis.

    """
    w = np.array(decay, dtype=np.float64)
    if treat_zeros:
        w[w <= 0.0] = zero_value
    return np.sqrt(w)


def counting_noise_combined_parallel_perpendicular(
        parallel: np.ndarray,
        perpendicular: np.ndarray,
        g_factor: float
) -> np.ndarray:
    """Computes the combined counting weights (1/noise) for two counting
    channels that are added with a scale parameter.

    The weight is calculated a combined decay total that was computed using two
    counting channels `parallel` and `perpendicular` that were added using
    a scale parameter `scale_perpendicular`

    .. math::

         f(t)_{combined} = f(t)_{parallel} + 2.0 * g * f(t)_{perpendicular}

    The noise of the combined decay is computed assuming that in the individual
    noise is Poissonian counting noise

    .. math::

         \Delta(f(t)_{parallel}) = \sqrt{f(t)_{parallel}}
         \Delta(f(t)_{parallel}) = \sqrt{f(t)_{parallel}}

    Parameters
    ----------
    parallel : numpy-array
        parallel model_decay
    perpendicular : numpy-array
        perpendicular model_decay
    g_factor : float
        weight of perpendicular model_decay

    Returns
    -------
    numpy-array
        The weights of the channels of a decay combined.

    """
    scale_perpendicular = 2.0 * g_factor
    vp = np.sqrt(parallel)
    vs = np.sqrt(perpendicular)
    vt = np.maximum(np.sqrt(vp ** 2 + scale_perpendicular ** 2 * vs ** 2), 1.0)
    return vt


def combine_parallel_perpendicular(
        parallel: np.ndarray,
        perpendicular: np.ndarray,
        g_factor: float
) -> np.ndarray:
    """Combine a parallel and a perpendicular decay to yield a anisotropy
    free total decay.

    The parallel decay defined by the parameter `parallel` and the perpendicular
    decay specified by the parameter `perpendicular` are added weighted by a correction
    factor that accounts for the detection efficiencies of for the two decays to
    yield a total fluorescence decay that is polarization free.

    .. math::

         \Delta(f(t)_{total}) = f(t)_{parallel} + 2 * g * f(t)_{perpendicular}

    The combined decay :math:`f(t)_{total}` is returned.

    Parameters
    ----------
    parallel : numpy-array
        The fluorescence in the parallel detection channel
    perpendicular : numpy-array
        The fluorescence in the perpendicular detection channel
    g_factor : float
        A parameter accounting for the detection efficiency of the parallel
        and the perpendicular detection channel.

    Returns
    -------


    """
    scale_perpendicular = 2.0 * g_factor
    return parallel + perpendicular * scale_perpendicular


def fitrange(
        fluorescence_decay,
        threshold: float = 10.0,
        area: float = 0.999
) -> typing.Tuple[int, int]:
    """Determines a fitting range based on the total number of photons to be
    fitted (fitting area).

    Parameters
    ----------
    fluorescence_decay : numpy-array
        a numpy array containing the photon counts
    threshold : float
        a threshold value. Lowest index of the fitting range is the first
        encountered bin with a photon count higher than the threshold.
    area : float
        The area which should be considered for fitting. Here 1.0 corresponds
        to all measured photons. 0.9 to 90% of the total measured photons.
    threshold : float
         (Default value = 10.0)
    area: float
         (Default value = 0.999)

    Returns
    -------
        A tuple of integers (start, stop)
    """
    f = fluorescence_decay
    x_min = np.where(f > threshold)[0][0]
    cumsum = np.cumsum(f, dtype=np.float64)
    s = np.sum(f, dtype=np.float64)
    x_max = np.where(cumsum >= s * area)[0][0]
    if fluorescence_decay[x_max] < threshold:
        x_max = len(f) - np.searchsorted(f[::-1], [threshold])[-1]
    return x_min, min(x_max, len(f) - 1)
