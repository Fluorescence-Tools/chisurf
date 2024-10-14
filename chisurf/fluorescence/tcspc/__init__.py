from __future__ import annotations

from . tcspc import *

from chisurf import typing
import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.tcspc.corrections

from .tcspc import rescale_w_bg


def counting_noise(
        decay: np.ndarray,
        treat_zeros: bool = True,
        zero_value: float = 1.0
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


def combined_counting_noise_parallel_perpendicular(
        parallel: np.ndarray,
        perpendicular: np.ndarray,
        g_factor: float,
        treat_zeros: bool = True,
        zero_value: float = 1.0
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
    zero_value : float
    treat_zeros : True

    Returns
    -------
    numpy-array
        The weights of the channels of a decay combined.

    """
    scale_perpendicular = 2.0 * g_factor
    vp = counting_noise(
        decay=parallel,
        treat_zeros=treat_zeros,
        zero_value=zero_value
    )
    vs = counting_noise(
        decay=perpendicular,
        treat_zeros=treat_zeros,
        zero_value=zero_value
    )
    vt = np.sqrt(vp ** 2 + scale_perpendicular ** 2 * vs ** 2)
    if treat_zeros:
        vt = np.maximum(vt, zero_value)
    return vt

counting_noise_combined_parallel_perpendicular = combined_counting_noise_parallel_perpendicular


def get_analysis_range(
        fluorescence_decay: np.ndarray,
        count_threshold: float = 10.0,
        area: float = 0.95,
        start_at_peak: bool = True,
        start_fraction: float = 0.8,
        skip_first_channels: int = 0,
        skip_last_channels: int = 0,
        verbose: bool = False
) -> typing.Tuple[int, int]:
    """Determines a fitting range based on the total number of photons to be
    fitted (fitting area).

    Parameters
    ----------
    fluorescence_decay : numpy-array
        a numpy array containing the photon counts
    count_threshold : float
        a threshold value. Lowest index of the fitting range is the first
        encountered bin with a photon count higher than the threshold.
    area : float
        The area which should be considered for fitting. Here 1.0 corresponds
        to all measured photons. 0.9 to 90% of the total measured photons.
    count_threshold : float
         The minimum number of photons in a channel (default value = 10.0)
    area: float
         (Default value = 0.999)
    start_at_peak: bool
        If set to True (default is False) sets the initial value of the fit range
        to min(start_fraction * max(fluorescence_decay))
    start_fraction: float
        Is the fraction of the maximum of the fluorescence decay at which the
        fit range starts (if start_at_peak is set to True)
    verbose: bool
        If set to True (default is False) prints the analysis range to the
        standard output.

    Returns
    -------
        A tuple of integers (start, stop)
    """
    if start_at_peak:
        x_min = np.where(fluorescence_decay > start_fraction * np.max(fluorescence_decay))[0][0]
    else:
        x_min = np.where(fluorescence_decay > count_threshold)[0][0]
    f_min = fluorescence_decay[x_min:]
    cumsum = np.cumsum(f_min, dtype=np.float64)
    s = np.sum(f_min, dtype=np.float64)
    n_decay = len(fluorescence_decay)
    x_max = np.where(cumsum >= s * area)[0][0] + x_min
    if fluorescence_decay[x_max] < count_threshold:
        x_max = np.searchsorted(fluorescence_decay[::-1], [count_threshold])[-1]
    x_max = min(n_decay, x_max)
    while True:
        if x_max >= n_decay:
            break
        if fluorescence_decay[x_max] < count_threshold:
            break
        else:
            x_max += 1
    a_min = x_min + skip_first_channels
    a_max = min(x_max, len(fluorescence_decay) - skip_last_channels - 1)

    if verbose:
        print("GET_ANALYSIS_RANGE")
        print("-- count_threshold:", count_threshold)
        print("-- area:", area)
        print("-- start_at_peak:", start_at_peak)
        print("-- start_fraction:", start_fraction)
        print("-- analysis range: ", a_min, a_max)
    return a_min, a_max

initial_fit_range = get_analysis_range

# def initial_fit_range(
#         fluorescence_decay,
#         threshold: float = 10.0,
#         area: float = 0.999
# ) -> typing.Tuple[int, int]:
#     """Determines a fitting range based on the total number of photons to be
#     fitted (fitting area).
#
#     Parameters
#     ----------
#     fluorescence_decay : numpy-array
#         a numpy array containing the photon counts
#     threshold : float
#         a threshold value. Lowest index of the fitting range is the first
#         encountered bin with a photon count higher than the threshold.
#     area : float
#         The area which should be considered for fitting. Here 1.0 corresponds
#         to all measured photons. 0.9 to 90% of the total measured photons.
#     threshold : float
#          (Default value = 10.0)
#     area: float
#          (Default value = 0.999)
#
#     Returns
#     -------
#         A tuple of integers (start, stop)
#     """
#     f = fluorescence_decay
#     x_min = np.where(f > threshold)[0][0]
#     cumsum = np.cumsum(f, dtype=np.float64)
#     s = np.sum(f, dtype=np.float64)
#     x_max = np.where(cumsum >= s * area)[0][0]
#     if fluorescence_decay[x_max] < threshold:
#         x_max = len(f) - np.searchsorted(f[::-1], [threshold])[-1]
#     return x_min, min(x_max, len(f) - 1)
