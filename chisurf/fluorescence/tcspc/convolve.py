from __future__ import annotations

from math import exp, ceil
import numba as nb
import numpy as np
import tttrlib


@nb.jit(nopython=True, nogil=True)
def convolve_lifetime_spectrum_nb(
        output_decay: np.array,
        lifetime_spectrum: np.array,
        instrument_response_function: np.array,
        convolution_stop: int = -1,
        time_axis: np.array = None,
        amplitude_threshold: float = 0,
        use_amplitude_threshold: bool = False
) -> None:
    """Compute the fluorescence decay for a lifetime spectrum and am instrument
    response function.

    Fills the pre-allocated output array `output_decay` with a fluorescence
    intensity decay defined by a set of fluorescence lifetimes defined by the
    parameter `lifetime_spectrum`. The fluorescence decay will be convolved
    (non-periodically) with an instrumental response function that is defined
    by `instrument_response_function`.

    This function calculates a fluorescence intensity model_decay that is
    convolved with an instrument response function (IRF). The fluorescence
    intensity model_decay is specified by its fluorescence lifetime spectrum,
    i.e., an interleaved array containing fluorescence lifetimes with
    corresponding amplitudes.

    Parameters
    ----------
    output_decay : numpy.array
        Output array that is filled with the values of the computed
        fluorescence intensity decay model
    lifetime_spectrum : numpy.array
        Interleaved vector of amplitudes and fluorescence lifetimes
    instrument_response_function :
        the instrument response function
    convolution_stop :
        convolution stop channel (the index on the time-axis)
    time_axis :
        the time-axis of the model_decay
    amplitude_threshold : float
        Amplitudes in the fluorescence lifetime spectrum with an absolute value
        smaller than this number are not considered if `use_amplitude_threshold`
        is set to True
    use_amplitude_threshold : bool
        If this value is True (default False) fluorescence lifetimes in the
        lifetime spectrum which have an amplitude with an absolute value of
        that is smaller than `amplitude_threshold` are not considered.

    Examples
    --------
    >>> import scipy.stats
    >>> time_axis = np.linspace(0, 25, 50)
    >>> irf_position = 5.0
    >>> irf_width = 1.0
    >>> irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
    >>> lifetime_spectrum = np.array([0.8, 1.1, 0.2, 4.0])
    >>> model_decay = np.zeros_like(time_axis)
    >>> convolve_lifetime_spectrum(\
            model_decay, \
            lifetime_spectrum=lifetime_spectrum, \
            instrument_response_function=irf, \
            time_axis=time_axis\
        )
    """
    n_exp = lifetime_spectrum.shape[0] // 2
    if convolution_stop <= 0:
        convolution_stop = output_decay.shape[0]
    for i in range(convolution_stop):
        output_decay[i] = 0.0

    lt = lifetime_spectrum
    irf = instrument_response_function
    for ne in range(n_exp):
        a = lt[2 * ne]
        if (abs(a) < amplitude_threshold) and use_amplitude_threshold:
            continue
        current_lifetime = (lt[2 * ne + 1])
        if current_lifetime == 0.0:
            continue
        current_model_value = 0.0
        for i in range(1, convolution_stop):
            dt = (time_axis[i] - time_axis[i - 1])
            dt_2 = dt / 2.0
            current_exponential = exp(-dt / current_lifetime)
            current_model_value = (current_model_value + dt_2 * irf[i - 1]) * current_exponential + dt_2 * irf[i]
            output_decay[i] += current_model_value * a


@nb.jit(nopython=True, nogil=True)
def convolve_lifetime_spectrum_periodic_nb(
        decay: np.ndarray,
        lifetime_spectrum: np.ndarray,
        irf: np.ndarray,
        start: int,
        stop: int,
        n_points: int,
        period: float,
        dt: float,
        conv_stop: int
):
    """

    Parameters
    ----------
    decay : np.array
        array of doubles
        Here the convolved fit is stored
    lifetime_spectrum :
        array of doubles
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    irf : np.array
        array-doubles
        The instrument response function
    start : int
        Start channel of convolution (position in array of IRF)
    stop : int
        Stop channel of convolution (position in array of IRF)
    n_points : int
        Number of points in fit and lamp
    period : double
        Period of repetition in nano-seconds
    dt : double
        Channel-width in nano-seconds
    conv_stop : int
        Stopping channel of convolution

    Returns
    -------

    Examples
    --------
    >>> import scipy.stats
    >>> n_points = 2048
    >>> time_axis = np.linspace(0, 16, n_points)
    >>> irf_position = 5.0
    >>> irf_width = 0.5
    >>> dt = time_axis[1] - time_axis[0]
    >>> irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
    >>> lifetime_spectrum = np.array([0.8, 1.1, 0.2, 4.0])
    >>> model_decay = np.zeros_like(time_axis)
    >>> convolve_lifetime_spectrum_periodic(model_decay, lifetime_spectrum=lifetime_spectrum, irf=irf, start=0, stop=n_points, n_points=n_points, period=16, conv_stop=n_points, dt=dt)
    """
    stop = min(stop, n_points - 1)
    start = max(start, 0)

    n_exp = lifetime_spectrum.shape[0] // 2
    period_n = ceil(period / dt - 0.5)

    for i in range(start, stop):
        decay[i] = 0

    stop1 = min(n_points, period_n)
    dt_2 = dt * 0.5

    # convolution
    for ne in range(n_exp):
        x_curr = lifetime_spectrum[2 * ne]
        lt_curr = lifetime_spectrum[2 * ne + 1]
        tail_a = 1./(1.-exp(-period/lt_curr))
        exp_curr = exp(-dt/lt_curr)
        fit_curr = 0.
        decay[0] += dt_2 * irf[0] * (exp_curr + 1.) * x_curr
        for i in range(conv_stop):
            fit_curr = (fit_curr + dt_2 * irf[i - 1]) * exp_curr + dt_2 * irf[i]
            decay[i] += fit_curr * x_curr

        for i in range(conv_stop, stop1):
            fit_curr *= exp_curr
            decay[i] += fit_curr * x_curr

        fit_curr *= exp(-(period_n - stop1) * dt / lt_curr)
        for i in range(stop):
            fit_curr *= exp_curr
            decay[i] += fit_curr * x_curr * tail_a


def convolve_lifetime_spectrum_periodic(
        decay: np.ndarray,
        lifetime_spectrum: np.ndarray,
        irf: np.ndarray,
        start: int,
        stop: int,
        n_points: int,
        period: float,
        dt: float,
        conv_stop: int
):
    tttrlib.fconv_per_cs(
        decay,
        irf,
        lifetime_spectrum,
        period,
        conv_stop,
        stop,
        dt
    )

#
# @nb.jit(nopython=True, nogil=True)
# def fconv_per(
#         model_decay: np.array,
#         lifetime_spectrum: np.array,
#         irf: np.array,
#         start: int,
#         stop: int,
#         n_points: int,
#         period: float,
#         dt: float
# ):
#     dt_half = dt * 0.5
#     model_decay *= 0.0
#
#     x = lifetime_spectrum
#     n_exp = x.shape[0] // 2
#
#     period_n = int(ceil(period/dt-0.5))
#     stop1 = min(period_n, n_points - 1)
#
#     for ne in range(n_exp):
#         x_curr = lifetime_spectrum[2 * ne]
#         if x_curr == 0.0: continue
#         lt_curr = lifetime_spectrum[2 * ne + 1]
#         if lt_curr == 0.0: continue
#
#         exp_curr = exp(-dt / lt_curr)
#         tail_a = 1./(1.-exp(-period/lt_curr))
#
#         fit_curr = 0.0
#         for i in range(stop1):
#             fit_curr = (fit_curr + dt_half * irf[i-1]) * exp_curr + dt_half * irf[i]
#             model_decay[i] += fit_curr * x_curr
#         fit_curr *= exp(-(period_n - stop1 + start)*dt/lt_curr)
#
#         for i in range(start, stop):
#             fit_curr *= exp_curr
#             model_decay[i] += fit_curr * x_curr * tail_a
#     return 0
#
#
# @nb.jit(nopython=True, nogil=True)
# def fconv_per_dt(
#         model_decay: np.array,
#         lifetime_spectrum: np.array,
#         irf: np.array,
#         start: int,
#         stop: int,
#         n_points: int,
#         period: float,
#         time: np.array
# ):
#     # TODO: in future adaptive time-axis with increasing bin size
#     x = lifetime_spectrum
#     n_exp = x.shape[0]/2
#
#     #period_n = int(ceil(period/dt-0.5))
#
#
#     irf_start = 0
#     while irf[irf_start] == 0:
#         irf_start += 1
#
#     model_decay *= 0.0
#     #for i in range(stop):
#     #    model_decay[i] = 0.0
#
#     #stop1 = n_points - 1 if period_n + irf_start > n_points - 1 else period_n + irf_start
#     #if period_n + irf_start > n_points - 1:
#     #    stop1 = n_points - 1
#     #else:
#     #    stop1 = period_n + irf_start
#
#     for ne in range(n_exp):
#         x_curr = lifetime_spectrum[2 * ne]
#         if x_curr == 0.0:
#             continue
#         lt_curr = lifetime_spectrum[2 * ne + 1]
#         if lt_curr == 0.0:
#             continue
#
#         #tail_a = 1./(1.-exp(-period/lt_curr))
#
#         fit_curr = 0.0
#         for i in range(1, n_points):
#             dt = (time[i] - time[i-1])
#             dt_half = dt * 0.5
#             exp_curr = exp(-dt / lt_curr)
#             fit_curr = (fit_curr + dt_half * irf[i-1]) * exp_curr + dt_half * irf[i]
#             model_decay[i] += fit_curr * x_curr
#         #fit_curr *= exp(-(period_n - stop1 + start)*dt/lt_curr)
#
#         #for i in range(start, stop):
#         #    fit_curr *= exp_curr
#         #    model_decay[i] += fit_curr * x_curr * tail_a
#     return 0
#


@nb.jit(nopython=True, nogil=True)
def convolve_decay_nb(
        decay_curve: np.ndarray,
        irf: np.ndarray,
        start: int,
        stop: int,
        dt: float
) -> np.ndarray:
    """Convolves a fluorescence model_decay with an instrument response function.
    
    This function convolves a fluorescence model_decay, that is provided as a numpy array,
    with an experimental response function.

    Parameters
    ----------
    decay_curve : numpy-array
        fluorescence model_decay
    irf :
        instrument response function
    start : int
        convolution start
    stop : int
        convolution stop
    dt : float
        bin-width of the fluorescence model_decay
    irf : numpy-array
        np.array:

    Returns
    -------
    type
        convolved fluorescence model_decay as numpy array.
        
    Examples
    --------
    >>> import scipy.stats
    >>> n_points = 2048
    >>> time_axis = np.linspace(0, 16, n_points)
    >>> irf_position = 5.0
    >>> irf_width = 0.5
    >>> dt = time_axis[1] - time_axis[0]
    >>> irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
    >>> decay_u = np.exp(- time_axis / 4.1)
    >>> model_decay = convolve_decay(decay_u, irf=irf, start=0, stop=n_points, dt=dt)
    """
    decay_out = np.empty_like(decay_curve)
    for i in range(start, stop):
        decay_out[i] = 0.5 * irf[0] * decay_curve[i]
        for j in range(1, i):
            decay_out[i] += irf[j] * decay_curve[i - j]
        decay_out[i] += 0.5 * irf[i] * decay_curve[0]
        decay_out[i] *= dt
    decay_out[0] = 0
    return decay_out


def convolve_lifetime_spectrum(
        output_decay: np.array,
        lifetime_spectrum: np.array,
        instrument_response_function: np.array,
        convolution_stop: int = -1,
        time_axis: np.array = None,
        amplitude_threshold: float = 0,
        use_amplitude_threshold: bool = False
) -> None:
    dt = (time_axis[1] - time_axis[0])
    tttrlib.fconv(
        output_decay,
        instrument_response_function,
        lifetime_spectrum,
        0,
        convolution_stop,
        dt
    )
