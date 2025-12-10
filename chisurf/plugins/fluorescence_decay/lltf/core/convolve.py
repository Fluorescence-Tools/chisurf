"""
Convolution functions for lifetime fitting.

This module provides functions for convolving lifetime spectra with instrument
response functions (IRFs) to generate model decays.
"""

from __future__ import annotations

from math import exp, ceil
import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True)
def convolve_lifetime_spectrum_nb(
        output_decay: np.ndarray,
        lifetime_spectrum: np.ndarray,
        instrument_response_function: np.ndarray,
        convolution_stop: int = -1,
        time_axis: np.ndarray = None,
        amplitude_threshold: float = 0,
        use_amplitude_threshold: bool = False
) -> None:
    """Compute the fluorescence decay for a lifetime spectrum and an instrument
    response function.

    Fills the pre-allocated output array `output_decay` with a fluorescence
    intensity decay defined by a set of fluorescence lifetimes defined by the
    parameter `lifetime_spectrum`. The fluorescence decay will be convolved
    (non-periodically) with an instrumental response function that is defined
    by `instrument_response_function`.

    Parameters
    ----------
    output_decay : numpy.ndarray
        Output array that is filled with the values of the computed
        fluorescence intensity decay model
    lifetime_spectrum : numpy.ndarray
        Interleaved vector of amplitudes and fluorescence lifetimes
    instrument_response_function : numpy.ndarray
        The instrument response function
    convolution_stop : int
        Convolution stop channel (the index on the time-axis)
    time_axis : numpy.ndarray
        The time-axis of the model decay
    amplitude_threshold : float
        Amplitudes in the fluorescence lifetime spectrum with an absolute value
        smaller than this number are not considered if `use_amplitude_threshold`
        is set to True
    use_amplitude_threshold : bool
        If this value is True (default False) fluorescence lifetimes in the
        lifetime spectrum which have an amplitude with an absolute value of
        that is smaller than `amplitude_threshold` are not considered.
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
    """Compute the fluorescence decay for a lifetime spectrum and an instrument
    response function with periodic excitation.

    Parameters
    ----------
    decay : numpy.ndarray
        Array where the convolved fit is stored
    lifetime_spectrum : numpy.ndarray
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    irf : numpy.ndarray
        The instrument response function
    start : int
        Start channel of convolution (position in array of IRF)
    stop : int
        Stop channel of convolution (position in array of IRF)
    n_points : int
        Number of points in fit and lamp
    period : float
        Period of repetition in nano-seconds
    dt : float
        Channel-width in nano-seconds
    conv_stop : int
        Stopping channel of convolution
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
    """Wrapper for convolve_lifetime_spectrum_periodic_nb.

    Parameters
    ----------
    decay : numpy.ndarray
        Array where the convolved fit is stored
    lifetime_spectrum : numpy.ndarray
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    irf : numpy.ndarray
        The instrument response function
    start : int
        Start channel of convolution (position in array of IRF)
    stop : int
        Stop channel of convolution (position in array of IRF)
    n_points : int
        Number of points in fit and lamp
    period : float
        Period of repetition in nano-seconds
    dt : float
        Channel-width in nano-seconds
    conv_stop : int
        Stopping channel of convolution
    """
    convolve_lifetime_spectrum_periodic_nb(
        decay,
        lifetime_spectrum,
        irf,
        start,
        stop,
        n_points,
        period,
        dt,
        conv_stop
    )


@nb.jit(nopython=True, nogil=True)
def convolve_decay_nb(
        decay_curve: np.ndarray,
        irf: np.ndarray,
        start: int,
        stop: int,
        dt: float
) -> np.ndarray:
    """Convolves a fluorescence decay with an instrument response function.
    
    This function convolves a fluorescence decay, that is provided as a numpy array,
    with an experimental response function.

    Parameters
    ----------
    decay_curve : numpy.ndarray
        Fluorescence decay
    irf : numpy.ndarray
        Instrument response function
    start : int
        Convolution start
    stop : int
        Convolution stop
    dt : float
        Bin-width of the fluorescence decay

    Returns
    -------
    numpy.ndarray
        Convolved fluorescence decay
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
        output_decay: np.ndarray,
        lifetime_spectrum: np.ndarray,
        instrument_response_function: np.ndarray,
        convolution_stop: int = -1,
        time_axis: np.ndarray = None,
        amplitude_threshold: float = 0,
        use_amplitude_threshold: bool = False
) -> None:
    """Wrapper for convolve_lifetime_spectrum_nb.

    Parameters
    ----------
    output_decay : numpy.ndarray
        Output array that is filled with the values of the computed
        fluorescence intensity decay model
    lifetime_spectrum : numpy.ndarray
        Interleaved vector of amplitudes and fluorescence lifetimes
    instrument_response_function : numpy.ndarray
        The instrument response function
    convolution_stop : int
        Convolution stop channel (the index on the time-axis)
    time_axis : numpy.ndarray
        The time-axis of the model decay
    amplitude_threshold : float
        Amplitudes in the fluorescence lifetime spectrum with an absolute value
        smaller than this number are not considered if `use_amplitude_threshold`
        is set to True
    use_amplitude_threshold : bool
        If this value is True (default False) fluorescence lifetimes in the
        lifetime spectrum which have an amplitude with an absolute value of
        that is smaller than `amplitude_threshold` are not considered.
    """
    convolve_lifetime_spectrum_nb(
        output_decay,
        lifetime_spectrum,
        instrument_response_function,
        convolution_stop,
        time_axis,
        amplitude_threshold,
        use_amplitude_threshold
    )


@nb.jit(nopython=True, nogil=True)
def add_pile_up_to_model(
        data: np.ndarray,
        model: np.ndarray,
        rep_rate: float,
        dead_time: float,
        measurement_time: float,
        modify_inplace: bool = True
) -> np.ndarray:
    """Add pile up effect to model function.

    Notes
    -----
    This function scales the model function. Thus, the scaling needs to
    be adjusted after adding pile-up to the model. The function uses the
    assumptions as described in ref [1]_.

    Parameters
    ----------
    data : numpy-array
        The array containing the experimental decay
    model : numpy-array
        The array containing the model function
    rep_rate : float
        The repetition-rate in MHz
    dead_time : float
        The dead-time of the system in nanoseconds
    measurement_time : float
        The measurement time in seconds
    modify_inplace : bool
        If set to True (default) pile-up is added to the input model array and
        the input is modified inplace. If False a copy of the input model array
        is created and pile-up is added to the copy of the model.

    Returns
    -------
    numpy-array
        An array containing a model function with added pile-up.

    References
    ----------

    .. [1]  Coates, P.B., The correction for photon pile-up in the measurement
            of radiative lifetimes 1968 J. Phys. E: Sci. Instrum. 1 878

    .. [2]  Walker, J.G., Iterative correction for pile-up in single-photon
            lifetime measurement 2002 Optics Comm. 201 271-277

    """
    rep_rate *= 1e6
    dead_time *= 1e-9
    cum_sum = np.cumsum(data)
    n_pulse_detected = cum_sum[-1]
    total_dead_time = n_pulse_detected * dead_time
    live_time = measurement_time - total_dead_time
    n_excitation_pulses = max(live_time * rep_rate, n_pulse_detected)

    # Coates, 1968, eq. 2
    p = data / (n_excitation_pulses - np.cumsum(data))
    # Coates, 1968, eq. 4
    rescaled_data = -np.log(1.0 - p)
    rescaled_data[rescaled_data == 0] = 1.0

    # instead of rescaling the data, the model function is
    # rescaled, to preserve the counting statistics and the
    # known noise.
    sf = data / rescaled_data
    sf = sf / np.sum(sf) * len(data)
    if modify_inplace:
        model *= sf
        return model
    else:
        a = np.empty(model.shape, dtype=np.float64)
        for i, mv in enumerate(model):
            a[i] = mv * sf[i]
        return a