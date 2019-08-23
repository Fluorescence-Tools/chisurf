from __future__ import annotations

import numba as nb
import numpy as np

import mfm
import mfm.math


def compute_linearization_table(
        y: np.array,
        window_length: int,
        window_function_type: str,
        x_min: int,
        x_max: int
):
    """
    This function calculates a liberalization table for differential non-linearities given a measurement of
    uncorrelated light. The linearization table is smoothed to reduce the noise contained in the liberalization
    measurements.

    :param y:
    :param window_length:
    :param window_function_type:
    :param x_min:
    :param x_max:
    :return:
    """
    x2 = y.copy()
    x2 /= x2[x_min:x_max].mean()
    mnx = np.ma.array(x2)
    mask2 = np.array([i < x_min or i > x_max for i in range(len(x2))])
    mnx.mask = mask2
    mnx.fill_value = 1.0
    mnx /= mnx.mean()
    yn = mnx.filled()
    return mfm.math.signal.window(
        yn,
        window_length,
        window_function_type
    )


@nb.jit(nopython=True, nogil=True)
def correct_model_for_pile_up(
        data,
        model,
        rep_rate,
        dead_time,
        measurement_time,
        verbose: bool = False
) -> None:
    """
    Add pile up effect to model function.
    Attention: This changes the scaling of the model function.

    :param rep_rate: float
        The repetition-rate in MHz
    :param dead_time: float
        The dead-time of the system in nanoseconds
    :param measurement_time: float
        The measurement time in seconds
    :param data: numpy-array
        The array containing the experimental decay
    :param model: numpy-array
        The array containing the model function
    :param verbose:

    References
    ----------

    .. [1]  Coates, P.B., A fluorimetric attachment for an
            atomic-absorption spectrophotometer
            1968 J. Phys. E: Sci. Instrum. 1 878

    .. [2]  Walker, J.G., Iterative correction for pile-up in
            single-photon lifetime measurement
            2002 Optics Comm. 201 271-277

    """
    rep_rate *= 1e6
    dead_time *= 1e-9
    cum_sum = np.cumsum(data)
    n_pulse_detected = cum_sum[-1]
    total_dead_time = n_pulse_detected * dead_time
    live_time = measurement_time - total_dead_time
    n_excitation_pulses = max(live_time * rep_rate, n_pulse_detected)
    if verbose:
        print("------------------")
        print("rep. rate [Hz]: %s" % rep_rate)
        print("live time [s]: %s" % live_time)
        print("dead time per pulse [s]: %s" % dead_time)
        print("n_pulse: %s" % n_excitation_pulses)
        print("dead [s]: %s" % total_dead_time)

    c = n_excitation_pulses - np.cumsum(data)
    rescaled_data = -np.log(1.0 - data / c)
    rescaled_data[rescaled_data == 0] = 1.0
    sf = data / rescaled_data
    sf = sf / np.sum(sf) * len(data)
    model *= sf
