"""

"""
from __future__ import annotations

import numpy as np

weightCalculations = ['Koppel', 'none']
correlationMethods = ['tp']


def weights(
        times: np.ndarray,
        correlation: np.ndarray,
        measurement_duration: float,
        mean_count_rate: float,
        weight_type: str = 'suren'
) -> np.array:
    """
    :param times: correlation times [ms]
    :param correlation: correlation amplitude
    :param measurement_duration: measurement duration [s]
    :param mean_count_rate: count-rate [kHz]
    :param weight_type: weight type 'type' either 'suren' or 'uniform' for
    uniform weighting or Suren-weighting
    """
    if weight_type == 'suren':
        dt = np.diff(times)
        dt = np.hstack([dt, dt[-1]])
        ns = measurement_duration * 1000.0 / dt
        na = dt * mean_count_rate
        syn = (times < 10) + (times >= 10) * 10 ** (-np.log(times + 1e-12) / np.log(10) + 1)
        b = np.mean(correlation[1:5]) - 1

        # imaxhalf = len(g) - np.searchsorted(g[::-1], max(g[70:]) / 2, side='left')
        imaxhalf = np.min(np.nonzero(correlation < b / 2 + 1))
        tmaxhalf = times[imaxhalf]
        A = np.exp(-2 * dt / tmaxhalf)
        B = np.exp(-2 * times / tmaxhalf)
        m = times / dt
        S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) +
             2 * b / ns / na * (1 + B) + (
                     1 + b * np.sqrt(B)) / (ns * na * na)
             ) * syn
        S = np.abs(S)
        return 1. / np.sqrt(S)
    elif weight_type == 'uniform':
        return np.ones_like(correlation)
