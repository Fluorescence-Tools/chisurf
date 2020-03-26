"""

"""
from __future__ import annotations

import numpy as np

import chisurf.fluorescence.fcs.correlate
import chisurf.fluorescence.fcs.filtered
from chisurf import typing

weightCalculations = ['Koppel', 'none']
correlationMethods = ['tp']


def noise(
        times: np.ndarray,
        correlation: np.ndarray,
        measurement_duration: float,
        mean_count_rate: float,
        weight_type: str = 'starchev',
        skip_points: int = 0,
        correlation_amplitude_range: typing.Tuple[int, int] = (0, 16),
        time_upper: float = 10,
        z0_w0: float = 3.5,
        starchev_a1: float = 2.e-3,
        starchev_a2: float = 1.8e-1,
        starchev_c1: float = 1.0e-4,
        diffusion_time: float = None,
        verbose: bool = False
) -> np.array:
    """
    diffusion_time if not provided is estimated by the time where half the
    height of the correlation curve. As there may be afterpulsing the
    first correlation channels can be skipped using the parameter skip_points

    Method suren:
    ?? unclear where this is published

    Method starchev
    Journal of Colloid and Interface Science233,50–55 (2001)
    The parameters starchev_a1, starchev_a2, and starchev_c1 are
    empirical fitting parameters that should be optimized and depend on
    the correlator settings
    The constantsa1,a2, andc1depend on the equipment, such asthe correlator setting, as well asøc.

    :param times: correlation times [ms]
    :param correlation: correlation amplitude
    :param measurement_duration: measurement duration [s]
    :param mean_count_rate: count-rate [kHz]
    :param skip_points: skips the first points, e.g., because of after-pulsing
    :param weight_type: weight type 'type' either 'suren' or 'uniform' for
    uniform weighting or Suren-weighting
    """
    if verbose:
        print("Calculating FCS weights")
        print("Using method: %s" % weight_type)
        print("measurement_duration [s]: %s" % measurement_duration)
        print("mean_count_rate [kHz]: %s" % mean_count_rate)
        print("skip_points: %s" % skip_points)

    sd = np.ones_like(correlation)
    if skip_points > 0:
        times = times[skip_points:]
        correlation = correlation[skip_points:]

    lb, ub = correlation_amplitude_range

    correlation_offset = np.mean(correlation[-lb:-ub])
    mean_correlation_amplitude = np.mean(correlation[lb:ub]) - correlation_offset

    if diffusion_time is None:
        try:
            imaxhalf = np.min(np.nonzero(correlation < mean_correlation_amplitude / 2.0 + correlation_offset))
            diffusion_time = times[imaxhalf]
        except ValueError:
            diffusion_time = times[len(times) // 2]

    if weight_type == 'suren':
        dt = np.diff(times)
        dt = np.hstack([dt, dt[-1]])
        ns = measurement_duration * 1000. / dt
        na = dt * mean_count_rate

        A = np.exp(-2 * dt / diffusion_time)
        B = np.exp(-2 * times / diffusion_time)
        m = times / dt
        S = (mean_correlation_amplitude**2 / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) + 2 * mean_correlation_amplitude/ns**2.0 * (1 + B) + (1 + mean_correlation_amplitude * np.sqrt(B)) / (ns * na**2))
        S *= (times < time_upper) + (times >= time_upper) * 10 ** (-np.log(times + 1e-12) / np.log(10) + 1)
        S = np.sqrt(np.abs(S))
        sd[skip_points:] = S
    elif weight_type == 'starchev':
        # Noise on fluorescence correlation spectroscopy - J. colloid and interf. science, 2001
        # Journal of Colloid and Interface Science233,50–55 (2001)
        tc = diffusion_time
        N = 1. / mean_correlation_amplitude
        a1 = starchev_a1
        a2 = starchev_a2
        c1 = starchev_c1
        p = z0_w0  # shape factor of 3D gauss
        i = mean_count_rate
        var_g = 1./N**3.0 * (a1/i+a2/i**2)/(1+p**2*times/tc) + c1/(N**2.0) / (1+p**2.0 * times/tc)**0.33
        sd[skip_points:] = np.sqrt(var_g)
    elif weight_type == 'uniform':
        sd = np.ones_like(correlation)

    return sd

