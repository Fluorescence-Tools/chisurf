from __future__ import annotations

import numpy as np
import chisurf.fluorescence.fcs.correlate

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
    Calculate noise weights for fluorescence correlation spectroscopy (FCS)
    correlation curves.

    This function computes the noise weights (standard deviations) associated
    with each point in an FCS correlation curve using different weighting schemes.
    These weights can be used in subsequent fitting or analysis of the correlation
    data. The diffusion time, if not provided, is estimated as the time when the
    correlation curve falls to half of its (background-corrected) amplitude. To
    avoid early-time artefacts such as afterpulsing, the first few points can be
    skipped using the parameter `skip_points`.

    Weighting methods:

    - **suren**: Applies a noise model that accounts for shot noise and fluctuations
      in the correlation function. (Publication details are unclear for this method.)
    - **starchev**: Implements the noise model described by Starchev in:

          Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
          *Journal of Colloid and Interface Science, 233*, 50–55.

      The parameters `starchev_a1`, `starchev_a2`, and `starchev_c1` are empirical
      fitting parameters that depend on the correlator settings.
    - **uniform**: Assigns a uniform weight (i.e. all ones).

    Parameters
    ----------
    times : np.ndarray
        Array of correlation times in milliseconds.
    correlation : np.ndarray
        Array of correlation amplitudes.
    measurement_duration : float
        Total measurement duration in seconds.
    mean_count_rate : float
        Mean count rate in kHz.
    weight_type : str, optional
        Type of weighting to use. Options include 'suren', 'starchev', and 'uniform'.
        Default is 'starchev'.
    skip_points : int, optional
        Number of initial data points to skip (e.g., to avoid afterpulsing effects).
        Default is 0.
    correlation_amplitude_range : tuple of int, optional
        Tuple (lb, ub) defining the index range used to calculate the correlation offset
        and the mean correlation amplitude. Default is (0, 16).
    time_upper : float, optional
        Upper time limit (in ms) below which the standard weighting is applied.
        For times above this threshold, weights are scaled down. Default is 10.
    z0_w0 : float, optional
        Shape factor (z0/w0) for the 3D Gaussian detection volume. Default is 3.5.
    starchev_a1 : float, optional
        Empirical fitting parameter a1 for the Starchev method. Default is 2.e-3.
    starchev_a2 : float, optional
        Empirical fitting parameter a2 for the Starchev method. Default is 1.8e-1.
    starchev_c1 : float, optional
        Empirical fitting parameter c1 for the Starchev method. Default is 1.0e-4.
    diffusion_time : float, optional
        Characteristic diffusion time in milliseconds. If not provided, it is estimated
        as the time at which the (background-corrected) correlation curve falls to half
        its maximum value.
    verbose : bool, optional
        If True, prints diagnostic messages. Default is False.

    Returns
    -------
    np.ndarray
        Array of noise weights (standard deviations) corresponding to each point in the
        correlation curve.

    References
    ----------
    Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
    Journal of Colloid and Interface Science, 233, 50–55.

    Notes
    -----
    - The diffusion time is estimated by finding the first time point where the correlation
      falls below half of its mean amplitude (after background subtraction). If no such point
      is found, the middle of the time array is used.
    - The 'suren' weighting method is mentioned in the literature, but its publication details
      remain unclear.
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
    if mean_correlation_amplitude == 0:
        print("WARNING: the mean correlation amplitude seems to be zero!")
        mean_correlation_amplitude = 1.0
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
        S = (mean_correlation_amplitude ** 2 / ns *
             ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) +
             2 * mean_correlation_amplitude / ns ** 2.0 * (1 + B) +
             (1 + mean_correlation_amplitude * np.sqrt(B)) / (ns * na ** 2))
        S *= (times < time_upper) + (times >= time_upper) * 10 ** (-np.log(times + 1e-12) / np.log(10) + 1)
        S = np.sqrt(np.abs(S))
        sd[skip_points:] = S
    elif weight_type == 'starchev':
        # Noise on fluorescence correlation spectroscopy - Starchev method
        # Reference: Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
        # Journal of Colloid and Interface Science, 233, 50–55.
        tc = diffusion_time
        N = 1. / mean_correlation_amplitude
        a1 = starchev_a1
        a2 = starchev_a2
        c1 = starchev_c1
        p = z0_w0  # Shape factor of 3D Gaussian detection volume.
        i = mean_count_rate
        var_g = 1. / N ** 3.0 * (a1 / i + a2 / i ** 2) / (1 + p ** 2 * times / tc) + \
                c1 / (N ** 2.0) / (1 + p ** 2 * times / tc) ** 0.33
        sd[skip_points:] = np.sqrt(var_g)
    elif weight_type == 'uniform':
        sd = np.ones_like(correlation)

    return sd
