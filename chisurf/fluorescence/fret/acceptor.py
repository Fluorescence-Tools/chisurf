"""

"""
from __future__ import annotations

import numpy as np
import chisurf.fluorescence.tcspc.convolve


def da_a0_to_ad(
        times: np.array,
        decay_da: np.array,
        acceptor_lifetime_spectrum: np.array,
        transfer_efficiency: float = 0.5
) -> np.array:
    """Convolves the donor decay in presence of FRET with the acceptor decay in the absence of FRET to yield the
     FRET-sensitized decay of the acceptor

     This function convolves the fluorescence decay of the donor in the presence of FRET with the decay of
     the acceptor in the absence of FRET. The resulting decay is scaled by the provided transfer efficiency.

    :param times: linear spaced time axis
    :param decay_da: fluorescence lifetime decay of the donor in the presence of FRET
    :param acceptor_lifetime_spectrum: interleaved (amplitude, lifetime) lifetime spectrum of the acceptor in
    the absence of FRET
    :param transfer_efficiency:
    :return: fluorescence decay of the FRET sensitized acceptor

    Examples
    --------

    >>> times = np.linspace(0, 50, 1024)
    >>> tau_da = 3.5
    >>> decay_da = np.exp(-times / tau_da)
    >>> acceptor_lifetime_spectrum = np.array([0.1, 0.5, 0.9, 2.0])
    >>> transfer_efficiency = 0.3
    >>> decay_ad = da_a0_to_ad(times=times, decay_da=decay_da, acceptor_lifetime_spectrum=acceptor_lifetime_spectrum, transfer_efficiency=transfer_efficiency)

    """
    a0 = np.zeros_like(decay_da)
    for i in range(len(acceptor_lifetime_spectrum) // 2):
        a = acceptor_lifetime_spectrum[i]
        tau = acceptor_lifetime_spectrum[i + 1]
        a0 += a * np.exp(-times / tau)

    dt = times[1] - times[0]
    decay_ad = chisurf.fluorescence.tcspc.convolve.convolve_decay(
        decay=a0,
        irf=decay_da,
        start=0,
        stop=decay_da.shape[0],
        dt=dt
    )
    ad = scale_acceptor(
        donor=decay_da,
        acceptor=decay_ad,
        transfer_efficiency=transfer_efficiency
    )
    return ad


def scale_acceptor(
        donor: np.ndarray,
        acceptor: np.ndarray,
        transfer_efficiency: float
) -> np.ndarray:
    """Computes a scaled the fluorescence decay of the acceptor that corresponds to the provided fluorescence decay of
    the donor and the transfer efficiency

    :param donor: fluorescence decay of the donor in the presence of FRET
    :param acceptor: fluorescence decay of the acceptor in the presence of FRET
    :param transfer_efficiency: transfer efficiency
    :return:  scaled fluorescence decay of the acceptor in the presence of FRET

    Examples
    --------

    >>> times = np.linspace(0, 50, 1024)
    >>> tau_da = 3.5
    >>> decay_da = np.exp(-times / tau_da)
    >>> acceptor_lifetime_spectrum = np.array([0.1, 0.5, 0.9, 2.0])
    >>> transfer_efficiency = 0.3
    >>> decay_ad = da_a0_to_ad(times=times, decay_da=decay_da, acceptor_lifetime_spectrum=acceptor_lifetime_spectrum, transfer_efficiency=0.5)
    >>> target_value = 0.2
    >>> scaled_acceptor = scale_acceptor(donor=decay_da, acceptor=decay_ad, transfer_efficiency=target_value)
    >>> sum(scaled_acceptor) / (sum(scaled_acceptor) +  sum(decay_da))
    0.1999999999999999
    """
    s_d = np.sum(donor)
    s_a = np.sum(acceptor)
    scaling_factor = s_d / s_a * (1.0 / transfer_efficiency - 1.0)**(-1.0)
    return acceptor * scaling_factor
