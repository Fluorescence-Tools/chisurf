import numpy as np


def phasor_giw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.cos(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)


def phasor_siw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.sin(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)