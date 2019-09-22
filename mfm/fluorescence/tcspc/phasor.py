from __future__ import annotations

import numpy as np


def phasor_giw(
        f,
        n,
        omega,
        times
):
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


def phasor_siw(
        f,
        n,
        omega,
        times
):
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


class Phasor(object):

    @property
    def phasor_siwD0(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_siw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwD0(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_giw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwDA(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_siw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwDA(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_giw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwE(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_siw(self.et, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwE(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return phasor_giw(self.et, self.phasor_n, self.phasor_omega, self.times)

    def set_fd0_fda_et(self, fd0, fda, et):
        self.fd0 = fd0
        self.fda = fda
        self.et = et

    def __init__(
            self,
            phasor_n: float = 1.0,
            phasor_omega: float = 31.25
    ):
        super(Phasor, self).__init__()
        self._phasor_n = phasor_n
        self._phasor_omega = phasor_omega

