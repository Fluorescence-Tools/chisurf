import mfm
import mfm.fluorescence.phasor


class Phasor(object):

    @property
    def phasor_siwD0(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_siw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwD0(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_giw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwDA(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_siw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwDA(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_giw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwE(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_siw(self.et, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwE(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.phasor.phasor_giw(self.et, self.phasor_n, self.phasor_omega, self.times)

    def set_fd0_fda_et(self, fd0, fda, et):
        self.fd0 = fd0
        self.fda = fda
        self.et = et

    def __init__(self, **kwargs):
        self._phasor_n = kwargs.get('phasor_n', 1.0)
        self._phasor_omega = kwargs.get('phasor_omega', 31.25)