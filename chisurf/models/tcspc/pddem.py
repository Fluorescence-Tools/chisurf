from __future__ import annotations

import numpy as np

import chisurf.settings
from chisurf.models.tcspc.fret import Gaussians
import chisurf.fluorescence.tcspc
from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.models.tcspc.lifetime import Lifetime
from chisurf.models.tcspc.fret import FRETModel


class PDDEM(FittingParameterGroup):

    name = "PDDEM"

    @property
    def pxA(self):
        """
        :return: float
            Excitation probability of fluorphore A
        """
        return self._pxA.value

    @property
    def pxB(self):
        """
        :return: float
            Excitation probability of fluorphore B
        """
        return self._pxB.value

    @property
    def px(self):
        """
        :return: numpy-array
            Exciation probabilities of flurophore (A, B)
        """
        return np.array([self.pxA, self.pxB], dtype=np.float64)

    @property
    def pmA(self):
        """
        :return: float
            Emission probability of flurophore A
        """
        return self._pmA.value

    @property
    def pmB(self):
        """
        :return: float
            Emission probability of flurophore B
        """
        return self._pmB.value

    @property
    def pm(self):
        """
        :return: array
            Emission probability of flurophore (A, B)
        """
        return np.array([self.pmA, self.pmB], dtype=np.float64)

    @property
    def alpha_A(self):
        return self._alpha_A.value

    @property
    def alpha_B(self):
        return self._alpha_B.value

    def update(self):
        denom = float(self._pmA.value) + float(self._pmB.value)
        if denom > 0.0 and np.isfinite(denom):
            alpha_a = float(self._pmA.value) / denom
            alpha_b = float(self._pmB.value) / denom
        else:
            alpha_a = float('nan')
            alpha_b = float('nan')
        try:
            self._alpha_A.value = alpha_a
        except Exception:
            pass
        try:
            self._alpha_B.value = alpha_b
        except Exception:
            pass

    @property
    def pureA(self):
        """
        :return: float
            fraction of decay A in total decay
        """
        return self._pA.value

    @property
    def pureB(self):
        """
        :return: float
            fraction of decay B in total decay
        """
        return self._pB.value

    @property
    def pureAB(self):
        """
        :return: array
            fraction of decay (A, B) in total decay
        """
        return np.array([self.pureA, self.pureB], dtype=np.float64)

    @property
    def fAB(self):
        """
        :return: float
            probability of energy-transfer from A to B
        """
        return self._fAB.value

    @property
    def fBA(self):
        """
        :return: float
            probability of energy-transfer from B to A
        """
        return self._fBA.value

    @property
    def fABBA(self):
        """
        :return: array
            probability of energy-transfer from (A to B), (B to A)
        """
        return np.array([self.fAB, self.fBA], dtype=np.float64)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._fAB = FittingParameter(name='AtB', value=1.0, model=self.model, decimals=2, fixed=True)
        self._fBA = FittingParameter(name='BtA', value=0.0, model=self.model, decimals=2, fixed=True)

        self._pA = FittingParameter(value=0.0, name='pureA', model=self.model, decimals=2, fixed=True)
        self._pB = FittingParameter(value=0.0, name='pureB', model=self.model, decimals=2, fixed=True)

        self._pxA = FittingParameter(value=0.98, name='xA', model=self.model, decimals=2, fixed=True)
        self._pxB = FittingParameter(value=0.02, name='xB', model=self.model, decimals=2, fixed=True)

        self._pmA = FittingParameter(value=0.02, name='mA', model=self.model, decimals=2, fixed=True)
        self._pmB = FittingParameter(value=0.98, name='mB', model=self.model, decimals=2, fixed=True)

        self._alpha_A = FittingParameter(
            value=float('nan'),
            name='alpha_A',
            model=self.model,
            fixed=True,
            is_output=True
        )
        self._alpha_B = FittingParameter(
            value=float('nan'),
            name='alpha_B',
            model=self.model,
            fixed=True,
            is_output=True
        )

        self.update()


class PDDEMModel(FRETModel):
    """
    Kalinin, S., and Johansson, L.B.
    Energy Migration and Transfer Rates are Invariant to Modeling the
    Fluorescence Relaxation by Discrete and Continuous Distributions of
    Lifetimes.
    J. Phys. Chem. B, 108 (2004) 3092-3097.
    """

    name = "FRET: PDDEM"

    def __init__(self, fit, **kwargs):
        self.fa = Lifetime(name='fa', **kwargs)
        self.fb = Lifetime(name='fb', **kwargs)
        self.donor = self.fb
        FRETModel.__init__(self, fit, **kwargs)
        self.pddem = PDDEM(name='pddem', **kwargs)
        self.gaussians = Gaussians(name='gaussians', **kwargs)

    @property
    def distance_distribution(self):
        dist = self.gaussians.distribution
        return dist

    @property
    def lifetime_spectrum(self):
        decayA = self.fa.lifetime_spectrum
        decayB = self.fb.lifetime_spectrum
        rate_spectrum = self.fret_rate_spectrum

        p, rates = rate_spectrum[::2], rate_spectrum[1::2]
        decays = []
        for i, r in enumerate(rates):
            tmp = chisurf.fluorescence.tcspc.pddem(
                decayA, decayB,
                self.pddem.fABBA * r,
                self.pddem.px,
                self.pddem.pm,
                self.pddem.pureAB
            )
            tmp[0::2] *= p[i]
            decays.append(tmp)
        lt = np.concatenate(decays)
        if chisurf.settings.cs_settings['fret']['bin_lifetime']:
            n_lifetimes = chisurf.settings.cs_settings['fret']['lifetime_bins']
            discriminate_amplitude = chisurf.settings.cs_settings['fret']['discriminate_amplitude']
            return chisurf.fluorescence.tcspc.bin_lifetime_spectrum(
                lt,
                n_lifetimes=n_lifetimes,
                discriminate=False,
                discriminator=discriminate_amplitude
            )
        else:
            return lt
