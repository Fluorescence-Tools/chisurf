import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import mfm
from mfm import plots
from mfm.fitting.models.tcspc.fret import GaussianWidget, Gaussians
from mfm.fitting.models.tcspc.nusiance import GenericWidget, CorrectionsWidget
from mfm.fluorescence import tcspc
from mfm.fluorescence.general import distribution2rates
from mfm.fluorescence.widgets import AnisotropyWidget
from mfm.fitting import FittingParameterGroup, FittingParameter
from mfm.fitting.models.tcspc import Lifetime, LifetimeWidget, LifetimeModel, ConvolveWidget
from .. import ModelWidget


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
        FittingParameterGroup.__init__(self, **kwargs)

        self._fAB = FittingParameter(name='AtB', value=1.0, model=self.model, decimals=2, fixed=True)
        self._fBA = FittingParameter(name='BtA', value=0.0, model=self.model, decimals=2, fixed=True)

        self._pA = FittingParameter(value=0.0, name='pureA', model=self.model, decimals=2, fixed=True)
        self._pB = FittingParameter(value=0.0, name='pureB', model=self.model, decimals=2, fixed=True)

        self._pxA = FittingParameter(value=0.98, name='xA', model=self.model, decimals=2, fixed=True)
        self._pxB = FittingParameter(value=0.02, name='xB', model=self.model, decimals=2, fixed=True)

        self._pmA = FittingParameter(value=0.02, name='mA', model=self.model, decimals=2, fixed=True)
        self._pmB = FittingParameter(value=0.98, name='mB', model=self.model, decimals=2, fixed=True)


class PDDEMWidget(QtWidgets.QWidget, PDDEM):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        PDDEM.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/fitting/models/tcspc/pddem.ui', self)

        l = QtWidgets.QHBoxLayout()
        self._fAB = self._fAB.make_widget(layout=l, text='A>B')
        self._fBA = self._fBA.make_widget(layout=l, text='B>A')
        self.verticalLayout_3.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        self._pA = self._pA.make_widget(layout=l)
        self._pB = self._pB.make_widget(layout=l)
        self.verticalLayout_3.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        self._pxA = self._pxA.make_widget(layout=l, text='Ex<sub>A</sub>')
        self._pxB = self._pxB.make_widget(layout=l, text='Ex<sub>B</sub>')
        self.verticalLayout_3.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        self._pmA = self._pmA.make_widget(layout=l, text='Em<sub>A</sub>')
        self._pmB = self._pmB.make_widget(layout=l, text='Em<sub>B</sub>')
        self.verticalLayout_3.addLayout(l)



class PDDEMModel(LifetimeModel):
    """
    Kalinin, S., and Johansson, L.B.
    Energy Migration and Transfer Rates are Invariant to Modeling the
    Fluorescence Relaxation by Discrete and Continuous Distributions of
    Lifetimes.
    J. Phys. Chem. B, 108 (2004) 3092-3097.
    """

    name = "FRET: PDDEM"

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.pddem = PDDEM(name='pddem', **kwargs)
        self.gaussians = Gaussians(name='gaussians', **kwargs)
        self.fa = Lifetime(name='fa', **kwargs)
        self.fb = Lifetime(name='fb', **kwargs)

    @property
    def distance_distribution(self):
        dist = self.gaussians.distribution
        return dist

    @property
    def rate_spectrum(self):
        gaussians = self.gaussians
        rs = distribution2rates(gaussians.distribution, gaussians.tau0, gaussians.kappa2, gaussians.forster_radius)
        return np.hstack(rs).ravel([-1])

    @property
    def lifetime_spectrum(self):
        decayA = self.fa.lifetime_spectrum
        decayB = self.fb.lifetime_spectrum
        rate_spectrum = self.rate_spectrum

        p, rates = rate_spectrum[::2], rate_spectrum[1::2]
        decays = []
        for i, r in enumerate(rates):
            tmp = tcspc.pddem(decayA, decayB, self.pddem.fABBA * r,
                              self.pddem.px, self.pddem.pm, self.pddem.pureAB)
            tmp[0::2] *= p[i]
            decays.append(tmp)
        lt = np.concatenate(decays)
        if mfm.cs_settings['fret']['bin_lifetime']:
            n_lifetimes = mfm.cs_settings['fret']['lifetime_bins']
            discriminate_amplitude = mfm.cs_settings['fret']['discriminate_amplitude']
            return mfm.fluorescence.tcspc.bin_lifetime_spectrum(lt, n_lifetimes=n_lifetimes,
                                                                discriminate=False,
                                                                discriminator=discriminate_amplitude
                                                                )
        else:
            return lt

class PDDEMModelWidget(PDDEMModel, ModelWidget):

    plot_classes = [
                       (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                                         'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
                       (plots.FitInfo, {}), (plots.DistributionPlot, {}), (plots.ParameterScanPlot, {})
                    ]

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.ico"))
        PDDEMModel.__init__(self, fit=fit)

        self.convolve = ConvolveWidget(name='convolve', fit=fit, model=self, dt=fit.data.dt, hide_curve_convolution=True,
                                       **kwargs)

        self.corrections = CorrectionsWidget(fit=fit, model=self, **kwargs)
        self.generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        self.anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        self.pddem = PDDEMWidget(parent=self, model=self, short='P')
        self.gaussians = GaussianWidget(donors=None,  model=self.model, short='G', no_donly=True, name='gaussians')

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(self.convolve)
        self.layout.addWidget(self.generic)
        self.layout.addWidget(self.pddem)

        self.fa = LifetimeWidget(title='Lifetimes-A', model=self.model, short='A', name='fa')
        self.fb = LifetimeWidget(title='Lifetimes-B', model=self.model, short='B', name='fb')

        self.layout.addWidget(self.fa)
        self.layout.addWidget(self.fb)
        self.layout.addWidget(self.gaussians)
        self.layout.addWidget(self.anisotropy)
        self.layout.addWidget(self.corrections)

