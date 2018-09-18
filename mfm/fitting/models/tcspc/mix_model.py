import numpy as np
from PyQt4 import QtCore, QtGui

from mfm import plots
from mfm.fitting.fit import FittingControllerWidget
from mfm.fitting.models.tcspc.fret import FRETrateModelWidget, WormLikeChainModelWidget, FRETModel, SingleDistanceModelWidget
from mfm.fitting.models.tcspc.nusiance import GenericWidget, CorrectionsWidget
from mfm.fluorescence import stack_lifetime_spectra
from mfm.fluorescence.tcspc.convolve import ConvolveWidget
from mfm.parameter import FittingParameterWidget
from mfm.widgets import clear_layout
from .fret import GaussianModelWidget
from .tcspc import LifetimeModel
from .tcspc import LifetimeModelWidget


class MixModel(FRETModel):
    """
    Mix model
    """

    name = "Model mix"

    @property
    def fractions(self):
        """
        The fraction of the different models
        """
        x = np.array([f.value for f in self._fractions])
        x /= sum(x)
        return x

    @fractions.setter
    def fractions(self, x):
        x = np.array([f.value for f in self._fractions])
        x /= sum(x)
        for i, va in enumerate(x):
            self._fractions[i].value = va

    @property
    def lifetime_spectrum(self):
        if self.global_donor_enabled:
            for m in self.models:
                m.donors = self.donors
        if len(self) > 0:
            fractions = self.fractions
            lifetime_spectra = [m.lifetime_spectrum for m in self.models]
            return stack_lifetime_spectra(lifetime_spectra, fractions)
        else:
            return np.array([1.0, 1.0], dtype=np.float64)

    @property
    def global_donor_enabled(self):
        """
        If this attribute is True the donor lifetime distribution of the mixed decays are
        overwritten by the MixModel global donor-decay
        :return:
        """
        return self._enable_mix_model_donor

    def clear_models(self):
        self._fractions = list()
        self.models = list()

    def update(self):
        for m in self.models:
            m.update()
        self.update_widgets()
        self.update_plots()

    def append(self, model, fraction):
        self._fractions.append(fraction)
        self.models.append(model)

    def pop(self):
        return self.models.pop()

    def __len__(self):
        return len(self.models)

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.models = list()
        self._fractions = list()

    def __str__(self):
        s  = "Mix-model\n"
        s += "---------\n\n"
        for m, x in zip(self.models, self.fractions):
            s += "\nFraction: %.3f\n\n" % x
            s += str(m)
        return s



class MixModelWidget(MixModel, QtGui.QWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
                                                  }
                    ),
                        (plots.SurfacePlot, {})
    ]

    model_types = [GaussianModelWidget, WormLikeChainModelWidget,
                   LifetimeModelWidget,
                   SingleDistanceModelWidget, FRETrateModelWidget
                   ]

    @property
    def amplitude(self):
        layout = self.model_layout
        re = list()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item, FittingParameterWidget):
                re.append(item)
        return re

    @property
    def current_model_type(self):
        i = self.model_selector.currentIndex()
        return self.model_types[i]

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        layout = QtGui.QVBoxLayout(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")

        self.fit = fit
        self.kwargs = kwargs

        fitting = FittingControllerWidget(fit=fit, **kwargs)
        if kwargs.get('hide_fit', False):
            fitting.hide()

        donors = LifetimeModelWidget(fit=fit, **kwargs)
        self._enable_mix_model_donor = kwargs.get('enable_mix_model_donor', False)
        if not self._enable_mix_model_donor:
            donors.hide()

        hide_convolve = self.kwargs.get('hide_convolve', False)
        convolve = ConvolveWidget(fit=fit, model=self, hide_curve_convolution=hide_convolve, **kwargs)
        generic = GenericWidget(fit=fit, model=self, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)

        MixModel.__init__(self, fit, generic=generic, convolve=convolve, donors=donors, corrections=corrections,
                          **kwargs)

        self.y_values = np.zeros(fit.data.y.shape[0])
        self.dt = kwargs.get('dt', fit.data.x[1] - fit.data.x[0])

        layout.setSpacing(0)
        layout.setMargin(0)
        layout.setAlignment(QtCore.Qt.AlignTop)

        layout.addWidget(fitting)
        layout.addWidget(convolve)
        layout.addWidget(generic)
        layout.addWidget(donors)
        layout.addWidget(corrections)

        self.model_selector = QtGui.QComboBox()
        self.model_selector.addItems([m.name for m in self.model_types])
        self.add_button = QtGui.QPushButton('Add')
        self.clear_button = QtGui.QPushButton('Clear')

        l = QtGui.QHBoxLayout()
        l.addWidget(self.model_selector)
        l.addWidget(self.add_button)
        l.addWidget(self.clear_button)
        layout.addLayout(l)

        self.connect(self.add_button, QtCore.SIGNAL('clicked()'), self.add_model)
        self.connect(self.clear_button, QtCore.SIGNAL('clicked()'), self.clear_models)

        self.model_layout = QtGui.QVBoxLayout()
        layout.addLayout(self.model_layout)

    def add_model(self):
        gb = QtGui.QGroupBox()
        l = QtGui.QVBoxLayout()
        gb.setLayout(l)

        self.kwargs['hide_convolve'] = True
        self.kwargs['hide_corrections'] = True
        self.kwargs['hide_generic'] = True
        self.kwargs['hide_fit'] = True
        self.kwargs['disable_fit'] = True
        self.kwargs['hide_error'] = True

        model = self.current_model_type(fit=self.fit, **self.kwargs)
        name = "x(%s)" % (len(self) + 1)
        fraction = FittingParameterWidget(name=name, value=1.0, model=self, ub=1.0, lb=0.0, layout=l)
        l.addWidget(fraction)
        l.addWidget(model)

        self.model_layout.addWidget(gb)
        self.append(model, fraction)

    def clear_models(self):
        MixModel.clear_models(self)
        clear_layout(self.model_layout)