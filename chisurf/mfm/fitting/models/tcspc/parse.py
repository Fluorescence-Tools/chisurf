import os
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import ModelWidget
from mfm.fitting.widgets import FittingControllerWidget
from .. import parse
from mfm.fitting.models.tcspc import tcspc
import mfm.fitting.models.tcspc.nusiance


class ParseDecayModel(parse.ParseModel):

    def __init__(self, fit, **kwargs):
        parse.ParseModel.__init__(self, fit, **kwargs)
        self.convolve = kwargs.get('convolve', tcspc.Convolve(name='convolve', fit=fit, **kwargs))
        self.corrections = kwargs.get('corrections',
                                      mfm.fitting.models.tcspc.nusiance.Corrections(name='corrections', fit=fit, model=self, **kwargs))
        self.generic = kwargs.get('generic', mfm.fitting.models.tcspc.nusiance.Generic(name='generic', fit=fit, **kwargs))

    def update_model(self, **kwargs):
        #verbose = kwargs.get('verbose', self.verbose)
        #scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        #lintable = kwargs.get('lintable', self.corrections.lintable)

        parse.ParseModel.update_model(self, **kwargs)
        decay = self._y_values
        if self.convolve.irf is not None:
            decay = self.convolve.convolve(self._y_values, mode='full')[:self._y_values.shape[0]]

        self.convolve.scale(decay, self.fit.data, bg=background, start=self.fit.xmin, stop=self.fit.xmax)
        decay += self.generic.background
        decay[decay < 0.0] = 0.0
        if self.corrections.lintable is not None:
            decay *= self.corrections.lintable
        self._y_values = decay


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, icon=QtGui.QIcon(":/icons/icons/TCSPC.ico"))

        self.convolve = tcspc.ConvolveWidget(fit=fit, model=self, show_convolution_mode=False, dt=fit.data.dt, **kwargs)
        generic = mfm.fitting.models.tcspc.nusiance.GenericWidget(fit=fit, parent=self, model=self, **kwargs)
        #error_widget = mfm.fitting.error_estimate.ErrorWidget(fit, **kwargs)

        fn = os.path.join(mfm.package_directory, 'settings/tcspc.model.json')
        pw = parse.ParseFormulaWidget(self, model_file=fn)
        corrections = mfm.fitting.models.tcspc.nusiance.CorrectionsWidget(fit, model=self, **kwargs)

        self.fit = fit
        ParseDecayModel.__init__(self, fit=fit, parse=pw, convolve=self.convolve,
                                 generic=generic, corrections=corrections)
        fitting_widget = FittingControllerWidget(fit=fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(fitting_widget)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(pw)
        layout.addWidget(error_widget)
        layout.addWidget(corrections)
        self.setLayout(layout)

