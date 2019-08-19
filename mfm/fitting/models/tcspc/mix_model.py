import numpy as np
from PyQt5 import QtCore, QtWidgets

from mfm import plots
from mfm.fluorescence import stack_lifetime_spectra
from mfm.parameter import FittingParameterWidget
from mfm.widgets import clear_layout
from mfm.fitting.models.tcspc import LifetimeModel, LifetimeModelWidgetBase


class LifetimeMixModel(LifetimeModel):

    name = "Lifetime mix"

    @property
    def current_model_idx(self):
        return self._current_model_idx

    @current_model_idx.setter
    def current_model_idx(self, v):
        self._current_model_idx = v

    @property
    def fractions(self):
        x = np.abs([f.value for f in self._fractions])
        x /= sum(x)
        return x

    @fractions.setter
    def fractions(self, x):
        x = np.abs(x)
        x /= sum(x)
        for i, va in enumerate(x):
            self._fractions[i].value = va

    @property
    def lifetime_spectrum(self):
        if len(self) > 0:
            fractions = self.fractions
            lifetime_spectra = [m.lifetime_spectrum for m in self.models]
            return stack_lifetime_spectra(lifetime_spectra, fractions)
        else:
            return np.array([1.0, 1.0], dtype=np.float64)

    def clear_models(self):
        self._fractions = list()
        self.models = list()

    def append(self, model, fraction):
        self._fractions.append(fraction)
        model.find_parameters()
        self.models.append(model)

    def pop(self):
        return self.models.pop()

    @property
    def current_model(self):
        if len(self.models) > 0:
            return self.models[self.current_model_idx]
        else:
            return None

    @property
    def parameters_all(self):
        if self.current_model is not None:
            return self._parameters + self.current_model.parameters_all
        else:
            return self._parameters

    def update(self):
        self.find_parameters()
        for m in self.models:
            m.update()
        self.update_model()

    def finalize(self):
        LifetimeModel.finalize(self)
        for m in self.models:
            m.finalize()

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as initial:
            try:
                return object.__getattribute__(self.current_model, attr)
            except AttributeError:
                raise initial

    def __len__(self):
        return len(self.models)

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.__dict__.pop('lifetimes')
        self._current_model_idx = 0
        self.models = list()
        self._fractions = list()

    def __str__(self):
        s = "Mix-model\n"
        s += "========\n\n"
        for m, x in zip(self.models, self.fractions):
            s += "\nFraction: %.3f\n\n" % x
            s += str(m)
        return s


class LifetimeMixModelWidget(LifetimeModelWidgetBase, LifetimeMixModel):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
                                                  'x_label': 'x',
                                                  'y_label': 'y',
                                                  'plot_irf': True}
                     )
                    , (plots.FitInfo, {})
    ]

    @property
    def current_model_idx(self):
        return int(self._current_model.value())

    @current_model_idx.setter
    def current_model_idx(self, v):
        self._current_model.setValue(v)

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
    def selected_fit(self):
        i = self.model_selector.currentIndex()

        return self.model_types[i]

    def __init__(self, fit, **kwargs):
        LifetimeModelWidgetBase.__init__(self, fit, **kwargs)
        LifetimeMixModel.__init__(self, fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)

        l = QtWidgets.QHBoxLayout()

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems([m.name for m in self.model_types])
        l.addWidget(self.model_selector)

        self.add_button = QtWidgets.QPushButton('Add')
        l.addWidget(self.add_button)

        self.clear_button = QtWidgets.QPushButton('Clear')
        l.addWidget(self.clear_button)
        layout.addLayout(l)

        self.add_button.clicked.connect(self.add_model)
        self.clear_button.clicked.connect(self.clear_models)

        self.model_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.model_layout)

        self._current_model = QtWidgets.QSpinBox()
        self._current_model.setMinimum(0)
        self._current_model.setMaximum(0)
        self.layout_parameter.addWidget(self._current_model)
        self.layout_parameter.addLayout(layout)

    def add_model(self, fit=None):
        l = QtWidgets.QHBoxLayout()

        if fit is None:
            model = self.selected_fit.model
        else:
            model = fit.model

        fraction_name = "x(%s)" % (len(self) + 1)
        fraction = FittingParameterWidget(name=fraction_name, value=1.0, model=self, ub=1.0, lb=0.0, layout=l)
        l.addWidget(fraction)
        model_label = QtWidgets.QLabel(fit.name)
        l.addWidget(model_label)

        self.model_layout.addLayout(l)
        self.append(model, fraction)

    def clear_models(self):
        LifetimeMixModel.clear_models(self)
        clear_layout(self.model_layout)
