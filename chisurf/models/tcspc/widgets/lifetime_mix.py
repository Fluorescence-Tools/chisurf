from __future__ import annotations

import chisurf
from chisurf import typing
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.general
import chisurf.plots
import chisurf.math.datatools
import chisurf.fitting.fit

from chisurf.models.tcspc.mix_model import LifetimeMixModel
from chisurf.models.tcspc.widgets.lifetime import LifetimeModelWidgetBase


class LifetimeMixModelWidget(LifetimeModelWidgetBase, LifetimeMixModel):

    plot_classes = [
        (
            chisurf.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {}),
        (
            chisurf.plots.DistributionPlot,
            {
                'distribution_options': {
                    'Lifetime': {
                        'attribute': 'lifetime_spectrum',
                        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
                        'accessor_kwargs': {'sort': True},
                        'curve_options': {
                            'stepMode': False,
                            'connect': False,
                            'symbol': "o"
                        }
                    }
                }
            }
        )
    ]

    @property
    def current_model_idx(self) -> int:
        return int(self._current_model.value())

    @current_model_idx.setter
    def current_model_idx(self, v: int):
        self._current_model.setValue(v)

    @property
    def amplitude(self) -> typing.List[float]:
        layout = self.model_layout
        re = list()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(
                    item,
                    chisurf.gui.widgets.fitting.widgets.FittingParameterWidget
            ):
                re.append(item)
        return re

    @property
    def selected_fit(self):
        i = self.model_selector.currentIndex()
        return self.model_types[i]

    def __init__(self, fit, **kwargs):
        super().__init__(fit=fit, **kwargs)
        # LifetimeModelWidgetBase.__init__(self, fit, **kwargs)
        # LifetimeMixModel.__init__(self, fit, **kwargs)

        layout = QtWidgets.QHBoxLayout()

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems([m.name for m in self.model_types])
        layout.addWidget(self.model_selector)

        self.add_button = QtWidgets.QPushButton('Add')
        layout.addWidget(self.add_button)

        self.clear_button = QtWidgets.QPushButton('Clear')
        layout.addWidget(self.clear_button)
        layout.addLayout(layout)

        self.add_button.clicked.connect(self.add_model)
        self.clear_button.clicked.connect(self.clear_models)

        self.model_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.model_layout)

        self._current_model = QtWidgets.QSpinBox()
        self._current_model.setMinimum(0)
        self._current_model.setMaximum(0)
        self.layout_parameter.addWidget(self._current_model)
        self.layout_parameter.addLayout(layout)

    def add_model(self, fit: chisurf.fitting.fit.FitGroup = None):
        layout = QtWidgets.QHBoxLayout()

        if fit is None:
            model = self.selected_fit.model
        else:
            model = fit.model

        xi = len(self) + 1
        fraction_name = f"x({xi})"
        label_text = f"x<sub>{xi}</sub>"
        fraction = chisurf.gui.widgets.fitting.widgets.FittingParameterWidget(
            name=fraction_name,
            value=1.0,
            model=self,
            ub=1.0,
            lb=0.0,
            label_text=label_text,
            layout=layout
        )
        layout.addWidget(fraction)
        model_label = QtWidgets.QLabel(fit.name)
        layout.addWidget(model_label)

        self.model_layout.addLayout(layout)
        self.append(model, fraction)

    def clear_models(self):
        LifetimeMixModel.clear_models(self)
        chisurf.gui.widgets.general.clear_layout(self.model_layout)