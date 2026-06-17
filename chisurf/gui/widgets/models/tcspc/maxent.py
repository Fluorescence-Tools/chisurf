"""GUI widget classes for MaxEnt TCSPC lifetime/FRET MEM models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chisurf as cs
from chisurf.core.models.tcspc.maxent import MaxEntFRETModel, MaxEntLifetimeModel
from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from qtpy import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from chisurf.core.fitting.fit import Fit, FitGroup


class MaxEntWidgetBase(ModelWidget):

    plot_classes = [
        (
            cs.gui.plots.LinePlot,
            {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'time (ns)',
                'y_label': 'counts',
                'plot_irf': True,
            }
        ),
        (cs.gui.plots.FitTablePlot, {}),
        (cs.gui.plots.FitInfo, {}),
        (cs.gui.plots.ParameterScanPlot, {}),
        (cs.gui.plots.ResidualPlot, {}),
    ]

    def __init__(
        self,
        fit: Fit,
        icon: QtGui.QIcon | None = None,
        **kwargs
    ) -> None:
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(fit=fit, icon=icon)

        self._convolve = ConvolveWidget(
            name='convolve', fit=fit,
            hide_curve_convolution=False,
        )
        self._generic = GenericWidget(fit=fit, parent=self)
        self._corrections = CorrectionsWidget(fit=fit)
        self._param_box = self._build_param_box()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._convolve)
        layout.addWidget(self._generic)
        layout.addWidget(self._corrections)
        layout.addWidget(self._param_box)
        self.setLayout(layout)

    def _build_param_box(self) -> QtWidgets.QGroupBox:
        raise NotImplementedError

    @staticmethod
    def _make_param_row(
        param,
        label: str,
    ) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        cs.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=param,
            layout=row,
            label_text=label,
        )
        return row


class MaxEntLifetimeModelWidget(MaxEntWidgetBase, MaxEntLifetimeModel):

    name = "MaxEnt Lifetime MEM"

    def __init__(self, fit: FitGroup, **kwargs) -> None:
        super().__init__(fit=fit, **kwargs)

    def _build_param_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("MaxEnt (lifetime)")
        layout = QtWidgets.QVBoxLayout(box)
        layout.addLayout(self._make_param_row(self._nu_log10, "log10(nu)"))
        layout.addLayout(self._make_param_row(self._tau_min, "tau_min (ns)"))
        layout.addLayout(self._make_param_row(self._tau_max, "tau_max (ns)"))
        layout.addLayout(self._make_param_row(self._tau_bins, "tau_bins"))
        return box


class MaxEntFRETModelWidget(MaxEntWidgetBase, MaxEntFRETModel):

    name = "MaxEnt FRET Distance MEM"

    def __init__(self, fit: FitGroup, **kwargs) -> None:
        super().__init__(fit=fit, **kwargs)

    def _build_param_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("MaxEnt (FRET)")
        layout = QtWidgets.QVBoxLayout(box)
        layout.addLayout(self._make_param_row(self._nu_log10, "log10(nu)"))
        layout.addLayout(self._make_param_row(self._r_min_frac, "r_min_frac"))
        layout.addLayout(self._make_param_row(self._r_max_frac, "r_max_frac"))
        layout.addLayout(self._make_param_row(self._r_bins, "r_bins"))
        return box
