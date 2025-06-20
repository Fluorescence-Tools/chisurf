from __future__ import annotations

import pathlib

import chisurf
from chisurf import typing
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.plots
import chisurf.curve
import chisurf.fitting.fit

from chisurf.models.model import ModelWidget
from chisurf.models.tcspc.parse.tcspc_parse import ParseDecayModel
import chisurf.models.parse.widget

# These will be imported from the new module structure
from chisurf.models.tcspc.widgets.convolve import ConvolveWidget
from chisurf.models.tcspc.widgets.generic import GenericWidget
from chisurf.models.tcspc.widgets.corrections import CorrectionsWidget


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

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
        (chisurf.plots.ResidualPlot, {})
    ]

    def get_curves(self, copy_curves: bool = False) -> typing.Dict[str, chisurf.curve.Curve]:
        d = super().get_curves(copy_curves)
        d['IRF'] = self.convolve.irf
        return d

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super(ModelWidget, self).__init__(fit=fit, icon=icon)
        super(ParseDecayModel, self).__init__(fit=fit, icon=icon)

        self.convolve = ConvolveWidget(
            fit=fit,
            model=self,
            show_convolution_mode=False,
            dt=fit.data.dx,
            **kwargs
        )
        generic = GenericWidget(
            fit=fit,
            parent=self,
            model=self,
            **kwargs
        )
        fn = pathlib.Path(__file__).parent.parent / 'tcspc.models.json'
        pw = chisurf.models.parse.widget.ParseFormulaWidget(
            model=self,
            model_file=fn
        )
        corrections = CorrectionsWidget(
            fit=fit,
            model=self,
            **kwargs
        )

        self.fit = fit
        super().__init__(
            fit=fit,
            parse=pw,
            icon=icon,
            convolve=self.convolve,
            generic=generic,
            corrections=corrections
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(corrections)
        layout.addWidget(pw)
        self.setLayout(layout)