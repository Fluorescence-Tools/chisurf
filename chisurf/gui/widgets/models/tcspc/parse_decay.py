from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import chisurf
from chisurf import typing
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.plots
import chisurf.core.curve

from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.core.models.tcspc.parse.tcspc_parse import ParseDecayModel
import chisurf.core.models.parse.widget

# These will be imported from the new module structure
from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget

if TYPE_CHECKING:
    from chisurf.core.fitting.fit import FitGroup


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

    plot_classes = [
        (
            chisurf.gui.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'time (ns)',
                'y_label': 'counts',
                'plot_irf': True
            }
         ),
        (chisurf.gui.plots.FitTablePlot, {}),
        (chisurf.gui.plots.FitInfo, {}),
        (chisurf.gui.plots.ParameterScanPlot, {}),
        (chisurf.gui.plots.ResidualPlot, {})
    ]

    # TODO: needs docstring
    def get_curves(self, copy_curves: bool = False) -> typing.Dict[str, chisurf.core.curve.Curve]:
        """Return a dictionary of curves for plotting."""
        d = super().get_curves(copy_curves)
        d['IRF'] = self.convolve.irf
        return d

    # TODO: needs docstring
    def __init__(
            self,
            fit: FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        """Initialize the instance."""
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
        pw = chisurf.core.models.parse.widget.ParseFormulaWidget(
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
