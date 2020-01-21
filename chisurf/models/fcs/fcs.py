from __future__ import annotations

import os
from qtpy import QtGui

import chisurf
import chisurf.fitting
from chisurf import plots
from chisurf.models.parse import ParseModelWidget


class ParseFCSWidget(
    ParseModelWidget
):

    plot_classes = [
        (
            plots.LinePlot, {
                'scale_x': 'log',
                'd_scaley': 'lin',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y'
            }
        ),
        (plots.FitInfo, {}),
        (plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {})
    ]

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        self.icon = icon

        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models.yaml'
        )

        super().__init__(
            fit=fit,
            model_file=fn,
            **kwargs
        )

