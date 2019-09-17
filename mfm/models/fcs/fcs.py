from __future__ import annotations

import os
from qtpy import QtGui

import mfm
from mfm import plots
from mfm.models.parse import ParseModelWidget


class ParseFCSWidget(ParseModelWidget):
    """
    fcs
    """

    plot_classes = [
        (
            plots.LinePlot, {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
        ),
        (plots.FitInfo, {}),
        (plots.ParameterScanPlot, {})
    ]

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
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

        super(ParseFCSWidget, self).__init__(
            fit=fit,
            model_file=fn,
            **kwargs
        )


