from __future__ import annotations

import os
import pathlib
from qtpy import QtGui

import chisurf
import chisurf.fitting
from chisurf import plots
from chisurf.models.parse.widget import ParseModelWidget


class ParseFCSWidget(ParseModelWidget):

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
        (plots.FitTablePlot, {}),
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
        fn = pathlib.Path(__file__).parent / 'models.yaml'
        super().__init__(fit=fit, model_file=fn, **kwargs)

        # Ensure the default selected model is "Diffusion with one bunching"
        # which corresponds to the YAML entry "3D Gauss, 1 bunching".
        # Allow callers to override via kwargs['model_name'] if provided.
        default_model = kwargs.get('model_name', "3D Gauss, 1 bunching")
        try:
            if hasattr(self, 'parse') and default_model in self.parse.models:
                self.parse.model_name = default_model
                # Update equation, parameters, and UI to reflect the selection
                self.parse.onModelChanged()
        except Exception:
            # Be forgiving during initialization if UI is not fully wired yet
            pass

