from __future__ import annotations

import abc

from qtpy import QtGui, QtWidgets

import chisurf
import chisurf.core.fitting.fit
import chisurf.gui.plots
from chisurf.core.models.model import Model


class ModelWidget(Model, QtWidgets.QWidget):
    """Base class for GUI widgets that host a :class:`Model`.

    Subclasses combine the parameter-handling logic from :class:`Model`
    with a Qt widget used in the ChiSurf GUI. They typically implement
    :meth:`update_widgets` to synchronize GUI controls with the underlying
    parameters.
    """

    try:
        plot_classes = [
            (
                chisurf.gui.plots.LinePlot, {
                    'scale_x': 'lin',
                    'd_scaley': 'log',
                    'r_scaley': 'lin',
                    'x_label': 'x',
                    'y_label': 'y'
                }
            ),
            (chisurf.gui.plots.FitInfo, {}),
            (chisurf.gui.plots.ParameterScanPlot, {}),
            (chisurf.gui.plots.ResidualPlot, {})
        ]
    except Exception:
        plot_classes = []

    def update_plots(self, *args, **kwargs) -> None:
        for p in self.fit.plots:
            p.update(*args, **kwargs)

    @abc.abstractmethod
    def update_widgets(self) -> None:
        for parameter in self.parameters:
            parameter.update()

    @abc.abstractmethod
    def update(self) -> None:
        super().update()
        self.update_widgets()
        self.update_plots()

    def __init__(
            self,
            fit: chisurf.core.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            *args,
            **kwargs
    ):
        super().__init__(fit, *args, **kwargs)
        self.plots = list()
        if icon is None:
            icon = QtGui.QIcon(":/icons/document-open.png")
        self.icon = icon
