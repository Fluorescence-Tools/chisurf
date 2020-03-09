from __future__ import annotations

from qtpy import QtWidgets

import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets

from chisurf.gui.widgets import View


class Plot(
    View
):

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            parent=None,
            plot_controller: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.parent = parent
        self.fit = fit
        if plot_controller is None:
            self.plot_controller = QtWidgets.QWidget()
        else:
            self.plot_controller = plot_controller
        self.widgets = list()

    def update(
            self,
            *args,
            **kwargs
    ) -> None:
        super().update(*args, **kwargs)

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.plot_controller, QtWidgets.QWidget):
            self.plot_controller.close()
