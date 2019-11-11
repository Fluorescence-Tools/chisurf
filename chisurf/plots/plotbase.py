from __future__ import annotations

from qtpy import QtWidgets

import chisurf.fitting
import chisurf.widgets


class Plot(
    chisurf.widgets.View
):

    def __init__(
            self,
            fit: chisurf.fitting.fit,
            parent=None,
            **kwargs
    ):
        super().__init__()
        self.parent = parent
        self.fit = fit
        self.pltControl = QtWidgets.QWidget()
        self.widgets = list()

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.pltControl, QtWidgets.QWidget):
            self.pltControl.close()
