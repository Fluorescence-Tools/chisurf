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

    def update_widget(
            self
    ) -> None:
        for w in self.widgets:
            w.update()

    def update_all(
            self,
            *args,
            **kwargs
    ) -> None:
        pass

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.pltControl, QtWidgets.QWidget):
            self.pltControl.close()
