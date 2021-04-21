from __future__ import annotations

from qtpy import  QtWidgets

import chisurf.fitting
from chisurf.plots import plotbase


class FitInfo(plotbase.Plot):

    name = "Info"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__(
            fit,
            parent=parent,
            **kwargs
        )
        self.plot_controller = QtWidgets.QWidget()

        self.textedit = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.textedit)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        fit = self.fit
        self.textedit.setPlainText(str(fit))


