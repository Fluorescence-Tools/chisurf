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
        self.pltControl = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.textedit = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.textedit)

    def update_all(
            self,
            *args,
            **kwargs
    ):
        fit = self.fit
        self.textedit.setPlainText(str(fit))


