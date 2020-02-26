from __future__ import annotations

from qtpy import QtCore, QtWidgets

import chisurf.fitting
from chisurf.plots.plotbase import Plot


class GlobalFitPlot(Plot):
    name = "Global-Fits"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            logy: bool = False,
            logx: bool = False
    ):
        super(GlobalFitPlot, self).__init__(fit)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.pltControl = QtWidgets.QWidget()
        self.fit = fit

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        fit = self.fit
        layout = self.layout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter1)
        for i, f in enumerate(fit.model.fits):
            splitter1.addWidget(QtWidgets.QLabel(f.name))
