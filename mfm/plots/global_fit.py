from PyQt4 import QtGui, QtCore
from guiqwt.plot import CurveDialog
from guiqwt.builder import make
import numpy as np
from mfm.plots.plotbase import Plot


class GlobalFitPlot(Plot):
    name = "Global-Fits"

    def __init__(self, fit, logy=False, logx=False):
        Plot.__init__(self, fit)
        self.layout = QtGui.QVBoxLayout(self)
        self.pltControl = QtGui.QWidget()
        self.fit = fit

    def update_all(self, **kwargs):
        fit = self.fit
        layout = self.layout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

        splitter1 = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter1)
        for i, f in enumerate(fit.model.fits):
            splitter1.addWidget(QtGui.QLabel(f.name))
