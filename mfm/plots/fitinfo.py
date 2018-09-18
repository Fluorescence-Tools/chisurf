from PyQt4 import QtGui, QtCore, uic, Qt
import mfm
from mfm.plots import plotbase


class FitInfo(plotbase.Plot):

    name = "Info"

    def __init__(self, fit, **kwargs):
        mfm.plots.Plot.__init__(self, fit)
        self.pltControl = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(self)
        self.textedit = QtGui.QPlainTextEdit()
        self.layout.addWidget(self.textedit)

    def update_all(self, *args, **kwargs):
        fit = self.fit
        self.textedit.setPlainText(str(fit))


