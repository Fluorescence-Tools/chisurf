from PyQt5 import Qt, QtCore, QtGui, QtWidgets, uic
import mfm
from mfm.plots import plotbase


class FitInfo(plotbase.Plot):

    name = "Info"

    def __init__(self, fit, **kwargs):
        mfm.plots.Plot.__init__(self, fit)
        self.pltControl = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.textedit = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.textedit)

    def update_all(self, *args, **kwargs):
        fit = self.fit
        self.textedit.setPlainText(str(fit))


