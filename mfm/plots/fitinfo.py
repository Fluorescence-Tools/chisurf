from qtpy import  QtWidgets

import mfm
from mfm.plots import plotbase


class FitInfo(plotbase.Plot):

    name = "Info"

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            **kwargs
    ):
        mfm.plots.Plot.__init__(self, fit)
        self.pltControl = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.textedit = QtWidgets.QPlainTextEdit()
        self.layout.addWidget(self.textedit)

    def update_all(self, *args, **kwargs):
        fit = self.fit
        self.textedit.setPlainText(str(fit))


