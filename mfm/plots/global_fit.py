from qtpy import  QtCore, QtWidgets

from mfm.plots.plotbase import Plot


class GlobalFitPlot(Plot):
    name = "Global-Fits"

    def __init__(self, fit, logy=False, logx=False):
        super(GlobalFitPlot, self).__init__(fit)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.pltControl = QtWidgets.QWidget()
        self.fit = fit

    def update_all(self, **kwargs):
        fit = self.fit
        layout = self.layout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter1)
        for i, f in enumerate(fit.model.fits):
            splitter1.addWidget(QtWidgets.QLabel(f.name))
