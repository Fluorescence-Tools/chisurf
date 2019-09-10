from __future__ import annotations

import sys
import os
import numpy as np
from qtpy import QtWidgets, uic
import pyqtgraph as pg
import pyqtgraph.dockarea
import qdarkstyle

import mfm.fluorescence.anisotropy.kappa2
from mfm.fluorescence.anisotropy.kappa2 import s2delta
from mfm.fluorescence.anisotropy.kappa2 import kappasqAllDelta, kappasq_all


class Kappa2Dist(QtWidgets.QWidget):

    name = "Kappa2Dist"

    def __init__(self, kappa2=0.667):
        QtWidgets.QWidget.__init__(self)
        self.k2 = list()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'kappa2_dist.ui'
            ),
            self
        )
        self.kappa2 = kappa2

        ## pyqtgraph
        area = pyqtgraph.dockarea.DockArea()
        self.verticalLayout.addWidget(area)
        d1 = pyqtgraph.dockarea.Dock("res", size=(500, 80), hideTitle=True)
        p1 = pg.PlotWidget()
        d1.addWidget(p1)
        area.addDock(d1, 'top')
        self.kappa2_plot = p1.getPlotItem()
        self.kappa2_curve = self.kappa2_plot.plot(x=[0.0], y=[0.0], name='kappa2')

        ## Connections
        self.pushButton.clicked.connect(self.onUpdateHist)
        self.doubleSpinBox_4.valueChanged.connect(self.onUpdateRapp)
        self.hide()

    def onUpdateHist(self):
        if self.model == "cone":
            if self.rAD_known:
                x, k2hist, self.k2 = kappasqAllDelta(self.delta, self.SD2, self.SA2, self.step, self.n_bins)
            else:
                x, k2hist, self.k2 = kappasq_all(self.SD2, self.SA2, n=self.n_bins, m=self.n_bins)
        elif self.model == "diffusion":
            pass
        self.k2scale = x
        self.k2hist = k2hist
        self.kappa2_curve.setData(x=x[1:], y=k2hist)

        self.onUpdateRapp()

    def onUpdateRapp(self):
        k2 = self.k2_true
        k2scale = self.k2scale[1:]
        k2hist = self.k2hist
        # Rapp
        Rapp_scale = (1. / k2 * k2scale)**(1.0/6.0)
        Rapp_mean = np.dot(k2hist, Rapp_scale)/sum(k2hist)
        self.Rapp_mean = Rapp_mean
        Rapp_sd = np.sqrt(np.dot(k2hist, (Rapp_scale-Rapp_mean)**2) / sum(k2hist))
        self.RappSD = Rapp_sd
        # kappa2
        self.k2_mean = np.dot(k2hist, k2scale)/sum(k2hist)
        self.k2_sd = np.sqrt(np.dot(k2hist, (k2scale-self.k2_mean)**2) / sum(k2hist))


    @property
    def model(self):
        if self.radioButton_2.isChecked():
            return "cone"
        elif self.radioButton.isChecked():
            return "diffusion"

    @property
    def rAD_known(self):
        return self.checkBox.isChecked()

    @property
    def k2_mean(self):
        return float(self.doubleSpinBox_10.value())

    @k2_mean.setter
    def k2_mean(self, v):
        self.doubleSpinBox_10.setValue(v)

    @property
    def k2_sd(self):
        return float(self.doubleSpinBox_9.value())

    @k2_sd.setter
    def k2_sd(self, v):
        self.doubleSpinBox_9.setValue(v)

    @property
    def min_max(self):
        k2 = self.k2.flatten()
        return min(k2), max(k2)

    @property
    def k2_est(self):
        return mfm.fluorescence.anisotropy.kappa2.kappasq(self.delta, self.SA2, self.SD2)

    @property
    def n_bins(self):
        return int(self.spinBox.value())

    @property
    def r_0(self):
        return float(self.doubleSpinBox_2.value())

    @property
    def r_Dinf(self):
        return float(self.doubleSpinBox.value())

    @property
    def r_ADinf(self):
        return float(self.doubleSpinBox_7.value())

    @property
    def r_Ainf(self):
        return float(self.doubleSpinBox_5.value())

    @property
    def step(self):
        return float(self.doubleSpinBox_3.value())

    @property
    def SD2(self):
        return -np.sqrt(self.r_Dinf/self.r_0)

    @property
    def SA2(self):
        return np.sqrt(self.r_Ainf/self.r_0)

    @property
    def delta(self):
        return s2delta(self.r_0, self.SD2, self.SA2, self.r_ADinf)

    @property
    def k2_true(self):
        return float(self.doubleSpinBox_4.value())

    @property
    def Rapp_mean(self):
        return float(self.doubleSpinBox_6.value())

    @Rapp_mean.setter
    def Rapp_mean(self, v):
        self.doubleSpinBox_6.setValue(v)

    @property
    def RappSD(self):
        return float(self.doubleSpinBox_8.value())

    @RappSD.setter
    def RappSD(self, v):
        self.doubleSpinBox_8.setValue(v)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
