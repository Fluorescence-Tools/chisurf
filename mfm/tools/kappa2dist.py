import numpy as np
from PyQt5 import QtWidgets, uic
from guiqwt.builder import make
from guiqwt.plot import CurveDialog

from mfm.fluorescence.anisotropy import s2delta
from mfm.fluorescence.anisotropy.kappa2 import kappasqAllDelta, kappasq_all
from ..fluorescence import general as fluorescence


class Kappa2Dist(QtWidgets.QWidget):

    name = "Kappa2Dist"

    def __init__(self, kappa2=0.667):
        QtWidgets.QWidget.__init__(self)
        self.k2 = list()
        uic.loadUi('./mfm/ui/tools/kappa2_dist.ui', self)
        self.kappa2 = kappa2
        ## Plot
        win = CurveDialog(edit=True, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.kappa2_curve = make.curve([],  [], color="r", linewidth=2)
        #self.kappa2_curve = make.histogram([], color='#ff00ff')
        plot.add_item(self.kappa2_curve)
        self.kappa2_plot = plot
        self.verticalLayout.addWidget(self.kappa2_plot)
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
        self.kappa2_curve.set_data(x[1:], k2hist)
        self.kappa2_plot.do_autoscale()
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
        return fluorescence.kappasq(self.delta, self.SA2, self.SD2)

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
        return self.doubleSpinBox_6.setValue(v)

    @property
    def RappSD(self):
        return float(self.doubleSpinBox_8.value())

    @Rapp_mean.setter
    def RappSD(self, v):
        return self.doubleSpinBox_8.setValue(v)
