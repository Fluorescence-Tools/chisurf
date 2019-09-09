import numpy as np
from qtpy import  QtWidgets, uic

from mfm.fluorescence.tcspc.phasor import Phasor


class PhasorWidget(Phasor, QtWidgets.QWidget):

    @property
    def phasor_omega(self):
        return float(self.doubleSpinBox_12.value()) / 1000.0 * np.pi * 2.0

    @phasor_omega.setter
    def phasor_omega(self, v):
        self.doubleSpinBox_12.setValue(v)

    @property
    def phasor_n(self):
        return int(self.spinBox_5.value())

    @phasor_n.setter
    def phasor_n(self, v):
        self.spinBox_5.setValue(v)

    @property
    def phasor_omega(self):
        return self._phasor_omega / 1000.0 * np.pi * 2.0

    @phasor_omega.setter
    def phasor_omega(self, v):
        self._phasor_omega = v

    @property
    def phasor_n(self):
        return self._phasor_n

    @phasor_n.setter
    def phasor_n(self, v):
        self._phasor_n = float(v)

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Phasor.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/fitting/models/tcspc/phasor_widget.ui', self)
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)

    def onUpdatePhasor(self):
        self.lineEdit.setText(str(self.phasor_siwD0))
        self.lineEdit_3.setText(str(self.phasor_giwD0))

        self.lineEdit_4.setText(str(self.phasor_siwDA))
        self.lineEdit_5.setText(str(self.phasor_giwDA))

        self.lineEdit_6.setText(str(self.phasor_siwE))
        self.lineEdit_7.setText(str(self.phasor_giwE))