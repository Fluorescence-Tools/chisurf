from __future__ import annotations

import os
import numpy as np
from qtpy import QtWidgets, uic

from chisurf.fluorescence.tcspc.phasor import Phasor


class PhasorWidget(Phasor, QtWidgets.QWidget):

    @property
    def phasor_omega(
            self
    ) -> float:
        return float(self.doubleSpinBox_12.value()) / 1000.0 * np.pi * 2.0

    @phasor_omega.setter
    def phasor_omega(
            self,
            v: float
    ):
        self.doubleSpinBox_12.setValue(v)

    @property
    def phasor_n(self):
        return int(self.spinBox_5.value())

    @phasor_n.setter
    def phasor_n(self, v):
        self.spinBox_5.setValue(v)

    def __init__(
            self,
            **kwargs
    ):
        super(PhasorWidget, self).__init__(**kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "phasor_widget.ui"
            ),
            self
        )
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)

    def onUpdatePhasor(self):
        self.lineEdit.setText(str(self.phasor_siwD0))
        self.lineEdit_3.setText(str(self.phasor_giwD0))

        self.lineEdit_4.setText(str(self.phasor_siwDA))
        self.lineEdit_5.setText(str(self.phasor_giwDA))

        self.lineEdit_6.setText(str(self.phasor_siwE))
        self.lineEdit_7.setText(str(self.phasor_giwE))