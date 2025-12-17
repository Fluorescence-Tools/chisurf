from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.gui.widgets
import chisurf.structure
from chisurf.structure.potential.potentials import MJPotential


class MJPotentialWidget(MJPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            filename: str = './mfm/structure/potential/database/mj.npy',
            ca_cutoff: float = 6.5
    ):
        QtWidgets.QWidget.__init__(self)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("MJ-File", self)
        layout.addWidget(label, 0, 0)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        layout.addWidget(self.lineEdit, 0, 1)

        self.pushButton = QtWidgets.QPushButton("load", self)
        layout.addWidget(self.pushButton, 0, 2)

        label_2 = QtWidgets.QLabel("CA-cutoff", self)
        layout.addWidget(label_2, 1, 0)

        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setText("6.5")
        layout.addWidget(self.lineEdit_2, 1, 1)

        self.pushButton.clicked.connect(self.onOpenFile)

        super(MJPotentialWidget, self).__init__(structure, filename, ca_cutoff)
        self.potential = filename
        self.ca_cutoff = ca_cutoff

    def onOpenFile(self):
        filename = chisurf.gui.widgets.get_filename(
            'Open MJ-Potential',
            'CSV data files (*.npy)'
        )
        self.potential = filename

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(
            self,
            v: str
    ):
        self.mjPot = np.loadtxt(v)
        self.lineEdit.setText(v)

    @property
    def ca_cutoff(self) -> float:
        return float(self.lineEdit_2.text())

    @ca_cutoff.setter
    def ca_cutoff(
            self,
            v: float
    ):
        self.lineEdit_2.setText(str(v))
