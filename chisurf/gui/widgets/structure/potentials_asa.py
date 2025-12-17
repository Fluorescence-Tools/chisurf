from __future__ import annotations

from qtpy import QtWidgets

import chisurf.structure
from chisurf.structure.potential.potentials import ASA


class AsaWidget(ASA, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("sphere-points", self)
        layout.addWidget(label, 0, 0)
        self.lineEdit = QtWidgets.QLineEdit(self)
        layout.addWidget(self.lineEdit, 0, 1)

        label_2 = QtWidgets.QLabel("Probe-radius [A]", self)
        layout.addWidget(label_2, 1, 0)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        layout.addWidget(self.lineEdit_2, 1, 1)

        self.lineEdit.textChanged.connect(self.setParameterSphere)
        self.lineEdit_2.textChanged.connect(self.setParameterProbe)
        self.lineEdit.setText('590')
        self.lineEdit_2.setText('3.5')

        super(AsaWidget, self).__init__(
            structure,
            **kwargs
        )

    def setParameterSphere(self):
        self.n_sphere_point = int(self.lineEdit.text())

    def setParameterProbe(self):
        self.probe = float(self.lineEdit_2.text())
