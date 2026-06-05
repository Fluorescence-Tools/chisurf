from __future__ import annotations

from qtpy import QtWidgets

import chisurf.core.structure
from chisurf.core.structure.potential.potentials import GoPotential


class GoPotentialWidget(GoPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.core.structure.Structure = None,
            **kwargs
    ):
        QtWidgets.QWidget.__init__(self, parent=kwargs.get('parent'))
        GoPotential.__init__(self, structure=structure)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        label = QtWidgets.QLabel("epsilon", self)
        grid.addWidget(label, 0, 0)
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setText("1.0")
        grid.addWidget(self.lineEdit, 0, 1)

        label_2 = QtWidgets.QLabel("native cutoff [A]", self)
        grid.addWidget(label_2, 1, 0)
        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setText("6.5")
        grid.addWidget(self.lineEdit_3, 1, 1)
        self.checkBox = QtWidgets.QCheckBox("", self)
        self.checkBox.setChecked(True)
        grid.addWidget(self.checkBox, 1, 2)

        label_3 = QtWidgets.QLabel("non-native / scale", self)
        grid.addWidget(label_3, 2, 0)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setText("0.7")
        grid.addWidget(self.lineEdit_2, 2, 1)
        self.checkBox_2 = QtWidgets.QCheckBox("", self)
        self.checkBox_2.setChecked(True)
        grid.addWidget(self.checkBox_2, 2, 2)

        layout.addLayout(grid)

        self.lineEdit.textChanged.connect(self.setGo)
        self.lineEdit_2.textChanged.connect(self.setGo)
        self.lineEdit_3.textChanged.connect(self.setGo)

    @property
    def native_cutoff_on(self):
        return bool(self.checkBox.isChecked())

    @property
    def non_native_contact_on(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def epsilon(self):
        return float(self.lineEdit.text())

    @property
    def nnEFactor(self):
        return float(self.lineEdit_2.text())

    @property
    def cutoff(self):
        return float(self.lineEdit_3.text())
