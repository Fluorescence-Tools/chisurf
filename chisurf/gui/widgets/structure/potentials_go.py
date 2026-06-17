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

        GoPotential.__init__(self, structure=structure)

        self.lineEdit.textChanged.connect(self.setGo)
        self.lineEdit_2.textChanged.connect(self.setGo)
        self.lineEdit_3.textChanged.connect(self.setGo)

    @property
    def epsilon(self):
        return float(self.lineEdit.text())

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.lineEdit.setText(str(value))

    @property
    def nnEFactor(self):
        return float(self.lineEdit_2.text())

    @nnEFactor.setter
    def nnEFactor(self, value: float) -> None:
        self.lineEdit_2.setText(str(value))

    @property
    def cutoff(self):
        return float(self.lineEdit_3.text())

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self.lineEdit_3.setText(str(value))

    @property
    def native_cutoff_on(self):
        return bool(self.checkBox.isChecked())

    @native_cutoff_on.setter
    def native_cutoff_on(self, value: bool) -> None:
        self.checkBox.setChecked(bool(value))

    @property
    def non_native_contact_on(self):
        return bool(self.checkBox_2.isChecked())

    @non_native_contact_on.setter
    def non_native_contact_on(self, value: bool) -> None:
        self.checkBox_2.setChecked(bool(value))
