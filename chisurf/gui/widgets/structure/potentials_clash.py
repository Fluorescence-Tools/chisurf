from __future__ import annotations

from qtpy import QtWidgets

import chisurf.structure
from chisurf.structure.potential.potentials import ClashPotential


class ClashPotentialWidget(ClashPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            **kwargs
    ):
        QtWidgets.QWidget.__init__(self, parent=kwargs.get('parent'))

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(0)
        label = QtWidgets.QLabel("Clash-tolerance", self)
        row1.addWidget(label)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox.setMinimum(0.01)
        row1.addWidget(self.doubleSpinBox)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(0)
        label_2 = QtWidgets.QLabel("bond length", self)
        row2.addWidget(label_2)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_2.setMinimum(0.25)
        self.doubleSpinBox_2.setMaximum(10.0)
        self.doubleSpinBox_2.setSingleStep(0.25)
        self.doubleSpinBox_2.setValue(1.5)
        row2.addWidget(self.doubleSpinBox_2)
        layout.addLayout(row2)

        super(ClashPotentialWidget, self).__init__(
            structure=structure,
            **kwargs
        )

    @property
    def clash_tolerance(self):
        return float(self.doubleSpinBox.value())

    @clash_tolerance.setter
    def clash_tolerance(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def covalent_radius(self):
        return float(self.doubleSpinBox_2.value())

    @covalent_radius.setter
    def covalent_radius(self, v):
        self.doubleSpinBox_2.setValue(v)
