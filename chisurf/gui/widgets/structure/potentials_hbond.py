from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.gui.widgets
from chisurf.structure.potential.potentials import HPotential


class HPotentialWidget(HPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure,
            parent,
            cutoff_ca=8.0,
            cutoff_hbond=3.0
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        cutoff_row = QtWidgets.QHBoxLayout()
        cutoff_row.setContentsMargins(0, 0, 0, 0)
        cutoff_row.setSpacing(0)

        label = QtWidgets.QLabel("cutoff ", self)
        cutoff_row.addWidget(label)
        label_4 = QtWidgets.QLabel("CA", self)
        cutoff_row.addWidget(label_4)

        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_2.setDecimals(1)
        cutoff_row.addWidget(self.doubleSpinBox_2)

        label_2 = QtWidgets.QLabel("H", self)
        cutoff_row.addWidget(label_2)

        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox.setDecimals(1)
        cutoff_row.addWidget(self.doubleSpinBox)

        cutoff_row.addItem(
            QtWidgets.QSpacerItem(
                40,
                20,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Minimum
            )
        )

        checks_row = QtWidgets.QHBoxLayout()
        checks_row.setContentsMargins(0, 0, 0, 0)
        checks_row.setSpacing(0)

        self.checkBox_2 = QtWidgets.QCheckBox("CN", self)
        self.checkBox_2.setChecked(True)
        checks_row.addWidget(self.checkBox_2)

        self.checkBox_4 = QtWidgets.QCheckBox("ON", self)
        self.checkBox_4.setChecked(True)
        checks_row.addWidget(self.checkBox_4)

        self.checkBox = QtWidgets.QCheckBox("OH", self)
        self.checkBox.setChecked(True)
        checks_row.addWidget(self.checkBox)

        self.checkBox_3 = QtWidgets.QCheckBox("CH", self)
        self.checkBox_3.setChecked(True)
        checks_row.addWidget(self.checkBox_3)

        cutoff_row.addLayout(checks_row)
        layout.addLayout(cutoff_row, 0, 0, 1, 4)

        label_3 = QtWidgets.QLabel("Potential", self)
        layout.addWidget(label_3, 1, 0)

        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setReadOnly(True)
        layout.addWidget(self.lineEdit_3, 1, 1)

        self.toolButton = QtWidgets.QPushButton("...", self)
        layout.addWidget(self.toolButton, 1, 2)

        self.actionLoad_potential = QtWidgets.QAction("Load potential", self)
        self.actionLoad_potential.triggered.connect(self.onOpenFile)
        self.toolButton.clicked.connect(self.actionLoad_potential.trigger)

        self.checkBox.stateChanged[int].connect(self.updateParameter)
        self.checkBox_2.stateChanged[int].connect(self.updateParameter)
        self.checkBox_3.stateChanged[int].connect(self.updateParameter)
        self.checkBox_4.stateChanged[int].connect(self.updateParameter)
        self.cutoffCA = cutoff_ca
        self.cutoffH = cutoff_hbond
        super().__init__(
            structure,
            cutoff_ca,
            cutoff_hbond
        )

    def onOpenFile(self):
        filename = chisurf.gui.widgets.get_filename(
            'Open File',
            'NumPy data files (*.npy)'
        )
        self.potential = filename

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.load(v)
        # self._hPot = np.load(
        #     v,
        #     skiprows=1,
        #     dtype=np.float64
        # ).T[1:, :]
        self.hPot = self._hPot
        self.lineEdit_3.setText(str(v))

    @property
    def oh(self):
        return int(self.checkBox.isChecked())

    @oh.setter
    def oh(self, v):
        self.checkBox.setChecked(bool(v))

    @property
    def cn(self):
        return int(self.checkBox_2.isChecked())

    @cn.setter
    def cn(self, v):
        self.checkBox_2.setChecked(bool(v))

    @property
    def ch(self):
        return int(self.checkBox_3.isChecked())

    @ch.setter
    def ch(self, v):
        self.checkBox_3.setChecked(bool(v))

    @property
    def on(self):
        return int(self.checkBox_4.isChecked())

    @on.setter
    def on(self, v):
        self.checkBox_4.setChecked(bool(v))

    @property
    def cutoffH(self):
        return float(self.doubleSpinBox.value())

    @cutoffH.setter
    def cutoffH(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def cutoffCA(self):
        return float(self.doubleSpinBox_2.value())

    @cutoffCA.setter
    def cutoffCA(self, v):
        self.doubleSpinBox_2.setValue(float(v))
