from __future__ import annotations

import json
from qtpy import QtWidgets

import chisurf.gui.widgets
import chisurf.structure
from chisurf.structure.av.potential import AvPotential


class AvPotentialWidget(AvPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            parent=None
    ):
        super(AvPotentialWidget, self).__init__()
        QtWidgets.QWidget.__init__(self, parent=parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.groupBox = QtWidgets.QGroupBox("Av-properties", self)
        grid = QtWidgets.QGridLayout(self.groupBox)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        label_4 = QtWidgets.QLabel("nSamples", self.groupBox)
        grid.addWidget(label_4, 0, 0)
        self.spinBox_2 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_2.setMaximum(999999)
        self.spinBox_2.setSingleStep(1000)
        grid.addWidget(self.spinBox_2, 0, 2)

        label_2 = QtWidgets.QLabel("MinAV", self.groupBox)
        grid.addWidget(label_2, 0, 3)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setMinimum(0)
        self.spinBox.setMaximum(999999)
        self.spinBox.setSingleStep(50)
        self.spinBox.setValue(150)
        grid.addWidget(self.spinBox, 0, 4)

        layout.addWidget(self.groupBox)

        self.groupBox_6 = QtWidgets.QGroupBox("Labling", self)
        grid2 = QtWidgets.QGridLayout(self.groupBox_6)
        grid2.setContentsMargins(0, 0, 0, 0)
        grid2.setSpacing(0)

        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_6)
        self.lineEdit_2.setReadOnly(True)
        grid2.addWidget(self.lineEdit_2, 0, 2)

        self.toolButton = QtWidgets.QPushButton("...", self.groupBox_6)
        grid2.addWidget(self.toolButton, 0, 3)

        layout.addWidget(self.groupBox_6)

        self._filename = None

        self.actionOpenLabeling = QtWidgets.QAction("OpenLabeling", self)
        self.actionOpenLabeling.triggered.connect(self.onLoadAvJSON)
        self.toolButton.clicked.connect(self.actionOpenLabeling.trigger)

    def onLoadAvJSON(self):
        self.labeling_file = chisurf.gui.widgets.get_filename(
            description='Open FPS-JSON',
            file_type='FPS-file (*.fps.json)'
        )

    @property
    def labeling_file(self):
        return self._filename

    @labeling_file.setter
    def labeling_file(self, v):
        self._filename = v
        p = json.load(open(v))
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.lineEdit_2.setText(v)

    @property
    def n_av_samples(self):
        return int(self.spinBox_2.value())

    @n_av_samples.setter
    def n_av_samples(self, v):
        self.spinBox_2.setValue(int(v))

    @property
    def min_av(self):
        return int(self.spinBox_2.value())

    @min_av.setter
    def min_av(self, v):
        self.spinBox.setValue(int(v))
