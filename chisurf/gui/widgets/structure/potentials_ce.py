from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.gui.widgets
import chisurf.core.structure
from chisurf.core.settings.path_utils import get_path
from chisurf.core.structure.potential.potentials import CEPotential


class CEPotentialWidget(CEPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.core.structure.Structure,
            potential: str = None,
            ca_cutoff: float = 25.0,
            parent=None
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)
        
        # Set default potential path if not provided
        if potential is None:
            potential = str(get_path('chisurf') / 'structure/potential/database/unres.npy')

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("Potential", self)
        layout.addWidget(label, 0, 0)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        layout.addWidget(self.lineEdit, 0, 1)

        self.toolButton = QtWidgets.QPushButton("...", self)
        layout.addWidget(self.toolButton, 0, 2)

        label_3 = QtWidgets.QLabel("CA-cutoff", self)
        layout.addWidget(label_3, 1, 0)

        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox.setMinimum(10.0)
        self.doubleSpinBox.setMaximum(27.0)
        self.doubleSpinBox.setSingleStep(0.5)
        self.doubleSpinBox.setValue(20.0)
        layout.addWidget(self.doubleSpinBox, 1, 1, 1, 2)

        self.actionOpen_potential_file = QtWidgets.QAction("Open potential file", self)
        self.actionOpen_potential_file.triggered.connect(self.onOpenPotentialFile)
        self.toolButton.clicked.connect(self.actionOpen_potential_file.trigger)

        super(CEPotentialWidget, self).__init__(
            structure,
            potential=potential,
            ca_cutoff=ca_cutoff
        )
        self.ca_cutoff = ca_cutoff

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self.lineEdit.setText(str(v))
        try:
            self._potential = np.load(v)
        except (FileNotFoundError, IOError) as e:
            QtWidgets.QMessageBox.warning(
                None,
                "Missing Potential File",
                f"Potential file not found: {v}\n\n"
                f"The UNRES potential file should be located at:\n"
                f"{v}\n\n"
                f"Please check if the file exists or use the '...' button\n"
                f"to select a different potential file."
            )
            self._potential = np.zeros((20, 20))
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                None,
                "Potential File Error",
                f"Error loading potential file {v}:\n{str(e)}"
            )
            self._potential = np.zeros((20, 20))

    @property
    def ca_cutoff(self):
        return float(self.doubleSpinBox.value())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.doubleSpinBox.setValue(float(v))

    def onOpenPotentialFile(self):
        filename = chisurf.gui.widgets.get_filename('Open CE-Potential', 'Numpy file (*.npy)')
        self.potential = filename
