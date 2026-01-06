from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.gui.widgets
import chisurf.structure
from chisurf.settings.path_utils import get_path
from chisurf.structure.potential.potentials import Ramachandran


class RamachandranWidget(Ramachandran, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            filename: str = None,
            parent=None
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)
        
        # Set default filename path if not provided
        if filename is None:
            filename = str(get_path('chisurf') / 'structure/potential/database/rama_ala_pro_gly.npy')

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("Ramachandran File", self)
        layout.addWidget(label, 0, 0)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        layout.addWidget(self.lineEdit, 0, 1)

        self.toolButton = QtWidgets.QPushButton("...", self)
        layout.addWidget(self.toolButton, 0, 2)

        self.actionLoad_potential = QtWidgets.QAction("Load potential", self)
        self.actionLoad_potential.triggered.connect(self.onOpenFile)
        self.toolButton.clicked.connect(self.actionLoad_potential.trigger)

        # Initialize the parent Ramachandran class
        super().__init__(
            structure,
            filename
        )

    def onOpenFile(self):
        filename = chisurf.gui.widgets.get_filename(
            'Open File',
            'NumPy data files (*.npy)'
        )
        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        try:
            self._filename = v
            self.ramaPot = np.load(v)
            self.lineEdit.setText(str(v))
        except (FileNotFoundError, IOError) as e:
            QtWidgets.QMessageBox.warning(
                None,
                "Missing Ramachandran Potential File",
                f"Ramachandran potential file not found: {v}\n\n"
                f"The Ramachandran potential file should be located at:\n"
                f"{v}\n\n"
                f"Please check if the file exists or use the '...' button\n"
                f"to select a different potential file."
            )
            # Create a dummy potential to prevent crashes
            self._filename = v
            self.ramaPot = np.zeros((5, 129600))  # Same shape as expected
            self.lineEdit.setText(str(v))
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                None,
                "Ramachandran Potential File Error",
                f"Error loading Ramachandran potential file {v}:\n{str(e)}"
            )
            # Create a dummy potential to prevent crashes
            self._filename = v
            self.ramaPot = np.zeros((5, 129600))  # Same shape as expected
            self.lineEdit.setText(str(v))
