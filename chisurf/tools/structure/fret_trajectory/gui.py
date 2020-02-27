from __future__ import annotations

import tempfile

import mdtraj as md
from qtpy import QtWidgets

import chisurf.decorators
from chisurf.fio import coordinates
import chisurf.widgets
from chisurf.widgets.pdb import PDBSelector
from chisurf.tools.structure.fret_trajectory import CalculateTransfer


class Structure2Transfer(
    QtWidgets.QWidget,
    CalculateTransfer
):

    name = "Structure2Transfer"

    @chisurf.decorators.init_with_ui(ui_filename="structure2transfer.ui")
    def __init__(
            self,
            verbose: bool = True,
            *args,
            **kwargs
    ):
        self._trajectory_file = ''
        self.filenames = list()
        self._settings = {
            't_step': 1.0
        }

        self.verbose = verbose
        self.d1 = PDBSelector()
        self.d2 = PDBSelector(show_labels=False)

        self.a1 = PDBSelector()
        self.a2 = PDBSelector(show_labels=False)

        self.horizontalLayout_2.addWidget(self.d1)
        self.horizontalLayout_3.addWidget(self.a1)
        self.horizontalLayout_2.addWidget(self.d2)
        self.horizontalLayout_3.addWidget(self.a2)

        self.actionOpen_trajectory.triggered.connect(self.onLoadTrajectory)
        self.actionProcess_trajectory.triggered.connect(self.calc)
        self.hide()

    def calc(self, *args, **kwargs):
        output_file = chisurf.widgets.save_file(description='Output-file', file_type='All files (*.csv)')
        for filename in self.filenames:
            CalculateTransfer.calc(self, verbose=True,
                                   output_file=output_file,
                                   trajectory_file=filename)

    @property
    def stride(self):
        return int(self.spinBox.value())

    @stride.setter
    def stride(self, v):
        self.spinBox.setValue(v)

    @property
    def donor(self):
        return self.d1.atom_number, self.d2.atom_number

    @property
    def acceptor(self):
        return self.a1.atom_number, self.a2.atom_number

    @property
    def forster_radius(self):
        return float(self.doubleSpinBox.value())

    @forster_radius.setter
    def forster_radius(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def tau0(self):
        return self.doubleSpinBox_2.value()

    @tau0.setter
    def tau0(self, v):
        self.doubleSpinBox_2.setValue(float(v))

    @property
    def dipoles(self):
        return self.checkBox.isChecked()

    @dipoles.setter
    def dipoles(self, v):
        self.checkBox.setChecked(bool(v))

    @property
    def pdb(self):
        if self._pdb is None:
            raise ValueError("No pdb file set yet.")
        return self._pdb

    @pdb.setter
    def pdb(self, v):
        if isinstance(v, str):
            v = coordinates.read(v, verbose=self.verbose)
        self._pdb = v

    @property
    def trajectory_file(self):
        return str(self.lineEdit_3.text())

    @trajectory_file.setter
    def trajectory_file(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def topology_file(self):
        return str(self.lineEdit.text())

    @topology_file.setter
    def topology_file(self, value):
        self.pdb = str(value)

    def onLoadTrajectory(self):
        #self.trajectory_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Trajectory-File', '.h5', 'H5-Trajectory-Files (*.h5)'))
        filenames = chisurf.widgets.open_files('Open Trajectory-File', 'H5-Trajectory-Files (*.h5)')
        self.filenames = filenames
        self.trajectory_file = filenames[0]

        frame0 = md.load_frame(self.trajectory_file, 0)

        _, tmp = tempfile.mkstemp(
            suffix=".pdb"
        )
        frame0.save(tmp)

        self.topology_file = tmp

        self.d1.atoms = self.pdb
        self.d2.atoms = self.pdb

        self.a1.atoms = self.pdb
        self.a2.atoms = self.pdb


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Structure2Transfer()

    w.show()
    sys.exit(app.exec_())
