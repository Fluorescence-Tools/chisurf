from __future__ import annotations

import tempfile

import mdtraj as md
from qtpy import QtCore, QtGui, QtWidgets

import chisurf.core.decorators
import chisurf.gui.decorators
from chisurf.core.fio.structure import coordinates
import chisurf.gui.widgets
from .traj2fret import CalculateTransfer
from chisurf.gui.widgets.pdb import PDBSelector

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("fret_trajectory")
class Structure2Transfer(
    QtWidgets.QWidget,
    CalculateTransfer
):

    name = "Structure2Transfer"

    @chisurf.gui.decorators.init_with_ui(ui_filename="structure2transfer.ui")
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
        self._setup_layout()
        self.hide()

    def _empty_icon(self):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        return QtGui.QIcon(pixmap)

    def _setup_layout(self) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.setMaximumSize(16777215, 16777215)
        self.groupBox.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.groupBox_2.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.groupBox_3.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Preferred
        )
        self.groupBox_5.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred
        )
        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Log")
        self._log.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.gridLayout_5.addWidget(self._progress, 3, 0, 1, 2)
        self.gridLayout_5.addWidget(self._log, 4, 0, 1, 2)
        self.gridLayout_5.setRowStretch(4, 1)
        self.pushButton_2.setIcon(self._empty_icon())
        self.toolButton_2.setIcon(self._empty_icon())
        self._append_log("Ready")

    def _append_log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    def calc(self, *args, **kwargs):
        output_file = chisurf.gui.widgets.save_file(description='Output-file', file_type='All files (*.csv)')
        if not output_file:
            self._append_log("Process cancelled")
            return
        filenames = self.filenames or ([self.trajectory_file] if self.trajectory_file else [])
        if not filenames:
            self._append_log("No trajectory selected")
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        try:
            for index, filename in enumerate(filenames, start=1):
                self._append_log(f"Processing {index}/{len(filenames)}: {filename}")
                CalculateTransfer.calc(self, verbose=False,
                                       output_file=output_file if len(filenames) == 1 else f"{output_file}.{index}.csv",
                                       trajectory_file=filename)
                self._append_log(f"Finished {filename}")
        except Exception as exc:
            self._append_log(f"Processing failed: {exc}")
            raise
        finally:
            self._progress.setVisible(False)
            self._progress.setRange(0, 100)
            self._progress.setValue(100)

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
        filenames = chisurf.gui.widgets.open_files('Open Trajectory-File', 'H5-Trajectory-Files (*.h5)')
        if not filenames:
            return
        self.filenames = filenames
        self.trajectory_file = filenames[0]
        self._append_log(f"Loaded {len(filenames)} trajectory file(s)")

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
