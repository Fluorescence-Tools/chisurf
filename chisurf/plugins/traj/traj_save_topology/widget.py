import mdtraj as md
from qtpy import QtCore, QtGui, QtWidgets

import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("traj_save_topology")
class SaveTopology(QtWidgets.QWidget):

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    @chisurf.gui.decorators.init_with_ui(ui_filename="save_topology.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_clash_free_trajectory.triggered.connect(self.onSaveTopology)
        self._setup_layout()

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
        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Log")
        self._log.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.gridLayout.addWidget(self._log, 3, 0, 1, 3)
        self.gridLayout.setRowStretch(3, 1)
        self.pushButton.setIcon(self._empty_icon())
        self.toolButton.setIcon(self._empty_icon())
        self._append_log("Ready")

    def _append_log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    def onSaveTopology(self):
        target_filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Save PDB-file', '', 'PDB-files (*.pdb)'))[0]
        if not target_filename:
            self._append_log("Save cancelled")
            return
        filename = self.trajectory_filename
        try:
            self._append_log(f"Loading first frame: {filename}")
            frame_0 = md.load_frame(filename, 0)
            self._append_log(f"Saving topology to: {target_filename}")
            frame_0.save(target_filename)
            self._append_log("Topology saved")
        except Exception as exc:
            self._append_log(f"Save failed: {exc}")
            raise

    def onOpenTrajectory(self, filename=None):
        if filename is None:
            #self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
            filename = chisurf.gui.widgets.get_filename('Open H5-Model file', 'H5-files (*.h5)')
            self.trajectory_filename = filename
