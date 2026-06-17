from __future__ import annotations

import mdtraj as md
import numpy as np
import tables
from qtpy import QtCore, QtGui, QtWidgets

import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("traj_join")
class JoinTrajectoriesWidget(QtWidgets.QWidget):

    @property
    def stride(self) -> int:
        return

    @property
    def chunk_size(self) -> int:
        return int(self.spinBox.value())

    @property
    def reverse_traj_1(self) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def reverse_traj_2(self) -> bool:
        return bool(self.checkBox.isChecked())

    @property
    def trajectory_filename_1(self) -> str:
        return str(self.lineEdit.text())

    @trajectory_filename_1.setter
    def trajectory_filename_1(self, v: str):
        self.lineEdit.setText(str(v))

    @property
    def trajectory_filename_2(self) -> str:
        return str(self.lineEdit_2.text())

    @trajectory_filename_2.setter
    def trajectory_filename_2(self, v:str):
        self.lineEdit_2.setText(str(v))

    @property
    def join_mode(self) -> str:
        if self.radioButton_2.isChecked():
            return 'time'
        else:
            return 'atoms'

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="join_traj.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionOpen_first_trajectory.triggered.connect(self.onOpenTrajectory_1)
        self.actionOpen_second_trajectory.triggered.connect(self.onOpenTrajectory_2)
        self.actionSave_joined_trajectory.triggered.connect(self.onJoinTrajectories)
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
        self.gridLayout.addWidget(self._log, 3, 0, 1, 6)
        self.gridLayout.setRowStretch(3, 1)
        self.pushButton.setIcon(self._empty_icon())
        self.toolButton.setIcon(self._empty_icon())
        self.toolButton_2.setIcon(self._empty_icon())
        self._append_log("Ready")

    def _append_log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    def onJoinTrajectories(
            self,
            target_filename: str = None
    ) -> None:
        if target_filename is None:
            target_filename = str(
                QtWidgets.QFileDialog.getSaveFileName(
                    None, 'Save H5-Model file', '', 'H5-files (*.h5)'
                )
            )[0]
        if not target_filename:
            self._append_log("Join cancelled")
            return

        fn1 = self.trajectory_filename_1
        fn2 = self.trajectory_filename_2
        self._append_log(f"Join mode: {self.join_mode}")
        self._append_log(f"Trajectory 1: {fn1}")
        self._append_log(f"Trajectory 2: {fn2}")

        r1 = self.reverse_traj_1
        r2 = self.reverse_traj_2

        traj_1 = md.load_frame(fn1, index=0)
        traj_2 = md.load_frame(fn2, index=0)

        # Create empty trajectory
        if self.join_mode == 'time':
            traj_join = traj_1.join(traj_2)
            axis = 0
        elif self.join_mode == 'atoms':
            traj_join = traj_1.stack(traj_2)
            axis = 1

        target_traj = md.Trajectory(xyz=np.empty((0, traj_join.n_atoms, 3)), topology=traj_join.topology)
        target_traj.save(target_filename)

        chunk_size = self.chunk_size
        table = tables.open_file(target_filename, 'a')
        try:
            for i, (c1, c2) in enumerate(zip(md.iterload(fn1, chunk=chunk_size), md.iterload(fn2, chunk=chunk_size))):
                xyz_1 = c1.xyz[::-1] if r1 else c1.xyz
                xyz_2 = c2.xyz[::-1] if r2 else c2.xyz
                xyz = np.concatenate((xyz_1, xyz_2), axis=axis)

                table.root.coordinates.append(xyz)
                table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))
                if (i + 1) % 10 == 0:
                    self._append_log(f"Joined {i + 1} chunks")
        finally:
            table.close()
        self._append_log("Join complete")

    def onOpenTrajectory_1(
            self,
            filename: str = None
    ):
        if filename is None:
            #self.trajectory_filename_1 = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
            filename = chisurf.gui.widgets.get_filename(
                'Open H5-Model file', 'H5-files (*.h5)'
            )
            self.trajectory_filename_1 = filename

    def onOpenTrajectory_2(
            self,
            filename: str = None
    ):
        if filename is None:
            #self.trajectory_filename_2 = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
            filename = chisurf.gui.widgets.get_filename(
                'Open H5-Model file', 'H5-files (*.h5)'
            )
            self.trajectory_filename_2 = filename
