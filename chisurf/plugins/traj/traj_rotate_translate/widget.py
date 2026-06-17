from __future__ import annotations
from chisurf import typing

import numpy as np
import tables
from qtpy import QtCore, QtGui, QtWidgets
import mdtraj

import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.core.structure import translate, rotate

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("traj_rotate_translate")
class RotateTranslateTrajectoryWidget(QtWidgets.QWidget):
    # WORKS

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def rotation_matrix(self):
        r = np.array(
            [
                [float(self.lineEdit_3.text()), float(self.lineEdit_6.text()), float(self.lineEdit_9.text())],
                [float(self.lineEdit_4.text()), float(self.lineEdit_7.text()), float(self.lineEdit_10.text())],
                [float(self.lineEdit_5.text()), float(self.lineEdit_8.text()), float(self.lineEdit_11.text())]
            ],
            dtype=np.float32
        )
        return r

    @rotation_matrix.setter
    def rotation_matrix(self, v):
        self.lineEdit_3.setText(str(v[0, 0]))
        self.lineEdit_6.setText(str(v[0, 1]))
        self.lineEdit_9.setText(str(v[0, 2]))

        self.lineEdit_4.setText(str(v[1, 0]))
        self.lineEdit_7.setText(str(v[1, 1]))
        self.lineEdit_10.setText(str(v[1, 2]))

        self.lineEdit_5.setText(str(v[2, 0]))
        self.lineEdit_8.setText(str(v[2, 1]))
        self.lineEdit_11.setText(str(v[2, 2]))

    @property
    def translation_vector(self):
        r = np.array([
            float(self.lineEdit_12.text()),
            float(self.lineEdit_13.text()),
            float(self.lineEdit_14.text())
        ], dtype=np.float32)
        return r / 10.0

    @translation_vector.setter
    def translation_vector(
            self,
            v: typing.Tuple[float, float, float]
    ):
        self.lineEdit_12.setText(str(v[0]))
        self.lineEdit_13.setText(str(v[1]))
        self.lineEdit_14.setText(str(v[2]))

    @property
    def trajectory_filename(self) -> str:
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(
            self,
            v: str
    ):
        self.lineEdit.setText(str(v))

    def _empty_icon(self):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        return QtGui.QIcon(pixmap)

    def _stretch_layout(self) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.groupBox.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum
        )
        self.groupBox_2.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum
        )
        for name in (
            "lineEdit_3", "lineEdit_4", "lineEdit_5", "lineEdit_6",
            "lineEdit_7", "lineEdit_8", "lineEdit_9", "lineEdit_10",
            "lineEdit_11", "lineEdit_12", "lineEdit_13", "lineEdit_14",
        ):
            widget = getattr(self, name)
            widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.setRowStretch(1, 0)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(4, 0)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.pushButton_2.setIcon(self._empty_icon())
        self.toolButton.setIcon(self._empty_icon())

    def _append_log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    @chisurf.gui.decorators.init_with_ui(ui_filename="rotate_translate_traj.ui")
    def __init__(self, **kwargs):
        self.trajectory = None
        self.verbose = kwargs.get('verbose', chisurf.core.settings.cs_settings['verbose'])
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_trajectory.triggered.connect(self.onSaveTrajectory)
        self._stretch_layout()
        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Log")
        self._log.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.gridLayout_3.addWidget(self._log, 3, 0, 1, 5)
        self.gridLayout_3.setRowStretch(3, 1)
        self._append_log("Ready")

    def onOpenTrajectory(self, filename=None):
        print("onOpenTrajectory")
        #self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
        filename = chisurf.gui.widgets.get_filename('Open H5-Model file', 'H5-files (*.h5)')
        self.trajectory_filename = filename

    def onSaveTrajectory(self, target_filename=None):
        if not target_filename:
            self._append_log("Save cancelled")
            return

        try:
            self._append_log(f"Saving rotated/translated trajectory: {target_filename}")
            translation_vector = self.translation_vector
            rotation_matrix = self.rotation_matrix
            stride = self.stride

            if self.verbose:
                print("Stride: %s" % stride)
                print("\nRotation Matrix")
                print(rotation_matrix)
                print("\nTranslation vector")
                print(translation_vector)

            first_frame = mdtraj.load_frame(self.trajectory_filename, 0)
            self._append_log(f"Loaded first frame with {first_frame.n_atoms} atoms")
            traj_new = mdtraj.Trajectory(xyz=np.empty((1, first_frame.n_atoms, 3)), topology=first_frame.topology)
            traj_new.save(target_filename)

            chunk_size = 1000
            table = tables.open_file(target_filename, 'a')
            try:
                for i, chunk in enumerate(
                        mdtraj.iterload(
                            self.trajectory_filename,
                            chunk=chunk_size,
                            stride=stride
                        )
                ):
                    xyz = chunk.xyz.copy()
                    rotate(xyz, rotation_matrix)
                    translate(xyz, translation_vector)
                    table.root.xyz.append(xyz)
                    table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))
                    if (i + 1) % 10 == 0:
                        self._append_log(f"Processed {i + 1} chunks")
            finally:
                table.close()
            self._append_log("Save complete")
        except Exception as exc:
            self._append_log(f"Save failed: {exc}")
            raise
