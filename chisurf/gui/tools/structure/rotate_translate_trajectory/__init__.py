from __future__ import annotations
import typing

import numpy as np
import tables
from qtpy import QtWidgets
import mdtraj

import chisurf.decorators
import chisurf.gui.widgets
from chisurf.structure import translate, rotate


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

    @chisurf.decorators.init_with_ui(ui_filename="rotate_translate_traj.ui")
    def __init__(self, **kwargs):
        self.trajectory = None
        self.verbose = kwargs.get('verbose', chisurf.verbose)
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_trajectory.triggered.connect(self.onSaveTrajectory)

    def onOpenTrajectory(self, filename=None):
        print("onOpenTrajectory")
        #self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
        filename = chisurf.gui.widgets.get_filename('Open H5-Model file', 'H5-files (*.h5)')
        self.trajectory_filename = filename

    def onSaveTrajectory(self, target_filename=None):
        if target_filename is None:
            target_filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))[0]

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
        traj_new = mdtraj.Trajectory(xyz=np.empty((1, first_frame.n_atoms, 3)), topology=first_frame.topology)
        traj_new.save(target_filename)

        chunk_size = 1000
        table = tables.open_file(target_filename, 'a')
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
        table.close()
