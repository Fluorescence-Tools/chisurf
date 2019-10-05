from __future__ import annotations

import sys

from qtpy import QtWidgets
import qdarkstyle

import mdtraj as md
import numpy as np
import tables

import mfm
import mfm.decorators
import mfm.widgets
from mfm.tools.modelling.trajectory import below_min_distance


class RemoveClashedFrames(QtWidgets.QWidget):

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def atom_list(self):
        txt = str(self.plainTextEdit.toPlainText())
        return txt
        #atom_list = np.fromstring(txt, dtype=np.int32, sep=",")
        #return atom_list

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    @property
    def min_distance(self):
        return float(self.doubleSpinBox.value()) / 10.0

    def onRemoveClashes(self):
        print("onRemoveClashes")
        target_filename = mfm.widgets.save_file('H5-Trajectory file', 'H5-File (*.h5)')
        # target_filename = 'clash_dimer.h5'
        filename = self.trajectory_filename
        stride = self.stride
        min_distance = self.min_distance

        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        target_traj = md.Trajectory(xyz=np.empty((0, frame_0.n_atoms, 3)), topology=frame_0.topology)
        #atom_indices = np.array(self.atom_list)
        atom_selection = self.atom_list
        atom_list = target_traj.top.select(atom_selection)
        target_traj.save(target_filename)

        chunk_size = 1000
        for i, chunk in enumerate(md.iterload(filename, chunk=chunk_size, stride=stride)):
            xyz = chunk.xyz.copy()
            frames_below = below_min_distance(xyz, min_distance, atom_list=atom_list)
            selection = np.where(frames_below < 1)[0]
            xyz_clash_free = np.take(xyz, selection, axis=0)
            with tables.open_file(target_filename, 'a') as table:
                table.root.coordinates.append(xyz_clash_free)
                times = np.arange(table.root.time.shape[0],
                                  table.root.time.shape[0] + xyz_clash_free.shape[0], dtype=np.float32)
                table.root.time.append(times)

    def onOpenTrajectory(self, filename=None):
        print("onOpenTrajectory 1")
        if filename is None:
            filename = mfm.widgets.get_filename('Open H5-Model file', 'H5-files (*.h5)')
            self.trajectory_filename = filename

    @mfm.decorators.init_with_ui(ui_filename="remove_clashes.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_clash_free_trajectory.triggered.connect(self.onRemoveClashes)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = RemoveClashedFrames()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
