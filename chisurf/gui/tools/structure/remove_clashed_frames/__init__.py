from __future__ import annotations

import sys

import numba as nb
from qtpy import QtWidgets

import mdtraj
import numpy as np
import tables

import chisurf.decorators
import chisurf.gui.widgets


@nb.jit
def below_min_distance(
        xyz: np.ndarray,
        min_distance: float,
        atom_list: np.ndarray = np.empty(0, dtype=np.int32)
) -> np.ndarray:
    """Takes the xyz-coordinates (frame, atom, xyz) of a trajectory as an argument an returns a vector of booleans
    of length of the number of frames. The bool is False if the frame contains a atomic distance smaller than the
    min distance.

    :param xyz: numpy array
        The coordinates (frame fit_index, atom fit_index, coord)

    :param min_distance: float
        Minimum distance if a distance

    :return: numpy-array
        If a atom-atom distance within a frame is smaller than min_distance the value within the array is True otherwise
        it is False.

    """

    n_frames = xyz.shape[0]
    re = np.zeros(n_frames, dtype=np.uint8)

    atoms = np.arange(xyz.shape[1]) if atom_list.shape[0] == 0 else atom_list
    n_atoms = atoms.shape[0]
    min_distance2 = min_distance**2.0

    for i_frame in range(n_frames):

        for i in range(n_atoms):
            i_atom = atoms[i]
            x1 = xyz[i_frame, i_atom, 0]
            y1 = xyz[i_frame, i_atom, 1]
            z1 = xyz[i_frame, i_atom, 2]

            for j in range(i + 1, n_atoms):
                j_atom = atoms[j]

                x2 = xyz[i_frame, j_atom, 0]
                y2 = xyz[i_frame, j_atom, 1]
                z2 = xyz[i_frame, j_atom, 2]

                dx = (x1-x2)**2
                dy = (y1-y2)**2
                dz = (z1-z2)**2

                if dx + dy + dz < min_distance2:
                    re[i_frame] += 1
                    break

            if re[i_frame] > 0:
                break
    return re


class RemoveClashedFrames(
    QtWidgets.QWidget
):

    @property
    def stride(self) -> int:
        return int(self.spinBox.value())

    @property
    def atom_list(self) -> str:
        txt = str(self.plainTextEdit.toPlainText())
        return txt
        #atom_list = np.fromstring(txt, dtype=np.int32, sep=",")
        #return atom_list

    @property
    def trajectory_filename(self) -> str:
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v: str):
        self.lineEdit.setText(str(v))

    @property
    def min_distance(self) -> float:
        return float(self.doubleSpinBox.value()) / 10.0

    def onRemoveClashes(
            self,
            target_filename: str = None
    ):
        if target_filename is None:
            target_filename = chisurf.gui.widgets.save_file(
                'H5-Trajectory file', 'H5-File (*.h5)'
            )
        # target_filename = 'clash_dimer.h5'
        filename = self.trajectory_filename
        stride = self.stride
        min_distance = self.min_distance

        # Make empty trajectory
        frame_0 = mdtraj.load_frame(filename, 0)
        target_traj = mdtraj.Trajectory(
            xyz=np.empty((0, frame_0.n_atoms, 3)), topology=frame_0.topology
        )
        #atom_indices = np.array(self.atom_list)
        atom_selection = self.atom_list
        atom_list = target_traj.top.select(atom_selection)
        target_traj.save(target_filename)

        chunk_size = 1000
        for i, chunk in enumerate(
                mdtraj.iterload(
                    filename,
                    chunk=chunk_size,
                    stride=stride
                )
        ):
            xyz = chunk.xyz.copy()
            frames_below = below_min_distance(
                xyz=xyz,
                min_distance=min_distance,
                atom_list=atom_list
            )
            selection = np.where(frames_below < 1)[0]
            xyz_clash_free = np.take(xyz, selection, axis=0)
            with tables.open_file(target_filename, 'a') as table:
                table.root.coordinates.append(xyz_clash_free)
                times = np.arange(table.root.time.shape[0],
                                  table.root.time.shape[0] + xyz_clash_free.shape[0], dtype=np.float32)
                table.root.time.append(times)

    def onOpenTrajectory(
            self,
            filename: str = None
    ):
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                'Open H5-Model file', 'H5-files (*.h5)'
            )
            self.trajectory_filename = filename

    @chisurf.decorators.init_with_ui(
        ui_filename="remove_clashes.ui"
    )
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
    win.show()
    sys.exit(app.exec_())
