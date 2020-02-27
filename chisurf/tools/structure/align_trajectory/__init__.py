import mdtraj as md
import numpy as np
import tables
from qtpy import QtWidgets

import chisurf.decorators
import chisurf.widgets


class AlignTrajectoryWidget(QtWidgets.QWidget):

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def atom_list(self):
        txt = str(self.plainTextEdit.toPlainText())
        atom_list = np.fromstring(txt, dtype=np.int32, sep=",")
        return atom_list

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    @chisurf.decorators.init_with_ui(ui_filename="align_trajectory.ui")
    def __init__(self, **kwargs):
        self.trajectory = None
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_aligned_trajectory.triggered.connect(self.onSaveTrajectory)

    def onOpenTrajectory(
            self,
            filename: str = None
    ):
        #self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
        if filename is None:
            filename = chisurf.widgets.get_filename(
                'Open H5-Model file', 'H5-files (*.h5)'
            )
        self.trajectory_filename = filename

    def onSaveTrajectory(
            self,
            target_filename: str = None
    ):
        if target_filename is None:
            target_filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))[0]
        filename = self.trajectory_filename
        atom_indices = self.atom_list
        stride = self.stride

        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        target_traj = md.Trajectory(xyz=np.empty((0, frame_0.n_atoms, 3)), topology=frame_0.topology)
        target_traj.save(target_filename)

        chunk_size = 1000
        table = tables.open_file(target_filename, 'a')
        for i, chunk in enumerate(md.iterload(filename, chunk=chunk_size, stride=stride)):
            chunk = chunk.superpose(frame_0, frame=0, atom_indices=atom_indices)
            xyz = chunk.coordinates.copy()
            table.root.coordinates.append(xyz)
            table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))
        table.close()