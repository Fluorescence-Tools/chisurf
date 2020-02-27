import mdtraj as md
from qtpy import QtWidgets

import chisurf.decorators
import chisurf.widgets


class SaveTopology(QtWidgets.QWidget):

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    @chisurf.decorators.init_with_ui(ui_filename="save_topology.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionOpen_trajectory.triggered.connect(self.onOpenTrajectory)
        self.actionSave_clash_free_trajectory.triggered.connect(self.onSaveTopology)

    def onSaveTopology(self):
        target_filename = str(QtWidgets.QFileDialog.getSaveFileName(None, 'Save PDB-file', '', 'PDB-files (*.pdb)'))[0]
        filename = self.trajectory_filename
        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        frame_0.save(target_filename)

    def onOpenTrajectory(self, filename=None):
        if filename is None:
            #self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))
            filename = chisurf.widgets.get_filename('Open H5-Model file', 'H5-files (*.h5)')
            self.trajectory_filename = filename