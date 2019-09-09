# coding=utf-8
import os

from PyQt5 import QtWidgets, uic

from mfm.structure.structure import Structure
from mfm.structure.trajectory import TrajectoryFile
from mfm.widgets.pdb import LoadThread


class PDBFolderLoad(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi("mfm/ui/proteinFolderLoad.ui", self)
        self.pushButton_12.clicked.connect(self.onLoadStructure)
        self.updatePBar(0)
        self.load_thread = LoadThread()
        self.load_thread.partDone.connect(self.updatePBar)
        self.load_thread.procDone.connect(self.fin)
        self.trajectory = TrajectoryFile()

    def fin(self):
        print("Loading of structures finished")
        self.lineEdit.setText(str(self.nAtoms))
        self.lineEdit_2.setText(str(self.nResidues))

    def updatePBar(self, val):
        self.progressBar.setValue(val)

    def onLoadStructure(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.folder = directory
        filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
        filenames.sort()

        pdb_filenames = []
        for i, filename in enumerate(filenames):
            extension = os.path.splitext(filename)[1][1:]
            if filename.lower().endswith('.pdb') or extension.isdigit():
                pdb_filenames.append(filename)

        self.n_files = len(pdb_filenames)

        self.load_thread.read = Structure
        self.load_thread.read_parameter = [self.calc_internal, self.verbose]
        self.load_thread.append_parameter = [self.calc_rmsd]
        self.trajectory = TrajectoryFile(use_objects=self.use_objects, calc_internal=self.calc_internal,
                                     verbose=self.verbose)
        self.load_thread.filenames = pdb_filenames
        self.load_thread.target = self.trajectory
        self.load_thread.start()

    @property
    def calc_rmsd(self):
        return self.checkBox_4.isChecked()

    @property
    def n_files(self):
        return int(self.lineEdit_3.text())

    @n_files.setter
    def n_files(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def verbose(self):
        return bool(self.checkBox_3.isChecked())

    @property
    def nAtoms(self):
        return self.trajectory[0].n_atoms

    @property
    def nResidues(self):
        return self.trajectory[0].n_residues

    @property
    def folder(self):
        return str(self.lineEdit_7.text())

    @folder.setter
    def folder(self, v):
        self.lineEdit_7.setText(v)

    @property
    def use_objects(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def calc_internal(self):
        return self.checkBox.isChecked()
