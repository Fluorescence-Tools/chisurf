from PyQt5 import QtCore, QtGui, QtWidgets, uic
import mdtraj
import mfm
from mfm.structure.potential import potentials
from mfm.structure.trajectory import TrajectoryFile, Universe


class PotentialEnergyWidget(QtWidgets.QWidget):

    name = "Potential-Energy calculator"

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/tools//calculate_potential.ui', self)
        self._trajectory_file = ''
        self.potential_weight = 1.0
        self.energies = []

        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.structure = kwargs.get('structure', None)
        self.universe = Universe()

        self.actionOpen_trajectory.triggered.connect(self.onLoadTrajectory)
        self.actionProcess_trajectory.triggered.connect(self.onProcessTrajectory)
        self.actionAdd_potential.triggered.connect(self.onAddPotential)
        self.tableWidget.cellDoubleClicked [int, int].connect(self.onRemovePotential)
        self.actionCurrent_potential_changed.triggered.connect(self.onSelectedPotentialChanged)

        self.comboBox_2.addItems(list(potentials.potentialDict.keys()))

    @property
    def potential_number(self):
        return int(self.comboBox_2.currentIndex())

    @property
    def potential_name(self):
        return list(potentials.potentialDict.keys())[self.potential_number]

    def onProcessTrajectory(self):
        print("onProcessTrajectory")
        energy_file = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save energies', '.txt', 'CSV-text file (*.txt)'))[0]

        s = 'FrameNbr\t'
        for p in self.universe.potentials:
            s += '%s\t' % p.name
        s += '\n'
        open(energy_file, 'w').write(s)

        self.structure = TrajectoryFile(mdtraj.load_frame(self.trajectory_file, 0), make_coarse=True)[0]
        i = 0
        for chunk in mdtraj.iterload(self.trajectory_file):
            for frame in chunk:
                self.structure.xyz = frame.xyz * 10.0
                self.structure.update_dist()
                s = '%i\t' % (i * self.stride + 1)
                for e in self.universe.getEnergies(self.structure):
                    s += '%.3f\t' % e
                print(s)
                s += '\n'
                i += 1
                open(energy_file, 'a').write(s)

    def onSelectedPotentialChanged(self):
        layout = self.verticalLayout_2
        for i in range(layout.count()):
            layout.itemAt(i).widget().close()
        self.potential = potentials.potentialDict[self.potential_name](structure=self.structure, parent=self)
        layout.addWidget(self.potential)

    def onAddPotential(self):
        print("onAddPotential")
        self.universe.addPotential(self.potential, self.potential_weight)
        # update table
        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)
        tmp = QtWidgets.QTableWidgetItem(str(self.potential_name))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 0, tmp)
        tmp = QtWidgets.QTableWidgetItem(str(self.potential_weight))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 1, tmp)
        table.resizeRowsToContents()

    def onRemovePotential(self):
        print("onRemovePotential")
        table = self.tableWidget
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)
            self.universe.removePotential(idx)

    @property
    def stride(self):
        return int(self.spinBox.value())

    def onLoadTrajectory(self):
        #self.trajectory_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Trajectory-File', '.h5', 'H5-Trajectory-Files (*.h5)'))
        filename = mfm.widgets.get_filename('Open Trajectory-File', 'H5-Trajectory-Files (*.h5)')
        self.trajectory_file = filename
        self.lineEdit.setText(self.trajectory_file)

    @property
    def energy(self):
        return self.universe.getEnergy(self.structure)

