from __future__ import annotations

import sys

from qtpy import QtCore, QtWidgets
import qdarkstyle

import mdtraj

import chisurf.widgets
import mfm
import mfm.decorators
from chisurf.structure.potential import potentialDict
from chisurf.structure.trajectory import TrajectoryFile, Universe


class PotentialEnergyWidget(QtWidgets.QWidget):

    name = "Potential-Energy calculator"

    @mfm.decorators.init_with_ui(ui_filename="calculate_potential.ui")
    def __init__(
            self,
            verbose: bool = False,
            structure: mfm.structure.structure.Structure = None,
            *args,
            **kwargs
    ):
        self._trajectory_file = ''
        self.potential_weight = 1.0
        self.energies = list()

        self.verbose = verbose
        self.structure = structure
        self.universe = Universe()

        self.actionOpen_trajectory.triggered.connect(self.onLoadTrajectory)
        self.actionProcess_trajectory.triggered.connect(self.onProcessTrajectory)
        self.actionAdd_potential.triggered.connect(self.onAddPotential)
        self.tableWidget.cellDoubleClicked [int, int].connect(self.onRemovePotential)
        self.actionCurrent_potential_changed.triggered.connect(self.onSelectedPotentialChanged)

        self.comboBox_2.addItems(list(potentialDict))

    @property
    def potential_number(self) -> int:
        return int(self.comboBox_2.currentIndex())

    @property
    def potential_name(self) -> str:
        return list(potentialDict)[self.potential_number]

    def onProcessTrajectory(self):
        print("onProcessTrajectory")
        energy_file = chisurf.widgets.save_file(
            'Save energies',
            'CSV-name file (*.txt)'
        )

        s = 'FrameNbr\t'
        for p in self.universe.potentials:
            s += '%s\t' % p.name
        s += '\n'
        mfm.fio.zipped.open_maybe_zipped(
            filename=energy_file,
            mode='w'
        ).write(s)

        self.structure = TrajectoryFile(
            mdtraj.load_frame(
                self.trajectory_file, 0
            ),
            make_coarse=True
        )[0]
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

    def onSelectedPotentialChanged(self) -> None:
        layout = self.verticalLayout_2
        chisurf.widgets.hide_items_in_layout(layout)

        self.potential = potentialDict[self.potential_name](
            structure=self.structure,
            parent=self
        )

        layout.addWidget(self.potential)

    def onAddPotential(self) -> None:
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

    def onRemovePotential(self) -> None:
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
    def stride(self) -> int:
        return int(self.spinBox.value())

    def onLoadTrajectory(self) -> None:
        filename = chisurf.widgets.get_filename(
            'Open Trajectory-File',
            'H5-Trajectory-Files (*.h5)'
        )
        self.trajectory_file = filename
        self.lineEdit.setText(self.trajectory_file)

    @property
    def energy(self) -> float:
        return self.universe.getEnergy(self.structure)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PotentialEnergyWidget()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
