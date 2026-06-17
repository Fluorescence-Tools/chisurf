from __future__ import annotations

import sys

from qtpy import QtCore, QtWidgets

import mdtraj

import chisurf.core.fio as io
import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.core.decorators
import chisurf.core.structure.potential
import chisurf.core.structure.trajectory
import chisurf.gui.widgets.structure

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



@persist_plugin_state("potential_energy")
class PotentialEnergyWidget(QtWidgets.QWidget):

    name = "Potential-Energy calculator"

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="calculate_potential.ui"
    )
    def __init__(
            self,
            verbose: bool = False,
            structure: chisurf.core.structure.Structure = None
    ):
        self._trajectory_file = ''
        self.potential_weight = 1.0
        self.energies = list()

        self.verbose = verbose
        self.structure = structure
        self.universe = chisurf.core.structure.Universe()

        self.actionOpen_trajectory.triggered.connect(self.onLoadTrajectory)
        self.actionProcess_trajectory.triggered.connect(self.onProcessTrajectory)
        self.actionAdd_potential.triggered.connect(self.onAddPotential)
        self.tableWidget.cellDoubleClicked [int, int].connect(self.onRemovePotential)
        self.actionCurrent_potential_changed.triggered.connect(self.onSelectedPotentialChanged)

        self.comboBox_2.addItems(
            list(chisurf.gui.widgets.structure.potentialDict)
        )
        self._stretch_potential_layout()

    def _stretch_potential_layout(self) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.groupBox_3.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.groupBox_8.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.groupBox_8.setMaximumHeight(160)
        self.tableWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.tableWidget.setMinimumHeight(120)
        self.verticalLayout_4.setStretch(0, 0)
        self.verticalLayout_2.setStretch(0, 0)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout_3.setRowStretch(2, 0)
        self.gridLayout_3.setRowStretch(4, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(3, 1)

    @property
    def potential_number(self) -> int:
        return int(self.comboBox_2.currentIndex())

    @property
    def potential_name(self) -> str:
        return list(
            chisurf.gui.widgets.structure.potentialDict
        )[self.potential_number]

    def onProcessTrajectory(self):
        print("onProcessTrajectory")
        energy_file = chisurf.gui.widgets.save_file(
            description='Save energies',
            file_type='CSV-name file (*.txt)'
        )

        s = 'FrameNbr\t'
        for p in self.universe.potentials:
            s += '%s\t' % p.name
        s += '\n'
        io.zipped.open_maybe_zipped(
            filename=energy_file,
            mode='w'
        ).write(s)

        self.structure = chisurf.core.structure.TrajectoryFile(
            mdtraj.load_frame(
                self.trajectory_file, 0
            )
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
        chisurf.gui.widgets.hide_items_in_layout(layout)
        self.potential = chisurf.gui.widgets.structure.potentialDict[self.potential_name](
            structure=self.structure,
            parent=self
        )

        layout.addWidget(self.potential)
        self.potential.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.potential.setMaximumHeight(160)

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
        filename = chisurf.gui.widgets.get_filename(
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
    win.show()
    sys.exit(app.exec_())
