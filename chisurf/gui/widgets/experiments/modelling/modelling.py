from __future__ import annotations
import chisurf as cs


import chisurf.core.base
import chisurf.core.decorators
import chisurf.core.experiments
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.core.experiments.core import reader
from qtpy import QtWidgets


class LoadStructureFolder(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    name = 'Trajectory'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = kwargs.get('parent', None)
        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.pdbWidget = cs.gui.widgets.pdb.PDBFolderLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def read(
            self,
            name: str = None,
            **kwargs
    ):
        pass

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    def get_data(
            self,
            **kwargs
    ) -> cs.core.data.ExperimentDataGroup:
        return [self.pdbWidget.trajectory]

    @staticmethod
    def autofitrange(data: cs.core.base.Data, **kwargs):
        return None, None


class StructureReaderController(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    def get_filename(self) -> str:
        return cs.gui.widgets.get_filename(
            description='Open PDB-Structure',
            file_type='PDB-file (*.pdb)',
            working_path=None
        )

    @cs.gui.decorators.init_with_ui(
        ui_filename="proteinMCLoad.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)

    # def load(self, filename=None):
    #     self.lineEdit.setText(str(self.structure.n_atoms))
    #     self.lineEdit_2.setText(str(self.structure.n_residues))

    def onParametersChanged(self):
        compute_internal_coordinates = bool(self.checkBox.isChecked())
        cs.run(
            "\n".join(
                [
                    "cs.current_setup.compute_internal_coordinates = %s" % compute_internal_coordinates
                ]
            )
        )

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        try:
            setup = cs.cs.current_setup
        except Exception:
            return

        # compute_internal_coordinates
        try:
            cic = getattr(setup, 'compute_internal_coordinates', False)
            self.checkBox.blockSignals(True)
            self.checkBox.setChecked(bool(cic))
            self.checkBox.blockSignals(False)
        except Exception:
            pass
