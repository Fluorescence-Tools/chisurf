from __future__ import annotations


import chisurf.base
import chisurf.decorators
import chisurf.experiments
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.experiments import reader
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
        self.pdbWidget = chisurf.gui.widgets.pdb.PDBFolderLoad(self)
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
    ) -> chisurf.data.ExperimentDataGroup:
        return [self.pdbWidget.trajectory]

    @staticmethod
    def autofitrange(
            data: chisurf.base.Data,
            **kwargs
    ):
        return None, None


class StructureReaderController(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    def get_filename(
            self
    ) -> str:
        return chisurf.gui.widgets.get_filename(
            description='Open PDB-Structure',
            file_type='PDB-file (*.pdb)',
            working_path=None
        )

    @chisurf.gui.decorators.init_with_ui(
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
        chisurf.run(
            "\n".join(
                [
                    "cs.current_setup.compute_internal_coordinates = %s" % compute_internal_coordinates
                ]
            )
        )