"""

"""
from __future__ import annotations
import typing

from qtpy import QtWidgets

import chisurf.base
import chisurf.decorators
import chisurf.fio
import chisurf.structure
import chisurf.widgets
import chisurf.widgets.fio
import chisurf.widgets.pdb
from . import reader


class StructureReader(
    reader.ExperimentReader
):

    def __init__(
            self,
            compute_internal_coordinates: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.compute_internal_coordinates = compute_internal_coordinates

    @staticmethod
    def autofitrange(
            data: chisurf.base.Data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        return 0, 0

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.experiments.data.ExperimentDataGroup:
        structure = chisurf.structure.Structure(
            p_object=filename,
            make_coarse=self.compute_internal_coordinates
        )
        data_group = chisurf.experiments.data.ExperimentDataGroup(
            seq=[structure]
        )
        data_group.data_reader = self
        return data_group


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
        self.pdbWidget = chisurf.widgets.pdb.PDBFolderLoad(self)
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
    ) -> chisurf.experiments.data.ExperimentDataGroup:
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
        return chisurf.widgets.get_filename(
            description='Open PDB-Structure',
            file_type='PDB-file (*.pdb)',
            working_path=None
        )

    @chisurf.decorators.init_with_ui(
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
