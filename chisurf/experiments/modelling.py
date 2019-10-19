"""

"""
from __future__ import annotations

from qtpy import QtWidgets

import chisurf.settings as mfm
import chisurf.base
import chisurf.fio
import chisurf.fio.widgets
import chisurf.widgets.pdb
from . reader import ExperimentReader


class LoadStructure(
    ExperimentReader,
    QtWidgets.QWidget
):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        """

        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.pdbWidget = chisurf.fio.widgets.PDBLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def read(
            self,
            filename: str = None,
            **kwargs
    ):
        self.pdbWidget.load(filename=filename)
        s = self.pdbWidget.structure
        s.setup = self
        return [s]

    @staticmethod
    def autofitrange(
            data: chisurf.base.Data,
            **kwargs
    ):
        return None


class LoadStructureFolder(
    ExperimentReader,
    QtWidgets.QWidget
):
    """

    """

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
    ) -> experiments.data.ExperimentDataGroup:
        return [self.pdbWidget.trajectory]

    @staticmethod
    def autofitrange(
            data: chisurf.base.Data,
            **kwargs
    ):
        return None, None

