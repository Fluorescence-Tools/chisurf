from __future__ import annotations

from qtpy import QtWidgets

import mfm
import mfm.widgets.pdb.pdb
from mfm.experiments.reader import ExperimentReader
from mfm.io.widgets import PDBLoad


class LoadStructure(ExperimentReader, QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(LoadStructure, self).__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.pdbWidget = PDBLoad(self)
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
            data: mfm.experiments.data.Data,
            **kwargs
    ):
        return None, None


class LoadStructureFolder(ExperimentReader, QtWidgets.QWidget):

    name = 'Trajectory'

    def __init__(self, *args, **kwargs):
        super(LoadStructureFolder, self).__init__(*args, **kwargs)
        self.parent = kwargs.get('parent', None)
        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.pdbWidget = mfm.widgets.pdb.PDBFolderLoad(self)
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
    ) -> mfm.experiments.data.ExperimentDataGroup:
        return [self.pdbWidget.trajectory]

    @staticmethod
    def autofitrange(
            data: mfm.experiments.data.Data,
            **kwargs
    ):
        return None, None

