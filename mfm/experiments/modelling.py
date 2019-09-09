from __future__ import annotations

from qtpy import  QtWidgets

import mfm
import mfm.structure.widgets
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
        self.pdbWidget = mfm.structure.widgets.PDBFolderLoad(self)
        self.layout.addWidget(self.pdbWidget)

    @staticmethod
    def read(
            self,
            data: mfm.experiments.data.Data,
            **kwargs
    ):
        return [self.pdbWidget.trajectory]

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    @staticmethod
    def autofitrange(
            data: mfm.experiments.data.Data,
            **kwargs
    ):
        return None, None

