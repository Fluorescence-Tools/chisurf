from PyQt5 import QtGui, QtWidgets

import mfm
import mfm.structure.widgets
from mfm.experiments import Reader
from mfm.io.widgets import PDBLoad
import mfm.widgets as widgets


class LoadStructure(QtWidgets.QWidget, Reader):

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Reader.__init__(self, *args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        #self.layout.setMargin(0)
        #self.layout.setSpacing(0)
        self.pdbWidget = PDBLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def read(self, filename=None, **kwargs):
        self.pdbWidget.load(filename=filename)
        s = self.pdbWidget.structure
        s.setup = self
        return [s]

    def autofitrange(self, fit):
        return None, None


class LoadStructureFolder(QtWidgets.QWidget, Reader):

    name = 'Trajectory'

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Reader.__init__(self, *args, **kwargs)
        self.parent = kwargs.get('parent', None)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.pdbWidget = mfm.structure.widgets.PDBFolderLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def read(self):
        return [self.pdbWidget.trajectory]

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    def autofitrange(self, fit):
        return None, None

