from PyQt4 import QtGui

import mfm
from mfm.experiments import Setup
from mfm.io.widgets import PDBLoad
import mfm.widgets as widgets


class LoadStructure(QtGui.QWidget, Setup):

    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        Setup.__init__(self, *args, **kwargs)

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.pdbWidget = PDBLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def load_data(self, filename=None, **kwargs):
        self.pdbWidget.load(filename=filename)
        s = self.pdbWidget.structure
        s.setup = self
        return [s]

    def autofitrange(self, fit):
        return None, None


class LoadStructureFolder(QtGui.QWidget, Setup):

    name = 'Trajectory'

    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        Setup.__init__(self, *args, **kwargs)
        self.parent = kwargs.get('parent', None)

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.pdbWidget = widgets.PDBFolderLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def load_data(self):
        return [self.pdbWidget.trajectory]

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    def autofitrange(self, fit):
        return None, None

