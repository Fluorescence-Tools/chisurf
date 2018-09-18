from PyQt4 import QtGui

from mfm import Base


class Plot(QtGui.QWidget, Base):

    def __init__(self, fit, parent=None):
        Base.__init__(self)
        QtGui.QWidget.__init__(self, parent)
        self.parent = parent
        self.fit = fit
        self.pltControl = QtGui.QWidget()
        self.widgets = []

    def update_widget(self):
        for w in self.widgets:
            w.update()

    def update_all(self, *args, **kwargs):
        pass

    def close(self):
        QtGui.QWidget.close(self)
        if isinstance(self.pltControl, QtGui.QWidget):
            self.pltControl.close()