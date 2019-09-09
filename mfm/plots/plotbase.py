from qtpy import  QtWidgets

from mfm.base import Base


class Plot(QtWidgets.QWidget, Base):

    def __init__(self, fit, parent=None):
        super(Plot, self).__init__()
        self.parent = parent
        self.fit = fit
        self.pltControl = QtWidgets.QWidget()
        self.widgets = []

    def update_widget(self):
        for w in self.widgets:
            w.update()

    def update_all(self, *args, **kwargs):
        pass

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.pltControl, QtWidgets.QWidget):
            self.pltControl.close()
