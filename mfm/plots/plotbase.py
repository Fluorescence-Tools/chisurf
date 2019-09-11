from __future__ import annotations

from qtpy import QtWidgets

import mfm.fitting
from mfm.base import Base


class Plot(QtWidgets.QWidget, Base):

    def __init__(
            self,
            fit: mfm.fitting.fit,
            parent=None,
            **kwargs
    ):
        super(Plot, self).__init__()
        self.parent = parent
        self.fit = fit
        self.pltControl = QtWidgets.QWidget()
        self.widgets = list()

    def update_widget(
            self
    ) -> None:
        for w in self.widgets:
            w.update()

    def update_all(
            self,
            *args,
            **kwargs
    ) -> None:
        pass

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.pltControl, QtWidgets.QWidget):
            self.pltControl.close()
