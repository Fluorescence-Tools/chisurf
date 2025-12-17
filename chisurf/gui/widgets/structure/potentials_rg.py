from __future__ import annotations

from qtpy import QtWidgets

import chisurf.structure


class RadiusGyrationWidget(QtWidgets.QWidget):

    name = 'Radius-Gyration'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            parent=None
    ):
        super(RadiusGyrationWidget, self).__init__(parent)
        self.structure = structure
        self.parent = parent

    def getEnergy(self, c=None):
        if c is None:
            c = self.structure
        return c.radius_gyration
