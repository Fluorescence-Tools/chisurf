from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.models

from chisurf.models.tcspc.widgets.anisotropy import AnisotropyWidget

import chisurf.models.tcspc.fret as fret


class DiscreteDistanceWidget(fret.DiscreteDistance, QtWidgets.QWidget):

    def __init__(
            self,
            donors,
            model: chisurf.models.Model = None,
            **kwargs
    ):
        super().__init__(
            donors=donors,
            model=model,
            **kwargs
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()

        l = QtWidgets.QHBoxLayout()
        addFRETrate = QtWidgets.QPushButton()
        addFRETrate.setText("add")
        l.addWidget(addFRETrate)

        removeFRETrate = QtWidgets.QPushButton()
        removeFRETrate.setText("del")
        l.addWidget(removeFRETrate)
        self.lh.addLayout(l)

        self.lh.addLayout(self.grid_layout)

        addFRETrate.clicked.connect(self.onAddFRETrate)
        removeFRETrate.clicked.connect(self.onRemoveFRETrate)

        # add some initial distance
        self.append(1.0, 50.0, False)

        s = kwargs.pop('short', None)
        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.anisotropy)

    def onAddFRETrate(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.append()\n"\
            f"   f.model.update()"
        )

    def onRemoveFRETrate(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.pop()\n"\
            f"   f.model.update()"
        )

    def append(self, *args, **kwargs):
        super().append(50., 1.0)

        gb = QtWidgets.QGroupBox()
        n_rates = len(self)
        gb.setTitle(f'k{n_rates}')

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._distances[-1],
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._amplitudes[-1],
            layout=layout
        )

        gb.setLayout(layout)
        row = (n_rates - 1) // 2 + 1
        col = (n_rates - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def pop(self):
        super().pop()
        self._gb.pop().close()