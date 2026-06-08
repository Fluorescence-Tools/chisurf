from __future__ import annotations

import chisurf as cs
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.core.models

from chisurf.gui.widgets.models.tcspc.anisotropy import AnisotropyWidget

import chisurf.core.models.tcspc.fret as fret


class DiscreteDistanceWidget(fret.DiscreteDistance, QtWidgets.QWidget):

    # TODO: needs docstring
    def __init__(
            self,
            donors,
            model: cs.core.models.Model = None,
            **kwargs
    ):
        """Initialize the instance."""
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
            model=model,
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.anisotropy)

        try:
            self._install_code_badge()
        except Exception:
            pass

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not cs.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_object_source
            resolver = lambda: resolve_object_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass

    # TODO: needs docstring
    def onAddFRETrate(self):
        """Handle add FRET rate button click."""
        # Append a new discrete FRET-rate component to all fits in the
        # current fit group. The backend FRETrateModel exposes the
        # component group on the 'fret_rates' attribute.
        cs.core.actions.dispatch(
            name="model.add_component",
            payload={"component_name": "fret_rates"},
        )

    # TODO: needs docstring
    def onRemoveFRETrate(self):
        """Handle remove FRET rate button click."""
        # Remove the last discrete FRET-rate component from all fits in the
        # current fit group.
        cs.core.actions.dispatch(
            name="model.remove_component",
            payload={"component_name": "fret_rates"},
        )

    # TODO: needs docstring
    def append(self, *args, **kwargs):
        """Add a new component."""
        super().append(50., 1.0)

        gb = QtWidgets.QGroupBox()
        n_rates = len(self)
        gb.setTitle(f'k{n_rates}')

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._distances[-1],
            layout=layout
        )
        cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._amplitudes[-1],
            layout=layout
        )

        gb.setLayout(layout)
        row = (n_rates - 1) // 2 + 1
        col = (n_rates - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    # TODO: needs docstring
    def pop(self):
        """Remove the last component."""
        super().pop()
        self._gb.pop().close()
