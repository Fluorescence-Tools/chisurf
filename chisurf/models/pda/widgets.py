from __future__ import annotations
import typing
from typing import TYPE_CHECKING

import numpy as np

import chisurf
import chisurf.settings
import chisurf.gui.widgets.fitting
import chisurf.plots
from chisurf.fluorescence.general import distance_to_fret_efficiency
from chisurf.math.functions import distributions as distfuncs
import chisurf.models.tcspc.fret as tcspc_fret
from chisurf.models.tcspc.fret import rda_axis
from chisurf.settings.settings_utils import set_fret_rda_axis, build_fret_rda_axis

from chisurf.models.model import ModelWidget
from chisurf.gui import QtWidgets, QtGui, QtCore
from chisurf.models.pda.nusiance import Background, PdaFretNuisance, PdaPhotonRange
from chisurf.models.pda.simple import ProbCh0, PdaSimpleModel
from chisurf.models.pda.pdagauss import PdaGaussianDistances, PdaGaussianDistanceModel
from chisurf.models.pda.anisotropy import (
    PdaAnisotropyModel,
    PdaAnisotropyNuisance,
    PdaAnisotropySpecies,
)

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit


class BackgroundWidget(QtWidgets.QGroupBox, Background):

    def __init__(
            self,
            hide_generic: bool = False,
            *args,
            **kwargs
    ):
        """
        Initialize the BackgroundWidget.

        Parameters
        ----------
        hide_generic : bool, optional
            Whether to hide the generic parameters, by default False.
        *args
            Arguments passed to the super-class initialization method.
        **kwargs
            Keyword arguments passed to the super-class initialization method.

        Notes
        -----
        This method ensures the Background FittingParameterGroup is properly initialized (creates _bg0/_bg1),
        hides the generic parameters if specified, and sets up the layout for the generic parameters.
        """
        super().__init__(*args, **kwargs)
        # Ensure the Background FittingParameterGroup is properly initialized (creates _bg0/_bg1)
        try:
            Background.__init__(self, **kwargs)
        except Exception:
            # If already initialized or kwargs not applicable, continue
            pass
        if hide_generic:
            self.hide()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setTitle("Generic")

        # Generic parameters
        self._bg0_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bg0,
            label_text='Bg0',
        )
        self._bg1_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bg1,
            label_text='Bg1'
        )

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._bg0_widget, 1, 0)
        layout.addWidget(self._bg1_widget, 1, 1)
        self.layout.addLayout(layout)

    def update(self, *__args):
        # Call the group-box update for standard behavior
        QtWidgets.QGroupBox.update(self, *__args)
        # Synchronize UI widgets with underlying parameters
        try:
            # Preferred: use controller finalize to sync all UI elements (value, bounds, link state, etc.)
            self._bg0_widget.finalize()
            self._bg1_widget.finalize()
        except Exception:
            # Fallback: at least sync the numeric values
            try:
                self._bg0_widget.setValue(self.bg0)
                self._bg1_widget.setValue(self.bg1)
            except Exception:
                pass


class PdaPhotonRangeWidget(QtWidgets.QGroupBox):

    def __init__(
            self,
            nuisance,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nuisance = nuisance
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setTitle("Photon-number range")

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._nPh_min_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self.nuisance._nPh_min,
            label_text='nPh_min',
        )
        self._nPh_max_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self.nuisance._nPh_max,
            label_text='nPh_max',
        )

        layout.addWidget(self._nPh_min_widget, 0, 0)
        layout.addWidget(self._nPh_max_widget, 0, 1)

        self.layout.addLayout(layout)

    def update(self, *__args):
        QtWidgets.QGroupBox.update(self, *__args)
        try:
            self._nPh_min_widget.finalize()
            self._nPh_max_widget.finalize()
        except Exception:
            try:
                self._nPh_min_widget.setValue(self.nuisance.nPh_min)
                self._nPh_max_widget.setValue(self.nuisance.nPh_max)
            except Exception:
                pass


class PdaFretNuisanceWidget(QtWidgets.QGroupBox, PdaFretNuisance):

    def __init__(
            self,
            hide_generic: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        try:
            PdaFretNuisance.__init__(self, **kwargs)
        except Exception:
            # If already initialized or kwargs not applicable, continue
            pass
        if hide_generic:
            self.hide()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setTitle("PDA FRET nuisance")

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._alpha_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._alpha,
            label_text='alpha',
        )
        self._bgG_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bgG,
            label_text='BG',
        )
        self._bgR_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bgR,
            label_text='BR',
        )
        self._gG_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gG,
            label_text='gG',
        )
        self._gR_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gR,
            label_text='gR',
        )
        self._QYD_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._QYD,
            label_text='QYD',
        )
        self._QYA_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._QYA,
            label_text='QYA',
        )

        layout.addWidget(self._alpha_widget, 0, 0, 1, 2)
        layout.addWidget(self._bgG_widget, 1, 0)
        layout.addWidget(self._bgR_widget, 1, 1)
        layout.addWidget(self._gG_widget, 2, 0)
        layout.addWidget(self._gR_widget, 2, 1)
        layout.addWidget(self._QYD_widget, 3, 0)
        layout.addWidget(self._QYA_widget, 3, 1)

        self.layout.addLayout(layout)

    def update(self, *__args):
        # Call the group-box update for standard behavior
        QtWidgets.QGroupBox.update(self, *__args)
        # Synchronize UI widgets with underlying parameters
        try:
            self._alpha_widget.finalize()
            self._bgG_widget.finalize()
            self._bgR_widget.finalize()
            self._gG_widget.finalize()
            self._gR_widget.finalize()
            self._QYD_widget.finalize()
            self._QYA_widget.finalize()
        except Exception:
            # Fallback: at least sync the numeric values
            try:
                self._alpha_widget.setValue(self.alpha)
                self._bgG_widget.setValue(self.BG)
                self._bgR_widget.setValue(self.BR)
                self._gG_widget.setValue(self.gG)
                self._gR_widget.setValue(self.gR)
                self._QYD_widget.setValue(self.QYD)
                self._QYA_widget.setValue(self.QYA)
            except Exception:
                pass



class ProbCh0Widget(ProbCh0, QtWidgets.QWidget):

    def update(self, *__args):
        ProbCh0.update(self)
        QtWidgets.QWidget.update(self, *__args)
        # Sync amplitude widgets
        for w, v in zip(self._amp_widgets, self.amplitudes):
            w.setValue(v)
        # Sync p(ch0) widgets
        for w, v in zip(self._pch0_widgets, self.pch0):
            w.setValue(v)

    @property
    def parameter_widgets(self):
        return self._amp_widgets + self._pch0_widgets

    def read_values(self, target):

        def linkcall():
            fit_idx = self._amp_widgets[0].fitting_parameter.fit_idx
            for key in self.parameter_dict:
                p = target.parameters_all_dict[key]
                chisurf.run(f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{key}'].value = {p.value}")
                chisurf.run(f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{key}'].controller.finalize()")
            chisurf.run("cs.current_fit.update()")

        return linkcall

    def read_menu(self):
        menu = self.readFrom_menu
        menu.clear()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, self.__class__):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.read_values(a))
                menu.addMenu(submenu)

    def link_values(self, target):
        def linkcall():
            self._link = target
            chisurf.run("cs.current_fit.update()")
            self.gb.setChecked(False)
        return linkcall

    def onLinkToggeled(self, checked):
        if checked:
            self._link = None
            chisurf.run("cs.current_fit.update()")

    def link_menu(self):
        menu = self.linkFrom_menu
        menu.clear()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, self.__class__):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)

    def __init__(self, title: str = '', **kwargs):
        super().__init__(**kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setCheckable(True)
        self.gb.setChecked(True)
        self.gb.toggled.connect(self.onLinkToggeled)
        self.gb.setTitle(title)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self._amp_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._pch0_widgets : typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        addLifetime = QtWidgets.QPushButton()
        addLifetime.setText("add")
        addLifetime.clicked.connect(self.onAddLifetime)
        lh.addWidget(addLifetime)

        removeLifetime = QtWidgets.QPushButton()
        removeLifetime.setText("del")
        removeLifetime.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeLifetime)

        readFrom = QtWidgets.QToolButton()
        readFrom.setText("read")
        # assign attribute before using it as parent
        self.readFrom = readFrom
        self.readFrom_menu = QtWidgets.QMenu(self.readFrom)
        self.readFrom_menu.aboutToShow.connect(self.read_menu)
        readFrom.setMenu(self.readFrom_menu)
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(readFrom)

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
        # assign attribute before using it as parent
        self.linkFrom = linkFrom
        self.linkFrom_menu = QtWidgets.QMenu(self.linkFrom)
        self.linkFrom_menu.aboutToShow.connect(self.link_menu)
        linkFrom.setMenu(self.linkFrom_menu)
        linkFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(linkFrom)

        normalize_amplitude = QtWidgets.QCheckBox("Norm.")
        normalize_amplitude.setChecked(True)
        normalize_amplitude.setToolTip("Normalize amplitudes to unity.\nThe sum of all amplitudes equals one.")
        normalize_amplitude.clicked.connect(self.onNormalizeAmplitudes)
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        # Build parameter widgets for existing parameters if any; otherwise add one default component
        n_existing = len(self._amplitudes) if hasattr(self, '_amplitudes') and self._amplitudes is not None else 0
        if n_existing == 0:
            # No parameters yet: create the initial component and its widgets
            self.append()
        else:
            # Create controller widgets for all existing amplitude/pch0 parameter pairs
            for i in range(n_existing):
                row_layout = QtWidgets.QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(0)
                self._amp_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._amplitudes[i],
                        layout=row_layout
                    )
                )
                self._pch0_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._pch0[i],
                        layout=row_layout
                    )
                )
                self.lh.addLayout(row_layout)

    def onNormalizeAmplitudes(self):
        chisurf.run(f"chisurf.macros.model.normalize_amplitudes('{self.name}', {self.normalize_amplitude.isChecked()})")

    def onAbsoluteAmplitudes(self):
        chisurf.run(f"chisurf.macros.model.absolute_amplitudes('{self.name}', {self.absolute_amplitude.isChecked()})")

    def onAddLifetime(self):
        chisurf.run(f"chisurf.macros.model.add_component('{self.name}')")

    def onRemoveLifetime(self):
        chisurf.run(f"chisurf.macros.model.remove_component('{self.name}')")

    def append(self, *args, **kwargs):
        ProbCh0.append(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._amp_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=layout
            )
        )

        self._pch0_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._pch0[-1],
                layout=layout
            )
        )

        self.lh.addLayout(layout)

    def pop(self):
        self._amplitudes.pop()
        self._pch0.pop()
        self._amp_widgets.pop().close()
        self._pch0_widgets.pop().close()


class PdaGaussianDistancesWidget(PdaGaussianDistances, QtWidgets.QWidget):

    def update(self, *__args):
        PdaGaussianDistances.finalize(self)
        QtWidgets.QWidget.update(self, *__args)
        # Synchronize UI widgets with underlying parameters
        try:
            for w in self._mean_widgets:
                w.finalize()
            for w in self._sigma_widgets:
                w.finalize()
            for w in self._amp_widgets:
                w.finalize()
        except Exception:
            # Fallback: at least sync the numeric values
            try:
                for w, v in zip(self._mean_widgets, self.means):
                    w.setValue(v)
                for w, v in zip(self._sigma_widgets, self.sigmas):
                    w.setValue(v)
                for w, v in zip(self._amp_widgets, self.amplitudes):
                    w.setValue(v)
            except Exception:
                pass

    @property
    def parameter_widgets(self):
        return self._mean_widgets + self._sigma_widgets + self._amp_widgets

    def __init__(self, title: str = '', **kwargs):
        super().__init__(**kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        if not title:
            title = "Distance distribution"
        self.gb.setTitle(title)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)

        self._mean_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._sigma_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._amp_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._gb: typing.List[QtWidgets.QGroupBox] = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        self.cb_limited_width = QtWidgets.QCheckBox("lim width (σ = p%·R)")
        self.cb_limited_width.setChecked(getattr(self, "limited_width", False))
        self.cb_limited_width.toggled.connect(self.onLimitedWidthToggled)
        lh.addWidget(self.cb_limited_width)

        add_component = QtWidgets.QPushButton()
        add_component.setText("add")
        add_component.clicked.connect(self.onAddComponent)
        lh.addWidget(add_component)

        remove_component = QtWidgets.QPushButton()
        remove_component.setText("del")
        remove_component.clicked.connect(self.onRemoveComponent)
        lh.addWidget(remove_component)
        lh.addStretch(1)

        self.lh.addLayout(lh)

        # Grid layout for nicely grouped Gaussian components (similar to TCSPC Gaussian widget)
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(0)
        self.lh.addLayout(self.grid_layout)

        # Build parameter widgets for existing parameters if any; otherwise add one default component
        n_existing = len(self._amplitudes) if hasattr(self, '_amplitudes') and self._amplitudes is not None else 0
        if n_existing == 0:
            self.append()
        else:
            for i in range(n_existing):
                gb = QtWidgets.QGroupBox()
                gb.setTitle(f"G{i + 1}")
                vlayout = QtWidgets.QVBoxLayout()
                vlayout.setContentsMargins(0, 0, 0, 0)
                vlayout.setSpacing(0)

                self._mean_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._means[i],
                        layout=vlayout,
                        label_text=f"R<sub>P,{i + 1}</sub>"
                    )
                )
                self._sigma_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._sigmas[i],
                        layout=vlayout,
                        label_text=f"s<sub>P,{i + 1}</sub>"
                    )
                )
                self._amp_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._amplitudes[i],
                        layout=vlayout,
                        label_text=f"x<sub>P,{i + 1}</sub>"
                    )
                )

                gb.setLayout(vlayout)
                row = i // 2
                col = i % 2
                self.grid_layout.addWidget(gb, row, col)
                self._gb.append(gb)

    def onLimitedWidthToggled(self, checked: bool):
        self.limited_width = bool(checked)
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass

    def onAddComponent(self):
        # Add a new Gaussian component and update the current fit so that
        # the new parameters are associated with the model/fits.
        self.append()
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass

    def onRemoveComponent(self):
        # Remove the last Gaussian component and update the current fit so
        # parameter lists and controllers stay in sync.
        self.pop()
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass

    def append(self, mean: float = 50.0, sigma: float = 5.0, amplitude: float = 1.0):
        PdaGaussianDistances.append(self, mean=mean, sigma=sigma, amplitude=amplitude)
        n_gauss = len(self._amplitudes)

        gb = QtWidgets.QGroupBox()
        gb.setTitle(f"G{n_gauss}")

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)

        self._mean_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._means[-1],
                layout=vlayout,
                label_text=f"R<sub>P,{n_gauss}</sub>"
            )
        )
        self._sigma_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._sigmas[-1],
                layout=vlayout,
                label_text=f"s<sub>P,{n_gauss}</sub>"
            )
        )
        self._amp_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=vlayout,
                label_text=f"x<sub>P,{n_gauss}</sub>"
            )
        )

        gb.setLayout(vlayout)
        row = (n_gauss - 1) // 2
        col = (n_gauss - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def pop(self):
        if len(self._amplitudes) == 0:
            return
        PdaGaussianDistances.pop(self)
        self._mean_widgets.pop().close()
        self._sigma_widgets.pop().close()
        self._amp_widgets.pop().close()
        gb = self._gb.pop()
        self.grid_layout.removeWidget(gb)
        gb.close()


class PdaAnisotropySpeciesWidget(PdaAnisotropySpecies, QtWidgets.QWidget):

    def update(self, *__args):
        # Keep amplitudes normalized etc.
        PdaAnisotropySpecies.finalize(self)
        QtWidgets.QWidget.update(self, *__args)
        try:
            for w, v in zip(self._amp_widgets, self.amplitudes):
                w.finalize()
        except Exception:
            try:
                for w, v in zip(self._amp_widgets, self.amplitudes):
                    w.setValue(v)
            except Exception:
                pass
        try:
            for w, v in zip(self._r_widgets, self.anisotropies):
                w.finalize()
        except Exception:
            try:
                for w, v in zip(self._r_widgets, self.anisotropies):
                    w.setValue(v)
            except Exception:
                pass

    @property
    def parameter_widgets(self):
        return self._amp_widgets + self._r_widgets

    def __init__(self, title: str = "", **kwargs):
        super().__init__(**kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        if not title:
            title = "Anisotropy species"
        self.gb.setTitle(title)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)

        self._amp_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = []
        self._r_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = []

        # Header row with add/del buttons
        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        add_component = QtWidgets.QPushButton()
        add_component.setText("add")
        add_component.clicked.connect(self.onAddComponent)
        lh.addWidget(add_component)

        remove_component = QtWidgets.QPushButton()
        remove_component.setText("del")
        remove_component.clicked.connect(self.onRemoveComponent)
        lh.addWidget(remove_component)
        lh.addStretch(1)

        self.lh.addLayout(lh)

        # Container for per-species rows
        self.rows_layout = QtWidgets.QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(0)
        self.lh.addLayout(self.rows_layout)

        # Build parameter widgets for existing parameters if any; otherwise add one default species
        n_existing = len(self._amplitudes) if hasattr(self, "_amplitudes") and self._amplitudes is not None else 0
        if n_existing == 0:
            self.append()
        else:
            for i in range(n_existing):
                row_layout = QtWidgets.QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(0)

                self._amp_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._amplitudes[i],
                        layout=row_layout,
                        label_text=f"x<sub>A,{i + 1}</sub>",
                    )
                )
                self._r_widgets.append(
                    chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                        self._anisotropies[i],
                        layout=row_layout,
                        label_text=f"r<sub>A,{i + 1}</sub>",
                    )
                )
                self.rows_layout.addLayout(row_layout)

    def onAddComponent(self):
        # Append new species to model-side group and create controllers
        self.append()
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass

    def onRemoveComponent(self):
        # Remove last species if present
        self.pop()
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass

    def append(self, amplitude: float = 1.0, r: float = 0.3):
        PdaAnisotropySpecies.append(self, amplitude=amplitude, r=r)
        n = len(self._amplitudes)

        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        self._amp_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=row_layout,
                label_text=f"x<sub>A,{n}</sub>",
            )
        )
        self._r_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._anisotropies[-1],
                layout=row_layout,
                label_text=f"r<sub>A,{n}</sub>",
            )
        )

        self.rows_layout.addLayout(row_layout)

    def pop(self):
        if len(self._amplitudes) == 0:
            return
        PdaAnisotropySpecies.pop(self)
        # Remove last row of widgets
        amp_w = self._amp_widgets.pop()
        r_w = self._r_widgets.pop()
        try:
            amp_w.close()
        except Exception:
            pass
        try:
            r_w.close()
        except Exception:
            pass
        # Also remove the last layout from rows_layout
        try:
            count = self.rows_layout.count()
            if count > 0:
                last_item = self.rows_layout.takeAt(count - 1)
                last_layout = last_item.layout()
                if last_layout is not None:
                    while last_layout.count():
                        child = last_layout.takeAt(0)
                        w = child.widget()
                        if w is not None:
                            w.setParent(None)
        except Exception:
            pass


class FretRdaAxisSettingsWidget(QtWidgets.QGroupBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("R_DA axis (FRET distance)")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        description = QtWidgets.QLabel(
            "Distance axis used for FRET-related distance distributions.\n"
            "These values control chisurf.settings.fret['rda_min'], "
            "['rda_max'], ['rda_resolution'] and ['rda_scale'] which "
            "define the grid chisurf.models.tcspc.fret.rda_axis (log or "
            "linear spacing)."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)

        fret_cfg = getattr(chisurf.settings, "fret", {}) or {}
        rda_min = float(fret_cfg.get("rda_min", 1.0))
        rda_max = float(fret_cfg.get("rda_max", 130.0))
        rda_res = int(fret_cfg.get("rda_resolution", 96))
        rda_scale = str(fret_cfg.get("rda_scale", "log")).lower()

        self.sb_min = QtWidgets.QDoubleSpinBox()
        self.sb_min.setRange(0.01, 1.0e4)
        self.sb_min.setDecimals(3)
        self.sb_min.setValue(rda_min)
        self.sb_min.setSuffix(" 	")

        self.sb_max = QtWidgets.QDoubleSpinBox()
        self.sb_max.setRange(0.01, 1.0e4)
        self.sb_max.setDecimals(3)
        self.sb_max.setValue(rda_max)
        self.sb_max.setSuffix(" 	")

        self.sb_n = QtWidgets.QSpinBox()
        self.sb_n.setRange(4, 4096)
        self.sb_n.setValue(rda_res)

        self.cb_scale = QtWidgets.QComboBox()
        self.cb_scale.addItems(["log", "lin"])
        if rda_scale in ("log", "lin"):
            try:
                idx = self.cb_scale.findText(rda_scale)
                if idx >= 0:
                    self.cb_scale.setCurrentIndex(idx)
            except Exception:
                pass

        form.addRow("R_DA min:", self.sb_min)
        form.addRow("R_DA max:", self.sb_max)
        form.addRow("N points:", self.sb_n)
        form.addRow("Scale:", self.cb_scale)

        layout.addLayout(form)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)

        save_btn = QtWidgets.QPushButton("Save axis")
        save_btn.clicked.connect(self.on_save_clicked)
        button_layout.addWidget(save_btn)
        button_layout.addStretch(1)

        layout.addLayout(button_layout)

    def on_save_clicked(self):
        rda_min = float(self.sb_min.value())
        rda_max = float(self.sb_max.value())
        n_points = int(self.sb_n.value())
        scale = self.cb_scale.currentText().strip().lower() if hasattr(self, "cb_scale") else "log"
        if scale not in ("log", "lin"):
            scale = "log"

        if rda_max <= rda_min:
            tmp = rda_min
            rda_min = rda_max
            rda_max = tmp
            self.sb_min.setValue(rda_min)
            self.sb_max.setValue(rda_max)

        ok = set_fret_rda_axis(
            rda_min=rda_min,
            rda_max=rda_max,
            rda_resolution=n_points,
            rda_scale=scale,
        )
        if not ok:
            return

        try:
            if not isinstance(getattr(chisurf.settings, "fret", None), dict):
                chisurf.settings.fret = {}
        except Exception:
            chisurf.settings.fret = {}

        chisurf.settings.fret["rda_min"] = float(rda_min)
        chisurf.settings.fret["rda_max"] = float(rda_max)
        chisurf.settings.fret["rda_resolution"] = int(n_points)
        chisurf.settings.fret["rda_scale"] = scale

        try:
            new_axis = build_fret_rda_axis(
                chisurf.settings.fret["rda_min"],
                chisurf.settings.fret["rda_max"],
                chisurf.settings.fret["rda_resolution"],
                chisurf.settings.fret.get("rda_scale", scale),
            )
        except Exception:
            return

        try:
            try:
                chisurf.fluorescence.rda_axis = new_axis
            except Exception:
                pass
            tcspc_fret.rda_axis = new_axis
            globals()["rda_axis"] = new_axis
        except Exception:
            pass

        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass


def get_distribution(fit, kw_hist):
    pda = fit.model.pda

    # The tttrlib.Pda C++ code iterates over the S1S2 matrix in a way that
    # effectively calls the histogram callback with arguments in the order
    # (red, green), while the documented convention and chisurf's semantics
    # expect (green, red). To keep all existing model definitions consistent
    # (including S0/S1 and S1/(S0+S1) plots), we wrap the user-provided
    # histogram function so that it always receives (green, red).
    axis_type = kw_hist.pop('_axis_type', None)
    inner = kw_hist.pop('histogram_function', lambda ch1, ch2: ch1 / max(1, ch2))

    def histogram_function(ch1, ch2, _inner=inner):
        # ch1, ch2 from tttrlib are effectively (red, green); swap to
        # (green, red) before applying the semantic function.
        return _inner(ch2, ch1)

    pda.histogram_function = histogram_function

    data_obj = getattr(fit, 'data', None)
    pda_meta = getattr(data_obj, 'pda', None)
    if not isinstance(pda_meta, dict):
        s1s2_experimental = getattr(getattr(fit.data, 'pda', None), 's1s2', None)
        if s1s2_experimental is None:
            return []
        s1s2_shape = getattr(s1s2_experimental, 'shape', None)
    else:
        s1s2_experimental = pda_meta.get('s1s2')
        if s1s2_experimental is None:
            return []
        s1s2_shape = pda_meta.get('shape')

    s1s2_model = np.asarray(pda.get_S1S2_matrix(), dtype=float)
    s1s2_data = np.asarray(s1s2_experimental, dtype=float)

    try:
        if s1s2_shape is not None and len(s1s2_shape) == 2:
            ny, nx = int(s1s2_shape[0]), int(s1s2_shape[1])
            s1s2_model = s1s2_model[:ny, :nx]
            s1s2_data = s1s2_data[:ny, :nx]
    except Exception:
        pass

    # Apply photon-number gating (nPh_min/nPh_max) if available on the model
    # nuisance group. This mirrors the logic used in
    # :meth:`PdaGaussianDistanceModel._get_1d_residuals` so that the
    # displayed histograms follow the same N-range as the residuals.
    try:
        nuisance = getattr(fit.model, 'nuisance', None)
        row_indices = np.asarray(pda_meta.get('row_indices'), dtype=np.int64) if isinstance(pda_meta, dict) else np.array([], dtype=np.int64)
        col_indices = np.asarray(pda_meta.get('col_indices'), dtype=np.int64) if isinstance(pda_meta, dict) else np.array([], dtype=np.int64)
        if (
            nuisance is not None
            and row_indices.size
            and col_indices.size
            and row_indices.size == col_indices.size
        ):
            pda_nmin = int(pda_meta.get('minimum_number_of_photons', 0) or 0)
            pda_nmax = int(pda_meta.get('maximum_number_of_photons', 0) or 0)
            try:
                nmin_param = int(round(float(nuisance.nPh_min)))
            except Exception:
                nmin_param = 0
            try:
                nmax_param = int(round(float(nuisance.nPh_max)))
            except Exception:
                nmax_param = 0
            if nmin_param != 0 or nmax_param != 0:
                nmin = nmin_param if nmin_param > 0 else pda_nmin
                nmax = nmax_param if nmax_param > 0 else pda_nmax
                if nmax >= nmin:
                    shp = getattr(s1s2_data, 'shape', None)
                    if shp is not None and len(shp) == 2:
                        ny, nx = int(shp[0]), int(shp[1])
                        mask2d = np.zeros((ny, nx), dtype=bool)
                        N = row_indices + col_indices
                        sel = (N >= nmin) & (N <= nmax)
                        if np.any(sel):
                            mask2d[row_indices[sel], col_indices[sel]] = True
                            s1s2_model = np.where(mask2d, s1s2_model, 0.0)
                            s1s2_data = np.where(mask2d, s1s2_data, 0.0)
    except Exception:
        pass

    model_x, model_y = pda.get_1dhistogram(
        s1s2=s1s2_model.flatten(),
        **kw_hist
    )
    data_x, data_y = pda.get_1dhistogram(
        s1s2=s1s2_data.flatten(),
        **kw_hist
    )

    # Build a weighted residual curve assuming counting shot noise:
    # w-residual = (data - model) / sqrt(max(data, 1)).
    curves = [[data_y, data_x], [model_y, model_x]]
    try:
        if data_y is not None and model_y is not None:
            dy = np.asarray(data_y, dtype=float)
            my = np.asarray(model_y, dtype=float)
            if dy.shape == my.shape:
                sigma = np.sqrt(np.maximum(dy, 1.0))
                wres = (dy - my) / sigma
                curves.append([wres, data_x])
    except Exception:
        pass

    # For PDA Gaussian-distance models, optionally append per-Gaussian
    # component curves transformed to the same 1D axis as the PDA histogram.
    try:
        extra_curves = _get_gaussian_component_curves_for_pda(
            fit=fit,
            axis_type=axis_type,
            hist_x=model_x,
            hist_y=model_y,
        )
        if extra_curves:
            curves.extend(extra_curves)
    except Exception:
        pass

    return curves


def _get_gaussian_component_curves_for_pda(
        fit,
        axis_type: str | None,
        hist_x,
        hist_y,
):
    """Return per-Gaussian component curves for the PDA Gaussian-distance model.

    Each curve is mapped to the same x-axis (S0/S1 or S1/(S0+S1)) as the
    PDA 1D histogram and scaled so that the sum of Gaussian components sits
    visually below the main PDA model curve.
    """

    if axis_type is None:
        return []

    model = getattr(fit, 'model', None)
    if not isinstance(model, PdaGaussianDistanceModel):
        return []

    distances = getattr(model, 'distances', None)
    if not isinstance(distances, PdaGaussianDistances):
        return []

    means = distances.means
    sigmas = distances.sigmas
    amplitudes = distances.amplitudes
    if means.size == 0 or sigmas.size == 0 or amplitudes.size == 0:
        return []

    # Distance grid and per-Gaussian distributions in distance-space.
    r = rda_axis
    component_curves_r = []
    for mean, sigma, amp in zip(means, sigmas, amplitudes):
        if sigma <= 0.0 or amp <= 0.0:
            continue
        try:
            y_r = amp * distfuncs.normal_distribution(
                x=r,
                loc=float(mean),
                scale=float(sigma),
                norm=False,
            )
        except Exception:
            continue
        component_curves_r.append(y_r)

    if not component_curves_r:
        return []

    # Map distance r to the 1D PDA axis using the same nuisance/FRET
    # parameters as the Gaussian-distance model.
    try:
        R0 = model.fret_parameters.forster_radius
        E = distance_to_fret_efficiency(r, R0)

        n = model.nuisance
        alpha = n.alpha
        gG = n.gG
        gR = n.gR
        QYD = n.QYD
        QYA = n.QYA

        gamma = (gR * QYA) / (gG * QYD)
        eps = 1e-12
        E_safe = np.clip(E, eps, 1.0 - eps)
        p_G = 1.0 / (1.0 + alpha + gamma * E_safe / (1.0 - E_safe))

        if axis_type == 'S0/S1':
            # Approximate S0/S1 as p_G / (1 - p_G).
            x_axis = p_G / np.maximum(1.0 - p_G, eps)
        elif axis_type == 'S1/(S0+S1)':
            # Approximate S1/(S0+S1) as the red fraction ~ 1 - p_G.
            x_axis = 1.0 - p_G
        else:
            # Fallback: use FRET efficiency itself.
            x_axis = E_safe
    except Exception:
        return []

    # Sort by x to get monotonic curves for plotting.
    try:
        order = np.argsort(x_axis)
        x_sorted = x_axis[order]
    except Exception:
        return []

    # Clip component curves to the same x-range as the PDA 1D histogram so
    # that they respect the x_min/x_max of the Distribution plot control.
    x_min = None
    x_max = None
    try:
        if hist_x is not None:
            hx = np.asarray(hist_x, dtype=float)
            if hx.size > 0:
                x_min = float(np.nanmin(hx))
                x_max = float(np.nanmax(hx))
    except Exception:
        x_min = None
        x_max = None

    comp_curves = []
    for y_r in component_curves_r:
        try:
            y_sorted = y_r[order]
        except Exception:
            continue

        x_use = x_sorted
        y_use = y_sorted
        if x_min is not None and x_max is not None:
            mask = (x_sorted >= x_min) & (x_sorted <= x_max)
            if not np.any(mask):
                continue
            x_use = x_sorted[mask]
            y_use = y_sorted[mask]

        comp_curves.append([y_use, x_use])

    if not comp_curves:
        return []

    # Scale component curves so they sit below the full PDA model curve.
    try:
        max_model = float(np.nanmax(hist_y)) if hist_y is not None else 0.0
    except Exception:
        max_model = 0.0

    if max_model > 0.0:
        try:
            max_comp = max(float(np.nanmax(y)) for y, _ in comp_curves)
        except Exception:
            max_comp = 0.0
        if max_comp > 0.0:
            # Place components at about 30% of the PDA model amplitude.
            scale = 0.3 * max_model / max_comp
            comp_curves = [[y * scale, x] for (y, x) in comp_curves]

    return comp_curves


def get_pda_residual_image(fit_group, weighted: bool = True):
    """Return a 2D residual image for a PDA fit.

    This accessor is used by the generic chisurf.plots.Residual2DPlot and
    therefore must not depend on any plot internals. It only inspects the
    PDA data/model state and returns (image, x_axis, y_axis).
    """

    # Accept either a FitGroup (with .selected_fit) or a plain Fit
    fit = getattr(fit_group, 'selected_fit', fit_group)

    data_pda = getattr(getattr(fit, 'data', None), 'pda', None)
    model_obj = getattr(getattr(fit, 'model', None), 'pda', None)
    if data_pda is None or model_obj is None:
        return None, None, None

    try:
        data_2d = np.asarray(data_pda.get('s1s2'), dtype=float)
        model_2d = np.asarray(getattr(model_obj, 's1s2'), dtype=float)
    except Exception:
        return None, None, None

    if data_2d.ndim != 2 or model_2d.ndim != 2:
        return None, None, None

    n0 = min(data_2d.shape[0], model_2d.shape[0])
    n1 = min(data_2d.shape[1], model_2d.shape[1])
    d = data_2d[:n0, :n1]
    m = model_2d[:n0, :n1]

    if weighted:
        sigma = np.sqrt(np.maximum(d, 1.0))
        img = (d - m) / sigma
    else:
        img = d - m

    x = np.arange(n1, dtype=float)
    y = np.arange(n0, dtype=float)
    return img, x, y


class PdaSimpleModelWidget(ModelWidget, PdaSimpleModel):

    plot_classes = [
        (
            chisurf.plots.DistributionPlot,
            {
                'with_residual_panel': True,
                'distribution_options': {
                    'S1/(S0+S1)': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 1.0,
                                "x_min": 0.0,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch2 / max(1, ch2 + ch1),
                            }
                        },
                        'scale_x': 'lin',
                        'curve_options': {
                            'stepMode': False,
                            'connect': 'all',
                            'multi_curve': True,
                            # data, model, residual
                            'symbol': ['None', 'None', 'None'],
                            'pen': ['b', 'r', 'k'],
                            # Shade the experimental PDA histogram under
                            # the curve using the global data color, in the
                            # same visual style as MaxEnt FCS distributions.
                            'fillLevel': 0.0,
                            'fillBrush': chisurf.settings.gui['plot']['colors']['data'],
                        }
                    },
                    'S0/S1': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 500.0,
                                "x_min": 0.01,
                                "log_x": True,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch1 / max(1, ch2),
                            }
                        },
                        'scale_x': 'log',
                        'curve_options': {
                            'stepMode': False,
                            'connect': 'all',
                            'multi_curve': True,
                            # data, model, residual
                            'symbol': ['None', 'None', 'None'],
                            'pen': ['b', 'r', 'k'],
                            # Shade the experimental PDA histogram under
                            # the curve using the global data color.
                            'fillLevel': 0.0,
                            'fillBrush': chisurf.settings.gui['plot']['colors']['data'],
                        }
                    }
                }
            }
        ),
        (chisurf.plots.ResidualPlot, {}),
        (
            chisurf.plots.Residual2DPlot,
            {
                'accessor': get_pda_residual_image,
                'accessor_kwargs': {
                    'weighted': True
                }
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
    ]

    def __init__(
            self,
            fit: Fit,
            icon: QtGui.QIcon | None = None,
            hide_nuisances: bool = False,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(fit=fit, icon=icon, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        ## add widgets
        res_layout = QtWidgets.QHBoxLayout()
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(0)

        res_label = QtWidgets.QLabel("Residuals:")
        self.rb_res_2d = QtWidgets.QRadioButton("2D")
        self.rb_res_1d = QtWidgets.QRadioButton("1D proj")
        self.rb_res_1d.setChecked(True)
        self.rb_res_2d.toggled.connect(self.onResidualModeChanged)
        self.rb_res_1d.toggled.connect(self.onResidualModeChanged)
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.rb_res_2d)
        res_layout.addWidget(self.rb_res_1d)
        res_layout.addStretch(1)
        layout.addLayout(res_layout)

        photon_nuisance = PdaPhotonRange(name='pda_photon_range', fit=fit, **kwargs)
        photon_range = PdaPhotonRangeWidget(nuisance=photon_nuisance)
        background = BackgroundWidget(fit=fit, **kwargs)
        pch0 = ProbCh0Widget(fit=fit, **kwargs)

        layout.addWidget(photon_range)
        if not hide_nuisances:
            layout.addWidget(background)
        layout.addWidget(pch0)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.photon_range = photon_range
        self.background = background
        self.nuisance = photon_nuisance
        self.pch0 = pch0

    def onResidualModeChanged(self):
      mode = "1D" if getattr(self, "rb_res_1d", None) is not None and self.rb_res_1d.isChecked() else "2D"
      self.residual_mode = mode
      try:
          chisurf.run("cs.current_fit.update()")
      except Exception:
          pass


class PdaAnisotropyModelWidget(ModelWidget, PdaAnisotropyModel):

    plot_classes = [
        (
            chisurf.plots.DistributionPlot,
            {
                "with_residual_panel": True,
                "distribution_options": {
                    "S1/(S0+S1)": {
                        "attribute": "fit",
                        "accessor": get_distribution,
                        "accessor_kwargs": {
                            "kw_hist": {
                                "x_max": 1.0,
                                "x_min": 0.0,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch2 / max(1.0, ch2 + ch1),
                            }
                        },
                        "scale_x": "lin",
                        "curve_options": {
                            "stepMode": False,
                            "connect": "all",
                            "multi_curve": True,
                            "symbol": ["None", "None", "None"],
                            "pen": ["b", "r", "k"],
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    },
                    "S0/S1": {
                        "attribute": "fit",
                        "accessor": get_distribution,
                        "accessor_kwargs": {
                            "kw_hist": {
                                "x_max": 500.0,
                                "x_min": 0.01,
                                "log_x": True,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch1 / max(1.0, ch2),
                            }
                        },
                        "scale_x": "log",
                        "curve_options": {
                            "stepMode": False,
                            "connect": "all",
                            "multi_curve": True,
                            "symbol": ["None", "None", "None"],
                            "pen": ["b", "r", "k"],
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    },
                    # Raw polarization: (S0 - S1) / (S0 + S1)
                    "Δ/(S0+S1)": {
                        "attribute": "fit",
                        "accessor": get_distribution,
                        "accessor_kwargs": {
                            "kw_hist": {
                                "x_max": 0.5,
                                "x_min": -0.5,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: (ch1 - ch2)
                                / max(1.0, ch1 + ch2),
                            }
                        },
                        "scale_x": "lin",
                        "curve_options": {
                            "stepMode": False,
                            "connect": "all",
                            "multi_curve": True,
                            "symbol": ["None", "None", "None"],
                            "pen": ["b", "r", "k"],
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    },
                    # Raw anisotropy-like quantity: (S0 - S1) / (S0 + 2*S1)
                    "Δ/(S0+2*S1)": {
                        "attribute": "fit",
                        "accessor": get_distribution,
                        "accessor_kwargs": {
                            "kw_hist": {
                                "x_max": 0.5,
                                "x_min": -0.5,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: (ch1 - ch2)
                                / max(1.0, ch1 + 2.0 * ch2),
                            }
                        },
                        "scale_x": "lin",
                        "curve_options": {
                            "stepMode": False,
                            "connect": "all",
                            "multi_curve": True,
                            "symbol": ["None", "None", "None"],
                            "pen": ["b", "r", "k"],
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    },
                },
            },
        ),
        (chisurf.plots.ResidualPlot, {}),
        (
            chisurf.plots.Residual2DPlot,
            {
                "accessor": get_pda_residual_image,
                "accessor_kwargs": {
                    "weighted": True,
                },
            },
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
    ]

    def __init__(
        self,
        fit: "Fit",
        icon: QtGui.QIcon | None = None,
        hide_nuisances: bool = False,
        **kwargs,
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(fit=fit, icon=icon, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        res_layout = QtWidgets.QHBoxLayout()
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(0)

        res_label = QtWidgets.QLabel("Residuals:")
        self.rb_res_2d = QtWidgets.QRadioButton("2D")
        self.rb_res_1d = QtWidgets.QRadioButton("1D proj")
        self.rb_res_1d.setChecked(True)
        self.rb_res_2d.toggled.connect(self.onResidualModeChanged)
        self.rb_res_1d.toggled.connect(self.onResidualModeChanged)
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.rb_res_2d)
        res_layout.addWidget(self.rb_res_1d)
        res_layout.addStretch(1)
        layout.addLayout(res_layout)

        # Nuisance parameter group (backgrounds, G, l1, l2)
        nuisance_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.nuisance
        )

        # Anisotropy species (amplitude + r_i with add/del). We attach the
        # widget instance directly to the model so that the same
        # PdaAnisotropySpecies object is used for both GUI and backend,
        # mirroring the pattern used by the discrete and Gaussian PDA
        # models.
        species_widget = PdaAnisotropySpeciesWidget(fit=fit, name="pda_aniso_species")
        # Ensure the model sees the widget-backed species group.
        self.species = species_widget

        if not hide_nuisances:
            layout.addWidget(nuisance_widget)
        layout.addWidget(species_widget)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.nuisance_widget = nuisance_widget
        self.species_widget = species_widget

    def onResidualModeChanged(self):
        mode = "1D" if getattr(self, "rb_res_1d", None) is not None and self.rb_res_1d.isChecked() else "2D"
        self.residual_mode = mode
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass


class PdaGaussianDistanceModelWidget(ModelWidget, PdaGaussianDistanceModel):

    # Use a PDA distribution plot where S0/S1 with log-scaled x-axis is the
    # default option, while still exposing S1/(S0+S1) as an alternative.
    plot_classes = [
        (
            chisurf.plots.DistributionPlot,
            {
                'with_residual_panel': True,
                'distribution_options': {
                    'S0/S1': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 500.0,
                                "x_min": 0.01,
                                "log_x": True,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch1 / max(1, ch2),
                                "_axis_type": "S0/S1",
                            }
                        },
                        'scale_x': 'log',
                        'curve_options': {
                            'stepMode': False,
                            'connect': 'all',
                            'multi_curve': True,
                            # data, model, residual, + per-Gaussian components
                            'symbol': ['None', 'None', 'None'],
                            'pen': ['b', 'r', 'k', 'g', 'm', 'c', 'y'],
                            # Shade the experimental PDA histogram (first
                            # curve) under the line in the same style as the
                            # MaxEnt FCS distributions.
                            'fillLevel': 0.0,
                            'fillBrush': chisurf.settings.gui['plot']['colors']['data'],
                        }
                    },
                    'S1/(S0+S1)': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 1.0,
                                "x_min": 0.0,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch2 / max(1, ch2 + ch1),
                                "_axis_type": "S1/(S0+S1)",
                            }
                        },
                        'curve_options': {
                            'stepMode': False,
                            'connect': 'all',
                            'multi_curve': True,
                            # data, model, residual, + per-Gaussian components
                            'symbol': ['None', 'None', 'None'],
                            'pen': ['b', 'r', 'k', 'g', 'm', 'c', 'y'],
                            # Shade the experimental PDA histogram under the
                            # curve using the global data color.
                            'fillLevel': 0.0,
                            'fillBrush': chisurf.settings.gui['plot']['colors']['data'],
                        }
                    },
                }
            }
        ),
        (chisurf.plots.ResidualPlot, {}),
        (
            chisurf.plots.Residual2DPlot,
            {
                'accessor': get_pda_residual_image,
                'accessor_kwargs': {
                    'weighted': True
                }
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
    ]

    def __init__(
            self,
            fit: Fit,
            icon: QtGui.QIcon | None = None,
            hide_nuisances: bool = False,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        # Let the usual MRO handle initialization of ModelWidget and
        # PdaGaussianDistanceModel (including creation of self.fret_parameters
        # and the internal tttrlib.Pda instance).
        super().__init__(fit=fit, icon=icon, **kwargs)

        # Build dedicated widgets for the nuisance/FRET parameters and
        # distance distribution, following the same pattern as the TCSPC
        # GaussianModelWidget.
        nuisance = PdaFretNuisanceWidget(fit=fit, **kwargs)
        photon_range = PdaPhotonRangeWidget(nuisance=nuisance)
        distances = PdaGaussianDistancesWidget(fit=fit, **kwargs)

        # Generic widget for the shared FRET parameter group (R0, tau0, kappa2, ...)
        fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        res_layout = QtWidgets.QHBoxLayout()
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(0)

        res_label = QtWidgets.QLabel("Residuals:")
        self.rb_res_2d = QtWidgets.QRadioButton("2D")
        self.rb_res_1d = QtWidgets.QRadioButton("1D")
        self.rb_res_1d.setChecked(True)
        self.rb_res_2d.toggled.connect(self.onResidualModeChanged)
        self.rb_res_1d.toggled.connect(self.onResidualModeChanged)
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.rb_res_2d)
        res_layout.addWidget(self.rb_res_1d)
        res_layout.addStretch(1)
        layout.addLayout(res_layout)

        layout.addWidget(photon_range)
        if not hide_nuisances:
            layout.addWidget(nuisance)
        layout.addWidget(fret_parameters_widget)

        layout.addWidget(distances)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.nuisance = nuisance
        self.photon_range = photon_range
        self.distances = distances
        self.fret_parameters_widget = fret_parameters_widget

    def onResidualModeChanged(self):
        mode = "1D" if getattr(self, "rb_res_1d", None) is not None and self.rb_res_1d.isChecked() else "2D"
        self.residual_mode = mode
        try:
            chisurf.run("cs.current_fit.update()")
        except Exception:
            pass
