from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import chisurf as cs
from chisurf import typing
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.general
import chisurf.core.actions
import chisurf.core.math.datatools
import chisurf.core.plot_transforms as plot_transforms
import chisurf.gui.plots
import chisurf.core.fitting.parameter
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.core.models.tcspc.lifetime import Lifetime, LifetimeModel, LifetimeMixtureModel

# These will be imported from the new module structure
from chisurf import logging
from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from chisurf.gui.widgets.models.tcspc.anisotropy import AnisotropyWidget

if TYPE_CHECKING:
    from chisurf.core.fitting.fit import Fit, FitGroup


ADD_BUTTON_STYLE = (
    "QPushButton { background-color: #1f7a1f; color: white; border: 1px solid #166016; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #249124; }"
    "QPushButton:pressed { background-color: #155815; }"
)

REMOVE_BUTTON_STYLE = (
    "QPushButton { background-color: #a82020; color: white; border: 1px solid #7d1717; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #bf2626; }"
    "QPushButton:pressed { background-color: #7d1717; }"
)


class LifetimeWidget(Lifetime, QtWidgets.QWidget):

    # TODO: needs docstring
    def update(self, *__args):
        """Update the state and emit signals."""
        Lifetime.update(self)
        QtWidgets.QWidget.update(self, *__args)
        for w, v in zip(self._amp_widgets, self.amplitudes):
            w.setValue(v)
        for w, v in zip(self._lifetime_widgets, self.lifetimes):
            w.setValue(v)

    @property
    def parameter_widgets(self):
        """List of parameter widgets for amplitude and lifetime."""
        return self._amp_widgets + self._lifetime_widgets

    # TODO: needs docstring
    def read_values(self, target):
        """Create a callback to read values from another widget."""

        def linkcall():
            """Read parameter values from the target widget into this one."""
            fit_idx = self._amp_widgets[0].fitting_parameter.fit_idx
            for key in self.parameter_dict:
                p = target.parameters_all_dict[key]
                cs.core.actions.dispatch(
                    name="parameter.value",
                    payload={
                        "parameter_name": str(key),
                        "value": float(p.value),
                        "fit_index": int(fit_idx),
                    },
                )
            cs.core.actions.dispatch(
                name="fit.update",
                payload={"fit_index": int(fit_idx)},
            )

        return linkcall

    # TODO: needs docstring
    def read_menu(self):
        """Build the read-from menu."""
        menu = self.readFrom_menu
        menu.clear()
        for f in get_fitting_client().get_fit_objects():
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.read_values(a))
                menu.addMenu(submenu)

    # TODO: needs docstring
    def link_values(self, target):
        """Create a callback to link values to another widget."""
        def linkcall():
            """Link values from the target widget and trigger fit update."""
            self._link = target
            # Find the correct fit index for this model
            fit_index = 0
            try:
                # Try to find which fit contains this model
                for i, fit_obj in enumerate(get_fitting_client().get_fit_objects()):
                    if hasattr(fit_obj, 'model') and fit_obj.model is self:
                        fit_index = i
                        break
            except Exception:
                pass
            
            cs.core.actions.dispatch(
                name="fit.update",
                payload={"fit_index": int(fit_index)},
            )
            self.gb.setChecked(False)
        return linkcall

    # TODO: needs docstring
    def onLinkToggeled(self, checked):
        """Handle link toggle."""
        if checked:
            self._link = None
            # Find the correct fit index for this model
            fit_index = 0
            try:
                # Try to find which fit contains this model
                for i, fit_obj in enumerate(get_fitting_client().get_fit_objects()):
                    if hasattr(fit_obj, 'model') and fit_obj.model is self:
                        fit_index = i
                        break
            except Exception:
                pass
            
            cs.core.actions.dispatch(
                name="fit.update",
                payload={"fit_index": int(fit_index)},
            )

    # TODO: needs docstring
    def link_menu(self):
        """Build the link-from menu."""
        menu = self.linkFrom_menu
        menu.clear()
        for f in get_fitting_client().get_fit_objects():
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)

    # TODO: needs docstring
    def __init__(self, title: str = '', **kwargs):
        """Initialize the instance."""
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
        self._amp_widgets: typing.List[cs.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._lifetime_widgets: typing.List[cs.gui.widgets.fitting.widgets.FittingParameterWidget] = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        addLifetime = QtWidgets.QPushButton()
        addLifetime.setText("add")
        addLifetime.setStyleSheet(ADD_BUTTON_STYLE)
        addLifetime.clicked.connect(self.onAddLifetime)
        lh.addWidget(addLifetime)

        removeLifetime = QtWidgets.QPushButton()
        removeLifetime.setText("del")
        removeLifetime.setStyleSheet(REMOVE_BUTTON_STYLE)
        removeLifetime.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeLifetime)

        readFrom = QtWidgets.QToolButton()
        readFrom.setText("read")
        self.readFrom = readFrom
        self.readFrom_menu = QtWidgets.QMenu(self.readFrom)
        self.readFrom_menu.aboutToShow.connect(self.read_menu)
        readFrom.setMenu(self.readFrom_menu)
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(readFrom)

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
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
        normalize_amplitude.setStyleSheet("QCheckBox { spacing: 2px; min-height: 14px; max-height: 16px; }")
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        absolute_amplitude.setStyleSheet("QCheckBox { spacing: 2px; min-height: 14px; max-height: 16px; }")
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        self.append()

    def __setstate__(self, state):
        """Restore state from a serialized dictionary."""
        n_lifetime = (len(state.keys()) - 2) // 2
        for _ in range(n_lifetime):
            self.onAddLifetime()
        super().__setstate__(state)

    # TODO: needs docstring
    def onNormalizeAmplitudes(self):
        """Handle normalize amplitudes checkbox."""
        cs.core.actions.dispatch(
            name="model.normalize_amplitudes",
            payload={
                "component_name": str(self.name),
                "normalize": bool(self.normalize_amplitude.isChecked()),
            },
        )
        cs.core.actions.dispatch(
            name="model.absolute_amplitudes",
            payload={
                "component_name": str(self.name),
                "absolute": bool(self.absolute_amplitude.isChecked()),
            },
        )
        cs.core.actions.dispatch(
            name="model.add_component",
            payload={"component_name": str(self.name)},
        )
        cs.core.actions.dispatch(
            name="model.remove_component",
            payload={"component_name": str(self.name)},
        )

    # TODO: needs docstring
    def onAbsoluteAmplitudes(self):
        """Handle absolute amplitudes checkbox."""
        self.onNormalizeAmplitudes()

    # TODO: needs docstring
    def onAddLifetime(self):
        """Add a lifetime component to every member of the fit group.

        Routed through the ``model.add_component`` action (like the anisotropy
        rotation controls) so it applies to all grouped fits and works over the
        RPC/server path, not only the locally displayed member.
        """
        cs.core.actions.dispatch(
            name="model.add_component",
            payload={"component_name": "lifetimes"},
        )

    # TODO: needs docstring
    def onRemoveLifetime(self):
        """Remove the last lifetime component from every member of the group."""
        cs.core.actions.dispatch(
            name="model.remove_component",
            payload={"component_name": "lifetimes"},
        )

    # TODO: needs docstring
    def append(self, *args, **kwargs):
        """Add a new component."""
        Lifetime.append(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._amp_widgets.append(
            cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=layout
            )
        )

        self._lifetime_widgets.append(
            cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._lifetimes[-1],
                layout=layout
            )
        )

        self.lh.addLayout(layout)

    # TODO: needs docstring
    def pop(self):
        """Remove the last component."""
        self._amplitudes.pop()
        self._lifetimes.pop()
        self._amp_widgets.pop().close()
        self._lifetime_widgets.pop().close()


class LifetimeModelWidgetBase(ModelWidget, LifetimeModel):

    plot_classes = [
        (
            cs.gui.plots.LinePlot,
            {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'time (ns)',
                'y_label': 'counts'
            }
        ),
        (cs.gui.plots.FitTablePlot, {}),
        (cs.gui.plots.FitInfo, {}),
        (cs.gui.plots.ParameterScanPlot, {}),
        (
            cs.gui.plots.DistributionPlot,
            {
                'distribution_options': {
                    'Lifetime': {
                        'attribute': 'lifetime_spectrum',
                        'accessor': cs.core.math.datatools.interleaved_to_two_columns,
                        'accessor_kwargs': {'sort': True},
                        'curve_options': {
                            'stepMode': False,
                            'connect': False,
                            'bar_mode': 'sticks',
                            'symbol': "o"
                        }
                    }
                }
            }
        ),
        (cs.gui.plots.ResidualPlot, {})
    ]

    # TODO: needs docstring
    def __init__(
            self,
            fit: Fit,
            icon: QtGui.QIcon | None = None,
            hide_nuisances: bool = False,
            **kwargs
    ):
        """Initialize the instance."""
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(fit=fit, icon=icon)

        corrections = CorrectionsWidget(
            fit=fit,
            **kwargs
        )
        generic = GenericWidget(fit=fit, parent=self, **kwargs)
        convolve = ConvolveWidget(
            name='convolve',
            fit=fit,
            hide_curve_convolution=False,
            **kwargs
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        ## add widgets
        if not hide_nuisances:
            layout.addWidget(convolve)
            layout.addWidget(generic)
            layout.addWidget(corrections)

        if hide_nuisances:
            corrections.hide()

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.generic = generic
        self.corrections = corrections
        self.convolve = convolve












class LifetimeModelWidget(LifetimeModelWidgetBase):
    """
    A widget for displaying and manipulating fluorescence lifetime models.

    This widget extends LifetimeModelWidgetBase by adding specific components
    for working with fluorescence lifetime data, including lifetime parameters
    and anisotropy settings. It provides a graphical interface for configuring
    and visualizing fluorescence lifetime models used in time-correlated single
    photon counting (TCSPC) experiments.
    """

    # TODO: needs docstring
    def __init__(
        self,
        fit: FitGroup,
        lifetimes: cs.core.fitting.parameter.FittingParameterGroup = None,
        **kwargs
     ):
        """Initialize the instance."""
        super().__init__(fit=fit, **kwargs)
        if lifetimes is None:
            lifetimes = LifetimeWidget(
                name='lifetimes',
                parent=self,
                title='Lifetimes',
                short='L',
                fit=fit
            )
        self.lifetimes = lifetimes
        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            fit=fit,
            model=self,
            **kwargs
        )
        self.anisotropy = anisotropy

        # Automatically set polarization type for fits
        logging.debug("LifetimeModelWidget: Checking for polarization type setup.")
        # Use the unified method to set polarization based on group position
        polarization_set = self.anisotropy.set_polarization_by_group_position(fit, self)
        if polarization_set:
            logging.info(f"Polarization type set to {self.anisotropy.polarization_type}")

        self.layout.addWidget(self.lifetimes)
        self.layout.addWidget(anisotropy)

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        super().finalize()
        self.lifetimes.update()


class LifetimeMixtureModelWidget(LifetimeMixtureModel, LifetimeModelWidgetBase):

    plot_classes = [
        (
            cs.gui.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'time (ns)',
                'y_label': 'counts',
                'plot_irf': True
            }
         ),
        (cs.gui.plots.FitTablePlot, {}),
        (cs.gui.plots.FitInfo, {}),
        (cs.gui.plots.ParameterScanPlot, {}),
        (cs.gui.plots.ResidualPlot, {}),
        (
            cs.gui.plots.DistributionPlot,
            {
                'distribution_options': {
                    'Lifetime': {
                        'attribute': 'lifetime_spectrum',
                        'accessor': cs.core.math.datatools.interleaved_to_two_columns,
                        'accessor_kwargs': {'sort': True},
                        'curve_options': {
                            'stepMode': False,
                            'connect': False,
                            'symbol': "o"
                        }
                    }
                }
            }
        )
    ]

    # TODO: needs docstring
    def __init__(self, fit: cs.core.fitting.fit.FitGroup, **kwargs):
        """Initialize the instance."""
        super().__init__(fit=fit, **kwargs)

        hl = QtWidgets.QHBoxLayout()
        self.layout.addLayout(hl)
        self.cb = QtWidgets.QComboBox(None)
        hl.addWidget(self.cb)

        self.update_button = QtWidgets.QToolButton(None)
        self.update_button.setText("update")
        hl.addWidget(self.update_button)
        self.update_button.clicked.connect(self.onUpdataFitList)

        label = QtWidgets.QLabel('Name')
        self.name_box = QtWidgets.QLineEdit()
        self.name_box.setPlaceholderText("Define name...")
        hl.addWidget(label)
        hl.addWidget(self.name_box)

        self.add_button = QtWidgets.QToolButton(None)
        self.all_fits = QtWidgets.QCheckBox()
        self.all_fits.setChecked(False)
        self.add_button.setText("add")
        self.add_button.clicked.connect(lambda: self.onAddFit(all_fits=self.all_fits.isChecked()))
        hl.addWidget(self.add_button)
        self.all_fits.setText('all')
        hl.addWidget(self.all_fits)

        self.fit_list = QtWidgets.QListWidget()
        self.fit_list.doubleClicked.connect(self.onRemoveFit)
        self.layout.addWidget(self.fit_list)

        self.layout_fractions = QtWidgets.QGridLayout()
        self.layout.addLayout(self.layout_fractions)

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
    def onRemoveFit(self):
        """Remove a fit from the mixture."""
        idx = self.fit_list.currentRow()
        if idx != -1:
            self.fit_list.takeItem(idx)
            self.pop_model(idx)
        else:
            logging.warning("Please select an item to remove.")
        self.onUpdateParameterUI()

    # TODO: needs docstring
    def onUpdataFitList(self):
        """Update fit selection combo box."""
        self.cb.clear()
        names = [f.name for f in self.lifetime_fits]
        self.cb.addItems(names)

    # TODO: needs docstring
    def onAddFit(self, all_fits: bool = False):
        """Add selected fit(s) to the mixture."""
        if not all_fits:
            idxs = [self.cb.currentIndex()]
        else:
            idxs = range(0, len(self.lifetime_fits))
        for idx in idxs:
            i = self.fit_list.count() + 1
            f = self.lifetime_fits[idx]
            if len(self.name_box.text()) == 0:
                name = f"x_{i}"
            else:
                name = self.name_box.text()
            self.fit_list.addItem(f'{i}: {f.name}')
            self.append_model(f.model, name)
        self.onUpdateParameterUI()

    # TODO: needs docstring
    def onUpdateParameterUI(self):
        """Rebuild the fraction parameter UI."""
        n_columns, row = 2, 1
        layout = self.layout_fractions
        cs.gui.widgets.general.clear_layout(layout)
        layout.addWidget(QtWidgets.QLabel("Fraction"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Model"), 0, 1)
        for i, (name, fraction) in enumerate(zip(self.model_names, self._fractions)):
            layout.addWidget(
                cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                    fraction,
                    label_text=''
                ),
                row, 0
            )
            layout.addWidget(QtWidgets.QLabel(name), row, 1)
            row += 1
