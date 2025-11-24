from __future__ import annotations
import typing
from typing import TYPE_CHECKING

import numpy as np

import chisurf.gui.widgets.fitting
import chisurf.plots
from chisurf.fluorescence.general import distance_to_fret_efficiency
from chisurf.math.functions import distributions as distfuncs
from chisurf.models.tcspc.fret import rda_axis

from chisurf.models.model import ModelWidget
from chisurf.gui import QtWidgets, QtGui, QtCore
from chisurf.models.pda.nusiance import Background, PdaFretNuisance
from chisurf.models.pda.simple import ProbCh0, PdaSimpleModel, PdaGaussianDistances, PdaGaussianDistanceModel

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

        add_component = QtWidgets.QPushButton()
        add_component.setText("add")
        add_component.clicked.connect(self.onAddComponent)
        lh.addWidget(add_component)

        remove_component = QtWidgets.QPushButton()
        remove_component.setText("del")
        remove_component.clicked.connect(self.onRemoveComponent)
        lh.addWidget(remove_component)

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
    s1s2_experimental = fit.data.pda['s1s2']
    model_x, model_y = pda.get_1dhistogram(
        s1s2=pda.get_S1S2_matrix().flatten(),
        **kw_hist
    )
    data_x, data_y = pda.get_1dhistogram(
        s1s2=s1s2_experimental.flatten(),
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
                            'pen': ['b', 'r', 'k']
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
                            'pen': ['b', 'r', 'k']
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

        background = BackgroundWidget(fit=fit, **kwargs)
        pch0 = ProbCh0Widget(fit=fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        ## add widgets
        if not hide_nuisances:
            layout.addWidget(background)
        layout.addWidget(pch0)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.background = background
        self.pch0 = pch0


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
                            'pen': ['b', 'r', 'k', 'g', 'm', 'c', 'y']
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
                            'pen': ['b', 'r', 'k', 'g', 'm', 'c', 'y']
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
        distances = PdaGaussianDistancesWidget(fit=fit, **kwargs)

        # Generic widget for the shared FRET parameter group (R0, tau0, kappa2, ...)
        fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        if not hide_nuisances:
            layout.addWidget(nuisance)
        layout.addWidget(fret_parameters_widget)
        layout.addWidget(distances)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.nuisance = nuisance
        self.distances = distances
        self.fret_parameters_widget = fret_parameters_widget
