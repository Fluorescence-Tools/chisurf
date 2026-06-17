from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import chisurf as cs
from chisurf import typing
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.general
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
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
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
        """Handle add lifetime button click."""
        self.append()

    # TODO: needs docstring
    def onRemoveLifetime(self):
        """Handle remove lifetime button click."""
        if len(self._lifetimes) > 1:
            self.pop()

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

    def _tcspc_reference_window(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> np.ndarray:
        """Return the y-window used for photon normalization.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        numpy.ndarray
            Finite y-values used for the denominator.
        """
        y = np.asarray(context.y, dtype=float)
        if not bool(context.parameters.get("fit_range_only", False)):
            return y[np.isfinite(y)]
        try:
            data_x = np.asarray(getattr(getattr(context.fit, "data", None), "x", []), dtype=float)
            if y.size == data_x.size:
                xmin = int(getattr(context.fit, "xmin", 0))
                xmax = int(getattr(context.fit, "xmax", y.size))
                y = y[max(0, xmin):min(y.size, xmax)]
        except Exception:
            pass
        return y[np.isfinite(y)]

    def _tcspc_total_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by total photons.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Photon-normalized curve.
        """
        window = self._tcspc_reference_window(context)
        denominator = float(np.nansum(window))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("total photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / total photons",
        )

    def _tcspc_peak_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by the peak photon count.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Peak-normalized curve.
        """
        window = self._tcspc_reference_window(context)
        if window.size == 0:
            raise ValueError("peak photon count is unavailable")
        denominator = float(np.nanmax(window))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("peak photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / peak photons",
        )

    def _donor_reference_curve(self, context: plot_transforms.PlotReferenceContext) -> np.ndarray:
        """Return a donor-reference curve for the current context.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        numpy.ndarray
            Reference curve.
        """
        raw_ref = None
        ref_model = getattr(self, "_reference", None)
        if ref_model is not None:
            try:
                ref_model.update_model()
                raw_ref = np.maximum(np.asarray(ref_model.y, dtype=float), 0.0)
            except Exception:
                raw_ref = None
        if raw_ref is None:
            raw_ref = np.asarray(getattr(self, "reference"), dtype=float)

        scale_mode = str(context.parameters.get("scale", "data_peak"))
        ref = np.asarray(raw_ref, dtype=float).copy()
        peak = float(np.nanmax(ref)) if ref.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("donor reference peak is unavailable")
        if scale_mode == "data_peak":
            y_peak = float(np.nanmax(np.asarray(context.y, dtype=float)))
            if np.isfinite(y_peak) and y_peak > 0.0:
                ref *= y_peak / peak
        elif scale_mode == "reference_peak":
            ref /= peak
        return ref

    def _tcspc_donor_reference_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC FRET curves by donor-reference decay.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Donor-reference-normalized curve.
        """
        ref = self._donor_reference_curve(context)
        y = np.asarray(context.y, dtype=float)
        x = np.asarray(context.x, dtype=float)
        n = min(y.size, ref.size, x.size)
        if n <= 0:
            raise ValueError("donor reference length is zero")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(ref[:n]) > 1e-15, y[:n] / ref[:n], np.nan)
        return plot_transforms.PlotReferenceResult(
            x=x[:n],
            y=out,
            y_label="counts / donor reference",
        )

    def _plot_anisotropy_widget(self):
        """Return the anisotropy component used for r(t) plotting.

        Returns
        -------
        object or None
            Anisotropy widget/component.
        """
        aniso = getattr(self, "anisotropy", None)
        if aniso is not None:
            return aniso
        for name in ("fret_rates", "fret", "distance_distribution"):
            candidate = getattr(getattr(self, name, None), "anisotropy", None)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _tcspc_rt_curves(t, vv, vh, g: float, l1: float, l2: float):
        """Compute uncorrected and corrected anisotropy curves.

        Parameters
        ----------
        t : array_like
            Time axis.
        vv : array_like
            Parallel channel.
        vh : array_like
            Perpendicular channel.
        g : float
            G-factor.
        l1 : float
            Leakage correction l1.
        l2 : float
            Leakage correction l2.

        Returns
        -------
        tuple
            ``(t, r_uncorrected, r_corrected)``.
        """
        t = np.asarray(t, dtype=float)
        vv = np.asarray(vv, dtype=float)
        vh = np.asarray(vh, dtype=float)
        det = (1.0 - l1) * (1.0 - l2) - l1 * l2
        if abs(det) < 1e-12:
            raise ValueError("anisotropy leakage correction is singular")
        den_unc = g * vv + 2.0 * vh
        with np.errstate(divide="ignore", invalid="ignore"):
            r_unc = np.where(np.abs(den_unc) > 1e-12, (vv - vh) / den_unc, np.nan)
        vv_u = ((1.0 - l2) * vv - l1 * vh) / det
        vh_u = (-l2 * vv + (1.0 - l1) * vh) / det
        den_cor = g * vv_u + 2.0 * vh_u
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cor = np.where(np.abs(den_cor) > 1e-12, (vv_u - vh_u) / den_cor, np.nan)
        finite = np.isfinite(t) & np.isfinite(r_unc) & np.isfinite(r_cor)
        if np.any(finite):
            return t[finite], r_unc[finite], r_cor[finite]
        return t, r_unc, r_cor

    def _tcspc_anisotropy_rt_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Plot time-resolved anisotropy from VV/VH channels.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Anisotropy curve, or hidden result for non-primary group members.
        """
        if context.curve_key not in ("data", "model"):
            return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)
        if context.group_index is not None and context.selected_group_index is not None:
            if int(context.group_index) != int(context.selected_group_index):
                return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)

        aniso = self._plot_anisotropy_widget()
        if aniso is None:
            raise ValueError("anisotropy component is unavailable")

        if context.curve_key == "model":
            t, vv, vh = aniso._extract_vv_vh_model_for_diag()
        else:
            t, vv, vh, _defaults = aniso._extract_vv_vh_raw_for_diag()
        if t is None or vv is None or vh is None:
            return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)

        vv = np.asarray(vv, dtype=float) - float(context.parameters.get("bg_vv", 0.0))
        vh = np.asarray(vh, dtype=float) - float(context.parameters.get("bg_vh", 0.0))
        shift = float(context.parameters.get("vh_shift", 0.0))
        if hasattr(aniso, "_shift_trace_to_reference"):
            vh = aniso._shift_trace_to_reference(np.asarray(t, dtype=float), vh, shift)

        tt, r_unc, r_cor = self._tcspc_rt_curves(
            t=t,
            vv=vv,
            vh=vh,
            g=float(context.parameters.get("g", getattr(aniso, "g", 1.0))),
            l1=float(context.parameters.get("l1", getattr(aniso, "l1", 0.0))),
            l2=float(context.parameters.get("l2", getattr(aniso, "l2", 0.0))),
        )
        variant = str(context.parameters.get("variant", "corrected"))
        y = r_unc if variant == "uncorrected" else r_cor
        return plot_transforms.PlotReferenceResult(x=tt, y=y, y_label="r(t)")

    def _tcspc_anisotropy_defaults(self) -> typing.Dict[str, float]:
        """Return default plot-only anisotropy parameters.

        Returns
        -------
        dict
            Defaults for r(t) correction controls.
        """
        aniso = self._plot_anisotropy_widget()
        defaults = {
            "g": 1.0,
            "l1": 0.0,
            "l2": 0.0,
            "bg_vv": 0.0,
            "bg_vh": 0.0,
            "vh_shift": 0.0,
        }
        if aniso is None:
            return defaults
        for key in ("g", "l1", "l2"):
            try:
                defaults[key] = float(getattr(aniso, key))
            except Exception:
                pass
        try:
            _t, _vv, _vh, diag_defaults = aniso._extract_vv_vh_raw_for_diag()
            if isinstance(diag_defaults, dict):
                defaults["bg_vv"] = float(diag_defaults.get("bg_vv", defaults["bg_vv"]))
                defaults["bg_vh"] = float(diag_defaults.get("bg_vh", defaults["bg_vh"]))
                defaults["vh_shift"] = float(diag_defaults.get("shift_vh", 0.0)) - float(diag_defaults.get("shift_vv", 0.0))
        except Exception:
            pass
        return defaults

    def get_plot_reference_modes(self) -> typing.List[plot_transforms.PlotReferenceMode]:
        """Return TCSPC reference modes for the line plot.

        Returns
        -------
        list
            Plot reference modes.
        """
        fit_range_param = plot_transforms.PlotReferenceParameter(
            key="fit_range_only",
            label="fit range",
            kind="bool",
            default=False,
        )
        modes = [
            plot_transforms.PlotReferenceMode(
                key="tcspc_total_photons",
                label="Total photons",
                callback=self._tcspc_total_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / total photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
            plot_transforms.PlotReferenceMode(
                key="tcspc_peak_photons",
                label="Peak photons",
                callback=self._tcspc_peak_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / peak photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
        ]
        if hasattr(self, "_reference") or hasattr(type(self), "reference"):
            modes.append(
                plot_transforms.PlotReferenceMode(
                    key="tcspc_donor_reference",
                    label="Donor reference",
                    callback=self._tcspc_donor_reference_mode,
                    parameters=(
                        plot_transforms.PlotReferenceParameter(
                            key="scale",
                            label="scale",
                            kind="choice",
                            default="data_peak",
                            choices=(
                                ("data_peak", "data peak"),
                                ("reference_peak", "reference peak"),
                                ("none", "none"),
                            ),
                        ),
                    ),
                    applies_to=("data", "model"),
                    y_label="counts / donor reference",
                    y_range=(0, 1.0),
                    y_padding=0.05,
                )
            )
        if self._plot_anisotropy_widget() is not None:
            defaults = self._tcspc_anisotropy_defaults()
            modes.append(
                plot_transforms.PlotReferenceMode(
                    key="tcspc_anisotropy_rt",
                    label="r(t) anisotropy",
                    callback=self._tcspc_anisotropy_rt_mode,
                    parameters=(
                        plot_transforms.PlotReferenceParameter("g", "g", "float", defaults["g"], step=0.01),
                        plot_transforms.PlotReferenceParameter("l1", "l1", "float", defaults["l1"], step=0.001),
                        plot_transforms.PlotReferenceParameter("l2", "l2", "float", defaults["l2"], step=0.001),
                        plot_transforms.PlotReferenceParameter("bg_vv", "BgVV", "float", defaults["bg_vv"], step=1.0),
                        plot_transforms.PlotReferenceParameter("bg_vh", "BgVH", "float", defaults["bg_vh"], step=1.0),
                        plot_transforms.PlotReferenceParameter("vh_shift", "dVH", "float", defaults["vh_shift"], step=0.01),
                        plot_transforms.PlotReferenceParameter(
                            "variant",
                            "variant",
                            "choice",
                            "corrected",
                            choices=(("corrected", "corrected"), ("uncorrected", "uncorrected")),
                        ),
                    ),
                    applies_to=("data", "model"),
                    y_label="r(t)",
                    y_range=(-0.05, 0.45),
                    y_padding=0.0,
                )
            )
        return modes


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
        for i, (name, fraction) in enumerate(zip(self.model_names, self.fractions)):
            layout.addWidget(
                cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                    fraction,
                    label_text=''
                ),
                row, 0
            )
            layout.addWidget(QtWidgets.QLabel(name), row, 1)
            row += 1
