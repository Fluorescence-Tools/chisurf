from __future__ import annotations

import math
from typing import Optional

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui

import chisurf as cs
import chisurf.core.fitting
from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.core.models.model import ModelCurve
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.gui import plots
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

# Reuse dye database and physical helpers from the FCS calculator plugin
from chisurf.plugins.fcs.fcs_calculator.wizard import (
    DYE_DATA,
    water_viscosity_Pa_s,
    stokes_einstein_D,
    stokes_einstein_rh,
    um2s_to_m2s,
)


def _dye_diffusion_m2_s(dye_name: str, temperature_K: float) -> float:
    """Return diffusion coefficient D(T) in m^2/s for a reference dye.

    Uses the dye's reference diffusion coefficient at 25 b0C in water
    (D25_um2_s) from :data:`DYE_DATA` and applies Einstein-Stokes scaling via
    an intermediate hydrodynamic radius r_h.
    """
    info = DYE_DATA.get(dye_name)
    if not info:
        return float("nan")

    try:
        D25_um2_s = float(info.get("D25_um2_s", float("nan")))
    except Exception:
        return float("nan")
    if not math.isfinite(D25_um2_s) or D25_um2_s <= 0.0:
        return float("nan")

    # Reference diffusion coefficient at 25 b0C (298.15 K) in water
    D25_m2_s = um2s_to_m2s(D25_um2_s)
    T_ref = 298.15
    try:
        eta_ref = water_viscosity_Pa_s(T_ref)
    except Exception:
        return float("nan")

    # Infer hydrodynamic radius from D25 via Stokes-Einstein
    try:
        r_h_m = stokes_einstein_rh(T_ref, eta_ref, D25_m2_s)
    except Exception:
        return float("nan")
    if not math.isfinite(r_h_m) or r_h_m <= 0.0:
        return float("nan")

    # Temperature-dependent viscosity of water and Stokes-Einstein for D(T)
    try:
        eta_T = water_viscosity_Pa_s(float(temperature_K))
    except Exception:
        return float("nan")

    try:
        D_T = stokes_einstein_D(float(temperature_K), eta_T, r_h_m)
    except Exception:
        return float("nan")
    return float(D_T)


class DyeShapeFCSModel(ModelCurve):

    """FCS model with fixed dye diffusion, fitting confocal volume shape.

    The user selects a reference dye from the FCS calculator database. Its
    diffusion coefficient at the experimental temperature is computed via
    Einstein-Stokes using the water viscosity model. The FCS curve is then
    described by a 3D Gaussian model with parameters:

    - N   : particle number
    - s   : structure parameter (wz / wxy)
    - w0  : lateral waist (nm)
    - b   : baseline
    - temp: temperature (K, sets viscosity)

    The objective of the fit is to determine the confocal volume shape
    (s and w0) for a known dye diffusion coefficient.
    """

    name = "FCS dye shape"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs):
        """Initialize the dye-shape FCS model.

        Parameters
        ----------
        fit : cs.core.fitting.fit.Fit
            The fit this model belongs to.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        super().__init__(fit, **kwargs)

        self._N = FittingParameter(
            name="N",
            value=1.0,
            lb=1.0e-6,
            ub=1.0e9,
            bounds_on=False,
            fixed=False,
        )

        self._s = FittingParameter(
            name="s",
            value=3.5,
            lb=0.1,
            ub=20.0,
            bounds_on=False,
            fixed=False,
        )

        # Confocal waist w0 specified in nanometers for the UI.
        self._w0 = FittingParameter(
            name="w0",
            value=350.0,
            lb=10.0,
            ub=5000.0,
            bounds_on=False,
            fixed=False,
        )

        self._b = FittingParameter(
            name="b",
            value=1.0,
            lb=-10.0,
            ub=10.0,
            bounds_on=False,
            fixed=False,
        )

        # Experimental temperature (user-entered, in °C); viscosity is
        # derived from this internally using a kelvin conversion.
        self._temp = FittingParameter(
            name="temp",
            value=20.0,
            lb=0.0,
            ub=100.0,
            bounds_on=False,
            fixed=True,
        )

        # Single global bunching term ba, bt (time in ms).
        self._ba = FittingParameter(
            name="ba",
            value=0.1,
            lb=0.0,
            ub=1.0,
            bounds_on=True,
            fixed=False,
        )

        self._bt = FittingParameter(
            name="bt",
            value=0.002,
            lb=1.0e-6,
            ub=float("inf"),
            bounds_on=False,
            fixed=False,
        )

        # Derived diffusion coefficient D (µm²/s) shown as a fixed parameter.
        self._D = FittingParameter(
            name="D",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        # Derived diffusion time tauD (ms) shown as a fixed parameter.
        self._tauD = FittingParameter(
            name="tauD",
            value=float("nan"),
            lb=0.0,
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        # Derived CPM parameters (counts per molecule), analogous to
        # ParseFCSWidget: cpm (bright molecules) and cpm_all (all molecules
        # including dark/bunching states).
        self._cpm = FittingParameter(
            name="cpm",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        self._cpm_all = FittingParameter(
            name="cpm_all",
            label_text="cpm<sub>all</sub>",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        # Register parameters
        self.find_parameters()

        # Current dye name from database
        self._dye_name: str = ""
        if DYE_DATA:
            try:
                self._dye_name = list(DYE_DATA.keys())[0]
            except Exception:
                self._dye_name = ""

    @property
    def dye_name(self) -> str:
        """Current dye name from the FCS calculator database."""
        return self._dye_name

    @dye_name.setter
    def dye_name(self, name: str) -> None:
        """Set the current dye name.

        Parameters
        ----------
        name : str
            Dye name matching a key in ``DYE_DATA``.
        """
        self._dye_name = str(name)

    def update_model(self, **kwargs) -> None:
        """Compute the FCS curve for the selected dye and confocal volume.

        The model is a 3D Gaussian with one bunching term. Derived
        parameters (``D``, ``tauD``, ``cpm``, ``cpm_all``) are updated
        as fixed output parameters.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()

        if tau.size == 0:
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)
            return

        # Diffusion coefficient of selected dye at experimental temperature.
        # The fitting parameter is specified in °C; convert to kelvin for the
        # physical model.
        temp_C = float(self._temp.value)
        T_K = temp_C + 273.15
        D_m2_s = _dye_diffusion_m2_s(self._dye_name, T_K)
        if not (math.isfinite(D_m2_s) and D_m2_s > 0.0):
            # If D is unavailable, fall back to NaN model curve
            self.x = tau
            self.y = np.full_like(tau, float("nan"))
            return

        # Confocal waist in meters
        w0_nm = float(self._w0.value)
        if not (math.isfinite(w0_nm) and w0_nm > 0.0):
            self.x = tau
            self.y = np.full_like(tau, float("nan"))
            return
        w0_m = w0_nm * 1.0e-9

        # 3D Gaussian: tau_D = w_xy^2 / (4 D); use ms to match t_c units
        tauD_s = (w0_m * w0_m) / (4.0 * D_m2_s)
        tauD_ms = tauD_s * 1.0e3
        if not (math.isfinite(tauD_ms) and tauD_ms > 0.0):
            self.x = tau
            self.y = np.full_like(tau, float("nan"))
            return

        # Update derived diffusion coefficient in µm²/s for display.
        try:
            D_um2_s = D_m2_s * 1.0e12
            fc = get_fitting_client()
            if fc is not None:
                fit_idx = getattr(self.fit, "fit_idx", None)
                fc.set_parameter_value(
                    parameter_name=str(self._D.name), value=D_um2_s, fit_index=fit_idx,
                )
                fc.set_parameter_fixed(
                    parameter_name=str(self._D.name), fixed=True, fit_index=fit_idx,
                )
        except Exception:
            pass

        # Update derived diffusion time in ms for display.
        try:
            fc = get_fitting_client()
            if fc is not None:
                fit_idx = getattr(self.fit, "fit_idx", None)
                fc.set_parameter_value(
                    parameter_name=str(self._tauD.name), value=tauD_ms, fit_index=fit_idx,
                )
                fc.set_parameter_fixed(
                    parameter_name=str(self._tauD.name), fixed=True, fit_index=fit_idx,
                )
        except Exception:
            pass

        N = float(self._N.value)
        s_val = float(self._s.value)
        b_val = float(self._b.value)
        ba_val = float(self._ba.value)
        bt_val = float(self._bt.value)
        if bt_val <= 0.0 or not math.isfinite(bt_val):
            bt_val = 1.0e-6

        t_ms = tau
        with np.errstate(divide="ignore", invalid="ignore"):
            x = t_ms / tauD_ms
            term1 = 1.0 / (1.0 + x)
            term2 = 1.0 / np.sqrt(1.0 + x / (s_val * s_val))
            bunch = 1.0 - ba_val + ba_val * np.exp(-t_ms / bt_val)
            g_fit = b_val + (1.0 / np.abs(N)) * term1 * term2 * bunch

        self.x = tau
        self.y = np.asarray(g_fit, dtype=float)

        # Derived CPM parameters using the same logic as ParseFCSWidget.
        try:
            data = self.fit.data
        except Exception:
            return

        meta = getattr(data, "meta_data", {}) or {}
        mean_cr = meta.get("mean_count_rate")
        if mean_cr is None:
            return

        try:
            N_val = float(self._N.value)
            cr = float(mean_cr)
        except Exception:
            return

        if not (N_val > 0.0):
            return

        # Simple CPM definition: mean count rate per bright molecule.
        cpm = cr / N_val
        try:
            fc = get_fitting_client()
            if fc is not None:
                fit_idx = getattr(self.fit, "fit_idx", None)
                fc.set_parameter_value(
                    parameter_name=str(self._cpm.name), value=cpm, fit_index=fit_idx,
                )
                fc.set_parameter_fixed(
                    parameter_name=str(self._cpm.name), fixed=True, fit_index=fit_idx,
                )
        except Exception:
            pass

        # Second CPM: correct N for dark/bunching states (ba parameters).
        try:
            bunch_sum = 0.0
            for name, p in self.parameters_all_dict.items():
                if not name.startswith("ba"):
                    continue
                try:
                    v = float(p.value)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                v = abs(v)
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                bunch_sum += v

            bright_fraction = 1.0 - bunch_sum
            if bright_fraction <= 0.0 or not np.isfinite(bright_fraction):
                return

            N_all = N_val / bright_fraction
            if not (N_all > 0.0 and np.isfinite(N_all)):
                return

            cpm_all = cr / N_all
            fc = get_fitting_client()
            if fc is not None:
                fit_idx = getattr(self.fit, "fit_idx", None)
                fc.set_parameter_value(
                    parameter_name=str(self._cpm_all.name), value=cpm_all, fit_index=fit_idx,
                )
                fc.set_parameter_fixed(
                    parameter_name=str(self._cpm_all.name), fixed=True, fit_index=fit_idx,
                )
        except Exception:
            pass


class DyeShapeFCSWidget(ModelWidget, DyeShapeFCSModel):

    name = "FCS dye shape"

    plot_classes = [
        (
            plots.LinePlot,
            {
                "scale_x": "log",
                "d_scaley": "lin",
                "r_scaley": "lin",
                "x_label": "t_c (ms)",
                "y_label": "G_c(t_c)",
            },
        ),
        (plots.FitTablePlot, {}),
        (plots.FitInfo, {}),
        (plots.ParameterScanPlot, {}),
        (cs.gui.plots.ResidualPlot, {}),
    ]

    def __init__(
        self,
        fit: cs.core.fitting.fit.FitGroup,
        icon: Optional[QtGui.QIcon] = None,
        **kwargs,
    ):
        """Initialize the dye-shape FCS widget.

        Parameters
        ----------
        fit : cs.core.fitting.fit.FitGroup
            The fit group this widget belongs to.
        icon : QtGui.QIcon, optional
            Icon for the model tab.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")

        # Chain through ModelWidget -> DyeShapeFCSModel -> ModelCurve
        super().__init__(fit=fit, icon=icon, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(0)

        # Dye selection from the FCS calculator database (placed on top and
        # expanding horizontally).
        dye_layout = QtWidgets.QHBoxLayout()
        dye_label = QtWidgets.QLabel("Dye")
        self._dye_combo = QtWidgets.QComboBox()
        self._dye_combo.addItems(list(DYE_DATA.keys()))

        # Initialize combobox to current dye_name if possible
        if self.dye_name:
            idx = self._dye_combo.findText(self.dye_name)
            if idx >= 0:
                self._dye_combo.setCurrentIndex(idx)
        elif self._dye_combo.count() > 0:
            self.dye_name = self._dye_combo.currentText()

        self._dye_combo.currentTextChanged.connect(self._on_dye_changed)

        # Make the combobox expand to width while keeping the label compact.
        dye_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Preferred,
        )
        self._dye_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

        dye_layout.addWidget(dye_label)
        dye_layout.addWidget(self._dye_combo)

        params_layout.addLayout(dye_layout)

        widgets = [
            fitting_widgets.make_fitting_parameter_widget(self._N),
            fitting_widgets.make_fitting_parameter_widget(self._s),
            fitting_widgets.make_fitting_parameter_widget(self._w0, label_text="w<sub>0</sub>", suffix=" nm"),
            fitting_widgets.make_fitting_parameter_widget(self._b),
            fitting_widgets.make_fitting_parameter_widget(self._temp, suffix=" °C"),
            fitting_widgets.make_fitting_parameter_widget(self._ba, label_text="b<sub>a</sub>"),
            fitting_widgets.make_fitting_parameter_widget(self._bt, label_text="b<sub>t</sub>"),
            fitting_widgets.make_fitting_parameter_widget(self._D, suffix=" µm²/s"),
            fitting_widgets.make_fitting_parameter_widget(self._tauD, label_text="&tau;<sub>D</sub>", suffix=" ms"),
            fitting_widgets.make_fitting_parameter_widget(self._cpm),
            fitting_widgets.make_fitting_parameter_widget(self._cpm_all, label_text="cpm<sub>all</sub>"),
        ]
        for w in widgets:
            params_layout.addWidget(w)

        layout.addLayout(params_layout)
        self.setLayout(layout)
        self.layout = layout

    def _on_dye_changed(self, text: str) -> None:
        """Qt slot: update dye selection and trigger model recomputation.

        Parameters
        ----------
        text : str
            New dye name selected in the combo box.
        """
        self.dye_name = text
        try:
            self.update()
        except Exception:
            pass

    def update_widgets(self) -> None:
        """Refresh all fitting-parameter widgets from the current model state."""
        for parameter in self.parameters:
            if hasattr(parameter, 'update') and callable(parameter.update):
                parameter.update()
