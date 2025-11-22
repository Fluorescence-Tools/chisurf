from __future__ import annotations

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui

import chisurf
import chisurf.fitting
from chisurf.models.model import ModelWidget, ModelCurve
from chisurf.fitting.parameter import FittingParameter
from chisurf.models.fcs.maxent import fcs_maxent, fcs_maxent_rh
from chisurf import plots
import chisurf.gui.widgets.fitting.widgets as fitting_widgets


class MaxEntFCSModel(ModelCurve):

    """FCS model that reconstructs a correlation curve via MaxEnt.

    This model treats the regularization parameter of the MaxEnt inversion
    as a (currently fixed) model parameter and computes a reconstructed
    FCS curve `G_fit(τ)` from the experimental FCS data.
    """

    name = "FCS MaxEnt"

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        super().__init__(fit, **kwargs)

        # Regularization strength nu is represented as log10(nu) for stability.
        # IMPORTANT: store the FittingParameter on an attribute name (*_log10_nu*)
        # that is different from its .name ("log10_nu"). This avoids collisions
        # with FittingParameterGroup.__getstate__ / __setstate__, which use the
        # parameter name as a key in the saved state and would otherwise
        # overwrite the attribute with a plain dict on restore.
        self._reg = FittingParameter(
            name="reg",
            value=-1.0,
            lb=-6.0,
            ub=3.0,
            bounds_on=True,
            fixed=True,
        )

        self._td_min = FittingParameter(
            name="td_min",
            value=1.0e-4,
            lb=0.0,
            ub=float("inf"),
            bounds_on=True,
            fixed=True,
        )

        self._td_max = FittingParameter(
            name="td_max",
            value=20.0,
            lb=0.0,
            ub=float("inf"),
            bounds_on=True,
            fixed=True,
        )

        self._n_td = FittingParameter(
            name="n_td",
            value=64.0,
            lb=10.0,
            ub=400.0,
            bounds_on=True,
            fixed=True,
        )

        self._s = FittingParameter(
            name="s",
            value=3.5,
            lb=0.1,
            ub=20.0,
            bounds_on=True,
            fixed=True,
        )

        self._b = FittingParameter(
            name="b",
            value=1.0,
            lb=-10.0,
            ub=10.0,
            bounds_on=True,
            fixed=True,
        )

        # Register parameters with the base machinery so that _log10_nu
        # appears in parameters_all_dict under the key "log10_nu".
        self.find_parameters()

        self._result = None

    @property
    def last_result(self) -> dict | None:
        return self._result

    @property
    def maxent_tauD_distribution(self):
        """Return the MaxEnt diffusion-time distribution as (y, x) = (p, td_grid).

        This is used by chisurf.plots.DistributionPlot via the
        "maxent_tauD_distribution" attribute.
        """
        if self._result is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        td_grid = np.asarray(self._result["td_grid"], dtype=float)
        p = np.asarray(self._result["p"], dtype=float)
        return p, td_grid

    def update_model(self, **kwargs) -> None:
        """Run MaxEnt on the current FCS dataset and update the model curve.

        The experimental data are taken from ``self.fit.data`` (x = lag times,
        y = correlation amplitudes). The MaxEnt inversion is run on the
        experimental curve, and the reconstructed curve is stored in
        ``self.y``, while the distribution and other details are stored in
        ``self._result``.
        """
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()

        if tau.size == 0 or g.size == 0:
            # Nothing to do if data are empty
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)
            self._result = None
            return

        # Optional weights derived from ey if available (ey ~ sigma_y)
        weights = None
        ey = getattr(data, "ey", None)
        if ey is not None:
            ey_arr = np.asarray(ey, dtype=float).ravel()
            if ey_arr.size == g.size:
                weights = 1.0 / np.maximum(ey_arr, 1e-12)

        # Regularization strength nu (10**log10_nu)
        nu = 10.0 ** float(self._reg.value)

        if float(self._td_min.value) > 0.0:
            td_min = float(self._td_min.value)
        else:
            td_min = None

        if float(self._td_max.value) > 0.0:
            td_max = float(self._td_max.value)
        else:
            td_max = None

        if float(self._n_td.value) > 0.0:
            n_td = int(self._n_td.value)
        else:
            n_td = 80

        if float(self._s.value) > 0.0:
            s_val = float(self._s.value)
        else:
            s_val = 3.5

        b_val = float(self._b.value)

        self._result = fcs_maxent(
            tau=tau,
            g=g,
            td_min=td_min,
            td_max=td_max,
            n_td=n_td,
            s=s_val,
            baseline=b_val,
            reg=nu,
            weights=weights,
        )

        # Update underlying model curve for fitting / residual plots
        self.x = self._result["tau"]
        self.y = self._result["g_fit"]


class MaxEntFCSWidget(ModelWidget, MaxEntFCSModel):

    """GUI model widget for MaxEnt-based FCS analysis.

    This appears as a separate FCS model in the GUI and uses the MaxEnt
    inversion to reconstruct the FCS curve and an underlying diffusion-time
    distribution. A simple control for the regularization parameter and a
    button to plot the MaxEnt result are provided.
    """

    name = "FCS MaxEnt"

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
        (
            chisurf.plots.DistributionPlot,
            {
                "distribution_options": {
                    "MaxEnt tau_D": {
                        "attribute": "maxent_tauD_distribution",
                        "accessor": lambda x, **kwargs: x,
                        "accessor_kwargs": {"sort": False},
                        "curve_options": {
                            "stepMode": False,
                            "connect": False,
                            "symbol": "o",
                            "multi_curve": False,
                        },
                    }
                },
                "scale_x": "log",
            },
        ),
        (chisurf.plots.ResidualPlot, {}),
    ]

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs,
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")

        # This will chain through ModelWidget -> MaxEntFCSModel -> ModelCurve
        super().__init__(fit=fit, icon=icon, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Regularization parameter control
        params_layout = QtWidgets.QVBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(0)

        widgets = [
            fitting_widgets.make_fitting_parameter_widget(self._reg),
            fitting_widgets.make_fitting_parameter_widget(self._td_min),
            fitting_widgets.make_fitting_parameter_widget(self._td_max),
            fitting_widgets.make_fitting_parameter_widget(self._n_td),
            fitting_widgets.make_fitting_parameter_widget(self._s),
            fitting_widgets.make_fitting_parameter_widget(self._b),
        ]
        for w in widgets:
            params_layout.addWidget(w)

        layout.addLayout(params_layout)
        self.setLayout(layout)
        self.layout = layout


class MaxEntRHModel(ModelCurve):

    name = "FCS MaxEnt rH"

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        super().__init__(fit, **kwargs)

        self._reg = FittingParameter(
            name="reg",
            value=-1.0,
            lb=-6.0,
            ub=3.0,
            bounds_on=True,
            fixed=True,
        )

        self._rh_min = FittingParameter(
            name="rh_min",
            value=0.01,
            lb=0.001,
            ub=1000.0,
            bounds_on=True,
            fixed=True,
        )

        self._rh_max = FittingParameter(
            name="rh_max",
            value=50.0,
            lb=0.01,
            ub=1000.0,
            bounds_on=True,
            fixed=True,
        )

        self._n_rh = FittingParameter(
            name="n_rh",
            value=128.0,
            lb=10.0,
            ub=400.0,
            bounds_on=True,
            fixed=True,
        )

        self._s = FittingParameter(
            name="s",
            value=3.5,
            lb=0.1,
            ub=20.0,
            bounds_on=True,
            fixed=True,
        )

        self._b = FittingParameter(
            name="b",
            value=1.0,
            lb=-10.0,
            ub=10.0,
            bounds_on=True,
            fixed=True,
        )

        # Confocal waist w0 specified in nanometers for the UI. Internally we
        # convert to micrometers when calling the MaxEnt solver.
        self._w0 = FittingParameter(
            name="w0",
            value=350.0,
            lb=10.0,
            ub=5000.0,
            bounds_on=True,
            fixed=True,
        )

        self._temp = FittingParameter(
            name="temp",
            value=20.0,
            lb=-50.0,
            ub=200.0,
            bounds_on=True,
            fixed=True,
        )

        self.find_parameters()
        self._result = None

    @property
    def last_result(self) -> dict | None:
        return self._result

    @property
    def maxent_rH_distribution(self):
        if self._result is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        rh_grid = np.asarray(self._result["rh_grid"], dtype=float)
        p = np.asarray(self._result["p"], dtype=float)
        return p, rh_grid

    def update_model(self, **kwargs) -> None:
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()

        if tau.size == 0 or g.size == 0:
            self.x = np.array([], dtype=float)
            self.y = np.array([], dtype=float)
            self._result = None
            return

        weights = None
        ey = getattr(data, "ey", None)
        if ey is not None:
            ey_arr = np.asarray(ey, dtype=float).ravel()
            if ey_arr.size == g.size:
                weights = 1.0 / np.maximum(ey_arr, 1e-12)

        nu = 10.0 ** float(self._reg.value)

        rh_min = max(float(self._rh_min.value), 1.0e-3)
        rh_max = max(float(self._rh_max.value), rh_min * 1.001)
        if float(self._n_rh.value) > 0.0:
            n_rh = int(self._n_rh.value)
        else:
            n_rh = 80

        if float(self._s.value) > 0.0:
            s_val = float(self._s.value)
        else:
            s_val = 3.5

        b_val = float(self._b.value)
        # Convert w0 from nm (parameter value) to µm for the solver
        w0_val = float(self._w0.value) * 1.0e-3
        # User parameter is temperature in °C; convert to K for the solver
        temp_C = float(self._temp.value)
        temp_val = temp_C + 273.15

        self._result = fcs_maxent_rh(
            tau=tau,
            g=g,
            rh_min=rh_min,
            rh_max=rh_max,
            n_rh=n_rh,
            w0=w0_val,
            s=s_val,
            baseline=b_val,
            reg=nu,
            temperature=temp_val,
            weights=weights,
        )

        self.x = self._result["tau"]
        self.y = self._result["g_fit"]


class MaxEntRHWidget(ModelWidget, MaxEntRHModel):

    name = "FCS MaxEnt rH"

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
        (
            chisurf.plots.DistributionPlot,
            {
                "distribution_options": {
                    "MaxEnt rH": {
                        "attribute": "maxent_rH_distribution",
                        "accessor": lambda x, **kwargs: x,
                        "accessor_kwargs": {"sort": False},
                        "curve_options": {
                            "stepMode": False,
                            "connect": False,
                            "symbol": "o",
                            "multi_curve": False,
                        },
                    }
                }
            },
        ),
        (chisurf.plots.ResidualPlot, {}),
    ]

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs,
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")

        super().__init__(fit=fit, icon=icon, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(0)

        widgets = [
            fitting_widgets.make_fitting_parameter_widget(self._reg),
            fitting_widgets.make_fitting_parameter_widget(self._rh_min, suffix=" nm"),
            fitting_widgets.make_fitting_parameter_widget(self._rh_max, suffix=" nm"),
            fitting_widgets.make_fitting_parameter_widget(self._n_rh),
            fitting_widgets.make_fitting_parameter_widget(self._s),
            fitting_widgets.make_fitting_parameter_widget(self._b),
            fitting_widgets.make_fitting_parameter_widget(self._w0, suffix=" nm"),
            fitting_widgets.make_fitting_parameter_widget(self._temp, suffix=" °C"),
        ]
        for w in widgets:
            params_layout.addWidget(w)

        layout.addLayout(params_layout)
        self.setLayout(layout)
        self.layout = layout
