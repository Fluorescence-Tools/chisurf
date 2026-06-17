from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.fitting
import chisurf.core.math.regularization
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.models.fcs.maxent import (
    compute_fcs_maxent_l_curve,
    compute_fcs_maxent_rh_l_curve,
    fcs_maxent,
    fcs_maxent_rh,
)
from chisurf.core.models.model import ModelCurve
from chisurf.gui import plots
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client
from chisurf.gui.widgets.models.model_widget import ModelWidget


class MaxEntFCSModel(ModelCurve):
    """FCS model that reconstructs a correlation curve via MaxEnt.

    This model treats the regularization parameter of the MaxEnt inversion
    as a (currently fixed) model parameter and computes a reconstructed
    FCS curve `G_fit(τ)` from the experimental FCS data.
    """

    name = "FCS MaxEnt"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs):
        """Initialize the MaxEnt FCS model.

        Parameters
        ----------
        fit : cs.core.fitting.fit.Fit
            The fit this model belongs to.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        super().__init__(fit, **kwargs)

        # Regularization strength nu is represented as log10(nu) for stability.
        # IMPORTANT: store the FittingParameter on an attribute name (*_log10_nu*)
        # that is different from its .name ("log10_nu"). This avoids collisions
        # with FittingParameterGroup.__getstate__ / __setstate__, which use the
        # parameter name as a key in the saved state and would otherwise
        # overwrite the attribute with a plain dict on restore.
        self._reg = FittingParameter(
            name="reg",
            value=-3.0,
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
        self._l_curve_log10_reg = None
        self._l_curve_chi2 = None
        self._l_curve_solution_norm = None
        self._l_curve_corner_index = None

    @property
    def last_result(self) -> dict | None:
        """Return the last MaxEnt result dictionary or ``None``.

        The structure matches the output of
        :func:`cs.core.models.fcs.maxent.fcs_maxent` and contains the
        reconstructed curve, diffusion-time grid and MaxEnt distribution.
        """
        return self._result

    @property
    def maxent_tauD_distribution(self):
        """Return the MaxEnt diffusion-time distribution as (y, x) = (p, td_grid).

        This is used by cs.gui.plots.DistributionPlot via the
        "maxent_tauD_distribution" attribute.
        """
        if self._result is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        td_grid = np.asarray(self._result["td_grid"], dtype=float)
        p = np.asarray(self._result["p"], dtype=float)
        return p, td_grid

    @property
    def l_curve_log10_reg(self):
        """L-curve log10(reg) grid points.

        Returns
        -------
        np.ndarray
            1-D array of log10(regularization) values, or empty.
        """
        if self._l_curve_log10_reg is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_log10_reg, dtype=float)

    @property
    def l_curve_reg(self):
        """L-curve regularization values (linear scale).

        Returns
        -------
        np.ndarray
            1-D array of regularization values, or empty.
        """
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return vals
        return 10.0 ** vals

    @property
    def l_curve_chi2(self):
        r"""L-curve chi-squared values per regularization point.

        Returns
        -------
        np.ndarray
            1-D array of :math:`\\chi^2_r` values, or empty.
        """
        if self._l_curve_chi2 is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_chi2, dtype=float)

    @property
    def l_curve_solution_norm(self):
        """L-curve solution norm (|p|) per regularization point.

        Returns
        -------
        np.ndarray
            1-D array of solution norms, or empty.
        """
        if self._l_curve_solution_norm is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_solution_norm, dtype=float)

    def compute_l_curve(
            self,
            n_points: int = 32,
            log10_min: float | None = None,
            log10_max: float | None = None,
    ) -> None:
        """Sweep the regularization parameter and build the L-curve.

        The numerical sweep is performed by
        :func:`chisurf.core.models.fcs.maxent.compute_fcs_maxent_l_curve`;
        this method only adapts the current model parameters and fit-window
        settings to the core helper.

        Parameters
        ----------
        n_points : int, optional
            Number of grid points.
        log10_min, log10_max : float, optional
            log10(reg) range. Defaults to current value ± 2.
        """
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()
        if tau.size == 0 or g.size == 0:
            self._l_curve_log10_reg = np.array([], dtype=float)
            self._l_curve_chi2 = np.array([], dtype=float)
            self._l_curve_solution_norm = np.array([], dtype=float)
            self._l_curve_corner_index = None
            return

        if n_points is None or int(n_points) <= 1:
            n_points = 2
        current_log10 = float(self._reg.value)
        try:
            lb, ub = [float(b) for b in self._reg.bounds]
        except Exception:
            lb, ub = float("-inf"), float("inf")
        if log10_min is None:
            log10_min = max(lb, current_log10 - 2.0)
        if log10_max is None:
            log10_max = min(ub, current_log10 + 2.0)
        if log10_min > log10_max:
            log10_min, log10_max = log10_max, log10_min

        result = compute_fcs_maxent_l_curve(
            tau=tau,
            g=g,
            y_error=getattr(data, "ey", None),
            log10_min=float(log10_min),
            log10_max=float(log10_max),
            n_points=int(n_points),
            xmin=getattr(self.fit, "xmin", 0),
            xmax=getattr(self.fit, "xmax", None),
            mask=getattr(self.fit, "mask", None),
            n_free=int(self.n_free),
            td_min=float(self._td_min.value) if float(self._td_min.value) > 0.0 else None,
            td_max=float(self._td_max.value) if float(self._td_max.value) > 0.0 else None,
            n_td=int(self._n_td.value) if int(self._n_td.value) > 0 else 80,
            s=float(self._s.value),
            baseline=float(self._b.value),
            num_iter=60,
        )
        self._l_curve_log10_reg = result.log10_reg
        self._l_curve_chi2 = result.chi2r
        self._l_curve_solution_norm = result.solution_norm
        self._l_curve_corner_index = result.corner_index
        self.update_model()

    def l_curve_corner_index(self) -> int | None:
        """Return index of the automatically detected L-curve corner.

        The index refers to the *global* L-curve arrays stored on the
        model (``l_curve_log10_reg``, ``l_curve_chi2``,
        ``l_curve_solution_norm``).  ``None`` is returned if no suitable
        corner can be determined.
        """
        if self._l_curve_corner_index is not None:
            return int(self._l_curve_corner_index)
        rho = self.l_curve_chi2
        eta = self.l_curve_solution_norm
        if rho.size < 3 or eta.size < 3:
            return None
        idx = cs.core.math.regularization.discrete_lcurve_corner(rho, eta)
        if idx is not None:
            self._l_curve_corner_index = int(idx)
        return int(idx) if idx is not None else None

    def set_reg_from_lcurve_index(self, idx: int) -> None:
        """Set the regularization parameter from an L-curve index.

        Parameters
        ----------
        idx : int
            Index into the global L-curve arrays.
        """
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return
        if idx < 0 or idx >= vals.size:
            return
        new_log10 = float(vals[idx])
        self._reg.value = new_log10
        fc = get_fitting_client()
        if fc is not None:
            fc.set_parameter_value(
                parameter_name=str(self._reg.name),
                value=new_log10,
                fit_index=getattr(self.fit, "fit_idx", None),
            )
        update = getattr(self, "update", None)
        if callable(update):
            try:
                update()
                return
            except Exception:
                pass
        try:
            self.update_model()
        except Exception:
            pass

    def on_auto_fit_range_completed(self) -> None:
        """Run an automatic L-curve sweep after an auto-fit range sweep."""
        compute_l = getattr(self, "compute_l_curve", None)
        corner_fn = getattr(self, "l_curve_corner_index", None)
        set_from_idx = getattr(self, "set_reg_from_lcurve_index", None)
        if not (callable(compute_l) and callable(corner_fn) and callable(set_from_idx)):
            return
        try:
            compute_l(n_points=32, log10_min=-3.0, log10_max=3.0)
            idx = corner_fn()
            if idx is not None:
                set_from_idx(int(idx))
        except Exception as e:
            try:
                cs.logging.warning(
                    f"MaxEntFCSModel.on_auto_fit_range_completed: auto L-curve sweep failed: {e}"
                )
            except Exception:
                pass

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


class MaxEntFCSLCurveController(QtWidgets.QWidget):
    """Controller widget for the FCS MaxEnt L-curve plot."""

    def __init__(self, parent_plot):
        """Initialize the L-curve controller widget.

        Parameters
        ----------
        parent_plot : MaxEntFCSLCurvePlot
            The L-curve plot this controller belongs to.
        """
        super().__init__(parent_plot)
        self._plot = parent_plot
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Axis scale controls
        scale_layout = QtWidgets.QHBoxLayout()
        self._cb_logx = QtWidgets.QCheckBox("log Chi2r")
        self._cb_logy = QtWidgets.QCheckBox("log |p|")
        self._cb_logx.setChecked(False)
        self._cb_logy.setChecked(False)
        self._cb_logx.stateChanged.connect(self._on_scale_changed)
        self._cb_logy.stateChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self._cb_logx)
        scale_layout.addWidget(self._cb_logy)
        scale_layout.addStretch(1)
        layout.addLayout(scale_layout)

        # L-curve range controls: log10_min, log10_max, n_points
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.setSpacing(4)

        self._sb_logmin = QtWidgets.QDoubleSpinBox()
        self._sb_logmin.setDecimals(2)
        self._sb_logmin.setSingleStep(0.5)
        self._sb_logmin.setRange(-6.0, 6.0)
        self._sb_logmin.setValue(-3.0)

        self._sb_logmax = QtWidgets.QDoubleSpinBox()
        self._sb_logmax.setDecimals(2)
        self._sb_logmax.setSingleStep(0.5)
        self._sb_logmax.setRange(-6.0, 6.0)
        self._sb_logmax.setValue(0.0)

        self._sb_npoints = QtWidgets.QSpinBox()
        self._sb_npoints.setRange(2, 256)
        self._sb_npoints.setSingleStep(2)
        self._sb_npoints.setValue(32)

        range_layout.addWidget(QtWidgets.QLabel("min"))
        range_layout.addWidget(self._sb_logmin)
        range_layout.addWidget(QtWidgets.QLabel("max"))
        range_layout.addWidget(self._sb_logmax)
        range_layout.addWidget(QtWidgets.QLabel("N"))
        range_layout.addWidget(self._sb_npoints)

        self._btn_compute = QtWidgets.QPushButton("Compute L-Curve")
        self._btn_compute.clicked.connect(self._on_compute_clicked)
        range_layout.addWidget(self._btn_compute)
        range_layout.addStretch(1)

        layout.addLayout(range_layout)

    def _on_scale_changed(self, *args):
        """Qt slot: update log scale mode when checkboxes change."""
        self._plot.set_log_mode(
            logx=self._cb_logx.isChecked(),
            logy=self._cb_logy.isChecked(),
        )

    @property
    def log10_min(self) -> float | None:
        """Minimum log10(reg) for the L-curve sweep."""
        return float(self._sb_logmin.value())

    @property
    def log10_max(self) -> float | None:
        """Maximum log10(reg) for the L-curve sweep."""
        return float(self._sb_logmax.value())

    @property
    def n_points(self) -> int:
        """Number of L-curve sweep points."""
        return int(self._sb_npoints.value())

    def _on_compute_clicked(self, *args):
        """Qt slot: trigger L-curve computation with current settings."""
        try:
            self._plot.update_all()
        except Exception:
            pass


class MaxEntFCSLCurvePlot(plots.Plot):
    """Interactive L-curve plot for FCS MaxEnt regularization selection."""

    name = "L-Curve"

    def __init__(self, fit: cs.core.fitting.fit.FitGroup, **kwargs):
        """Initialize the L-curve plot widget.

        Parameters
        ----------
        fit : cs.core.fitting.fit.FitGroup
            The fit group whose model provides L-curve data.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        super().__init__(fit=fit, **kwargs)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.getPlotItem().setMouseEnabled(x=True, y=True)
        self.layout.addWidget(self._plot_widget)

        self._curve = self._plot_widget.plot(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
            symbol="o",
            symbolSize=8,
            symbolBrush="#2f80ed",
            symbolPen="w",
        )
        self._selected_point = self._plot_widget.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolBrush="r",
            symbolPen="r",
            symbolSize=16,
        )
        try:
            self._selected_point.setExportHint(False)
        except Exception:
            pass
        self._corner_point = self._plot_widget.plot(
            [],
            [],
            pen=None,
            symbol="x",
            symbolBrush="#ffd166",
            symbolPen="#ffd166",
            symbolSize=14,
        )
        try:
            self._corner_point.setExportHint(False)
        except Exception:
            pass
        self._selected_point.hide()
        self._corner_point.hide()

        self._plot_widget.setLabel("bottom", "Chi2r")
        self._plot_widget.setLabel("left", "|p|")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._logx = False
        self._logy = False
        self._last_indices = np.array([], dtype=int)
        self._selected_global_index: int | None = None
        self.set_log_mode(self._logx, self._logy)
        self.plot_controller = MaxEntFCSLCurveController(self)

        try:
            self._curve.sigPointsClicked.connect(self._on_points_clicked)
        except Exception:
            pass
        try:
            self._plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)
        except Exception:
            pass

    def update(self, *args, **kwargs) -> None:
        """Refresh the L-curve plot from cached model data."""
        self.refresh_from_model(auto_select=False)

    def set_log_mode(self, logx: bool, logy: bool) -> None:
        """Set log scale mode for the L-curve axes.

        Parameters
        ----------
        logx : bool
            If True, use log scale for the x-axis (Chi2r).
        logy : bool
            If True, use log scale for the y-axis (|p|).
        """
        self._logx = bool(logx)
        self._logy = bool(logy)
        self._plot_widget.setLogMode(x=self._logx, y=self._logy)
        try:
            x_data, y_data = self._curve.getData()
        except Exception:
            x_data, y_data = None, None
        if x_data is None or y_data is None:
            return
        x_arr = np.asarray(x_data, dtype=float).ravel()
        y_arr = np.asarray(y_data, dtype=float).ravel()
        if x_arr.size == 0 or y_arr.size == 0:
            return
        eps = np.finfo(float).tiny
        if self._logx:
            x_arr = np.clip(x_arr, eps, np.inf)
        if self._logy:
            y_arr = np.clip(y_arr, eps, np.inf)
        self._curve.setData(x_arr, y_arr)

    def _lcurve_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return cached L-curve chi-squared and solution-norm arrays."""
        model = self.fit.model
        try:
            chi2 = np.asarray(model.l_curve_chi2, dtype=float)
            sol = np.asarray(model.l_curve_solution_norm, dtype=float)
        except Exception:
            chi2 = np.array([], dtype=float)
            sol = np.array([], dtype=float)
        return chi2, sol

    def _clear_plot(self) -> None:
        """Clear plotted L-curve data and selection markers."""
        self._last_indices = np.array([], dtype=int)
        self._selected_global_index = None
        self._curve.setData([], [])
        self._selected_point.setData([], [])
        self._corner_point.setData([], [])
        self._selected_point.hide()
        self._corner_point.hide()

    def refresh_from_model(self, auto_select: bool = False) -> None:
        """Refresh the plot from cached L-curve arrays.

        Parameters
        ----------
        auto_select : bool, optional
            If True, auto-select the detected L-curve corner and update the
            model regularization parameter to that value.
        """
        chi2, sol = self._lcurve_arrays()
        if chi2.size == 0 or sol.size == 0:
            self._clear_plot()
            return
        mask = np.isfinite(chi2) & np.isfinite(sol)
        if not np.any(mask):
            self._clear_plot()
            return

        self._last_indices = np.nonzero(mask)[0]
        chi2_plot = chi2[mask]
        sol_plot = sol[mask]
        self._curve.setData(chi2_plot, sol_plot)
        self.set_log_mode(self._logx, self._logy)
        self._plot_widget.enableAutoRange()

        model = self.fit.model
        corner_idx = None
        if auto_select:
            try:
                corner_idx = getattr(model, "l_curve_corner_index", lambda: None)()
            except Exception:
                corner_idx = None
            if corner_idx is not None and int(corner_idx) in set(self._last_indices.tolist()):
                self._selected_global_index = int(corner_idx)
                setter = getattr(model, "set_reg_from_lcurve_index", None)
                if callable(setter):
                    try:
                        setter(int(corner_idx))
                    except Exception:
                        pass
                self._highlight_global_index(int(corner_idx))
                return

        if self._selected_global_index is not None and int(self._selected_global_index) in set(self._last_indices.tolist()):
            self._highlight_global_index(int(self._selected_global_index))
        else:
            self._selected_point.hide()
            if corner_idx is not None and int(corner_idx) in set(self._last_indices.tolist()):
                self._highlight_corner(int(corner_idx))
            else:
                self._corner_point.hide()

    def update_all(self, *args, **kwargs) -> None:
        """Recompute the L-curve and refresh the plot.

        Parameters
        ----------
        *args
            Ignored (allows use as Qt slot).
        **kwargs
            Ignored.
        """
        model = self.fit.model
        self._selected_global_index = None
        ctrl = getattr(self, "plot_controller", None)
        log10_min = None
        log10_max = None
        n_points = None
        if ctrl is not None:
            try:
                log10_min = ctrl.log10_min
                log10_max = ctrl.log10_max
                n_points = ctrl.n_points
            except Exception:
                pass
        compute = getattr(model, "compute_l_curve", None)
        if callable(compute):
            try:
                if n_points is not None:
                    compute(
                        n_points=int(n_points),
                        log10_min=log10_min,
                        log10_max=log10_max,
                    )
                else:
                    compute()
            except Exception:
                pass
        self.refresh_from_model(auto_select=True)

    def _highlight_global_index(self, idx: int) -> None:
        """Highlight a global L-curve index."""
        if idx not in set(self._last_indices.tolist()):
            self._selected_point.hide()
            return
        local_idx = int(np.nonzero(self._last_indices == int(idx))[0][0])
        chi2, sol = self._lcurve_arrays()
        if local_idx >= chi2.size or local_idx >= sol.size:
            self._selected_point.hide()
            return
        self._selected_point.setData(
            [float(chi2[local_idx])],
            [float(sol[local_idx])],
        )
        self._selected_point.show()
        corner_idx = None
        try:
            corner_idx = getattr(self.fit.model, "l_curve_corner_index", lambda: None)()
        except Exception:
            corner_idx = None
        if corner_idx is not None and int(corner_idx) != int(idx):
            self._highlight_corner(int(corner_idx))
        else:
            self._corner_point.hide()

    def _highlight_corner(self, idx: int) -> None:
        """Highlight the automatically detected L-curve corner."""
        if idx not in set(self._last_indices.tolist()):
            self._corner_point.hide()
            return
        local_idx = int(np.nonzero(self._last_indices == int(idx))[0][0])
        chi2, sol = self._lcurve_arrays()
        if local_idx >= chi2.size or local_idx >= sol.size:
            self._corner_point.hide()
            return
        self._corner_point.setData(
            [float(chi2[local_idx])],
            [float(sol[local_idx])],
        )
        self._corner_point.show()

    def _nearest_global_index(self, x: float, y: float) -> int | None:
        """Return the global L-curve index closest to a plot position."""
        model = self.fit.model
        chi2 = np.asarray(model.l_curve_chi2, dtype=float)
        sol = np.asarray(model.l_curve_solution_norm, dtype=float)
        if chi2.size != sol.size or chi2.size == 0:
            return None
        mask = np.isfinite(chi2) & np.isfinite(sol)
        if not np.any(mask):
            return None
        idx = np.nonzero(mask)[0]
        x_ref = np.asarray(x, dtype=float)
        y_ref = np.asarray(y, dtype=float)
        x_data = chi2[idx]
        y_data = sol[idx]
        if self._logx:
            eps = np.finfo(float).tiny
            x_ref = np.log10(max(float(x_ref), eps))
            x_data = np.log10(np.clip(x_data, eps, np.inf))
        if self._logy:
            eps = np.finfo(float).tiny
            y_ref = np.log10(max(float(y_ref), eps))
            y_data = np.log10(np.clip(y_data, eps, np.inf))
        d2 = (x_data - x_ref) ** 2 + (y_data - y_ref) ** 2
        return int(idx[int(np.argmin(d2))])

    def _select_global_index(self, idx: int) -> None:
        """Select an L-curve index and update the model regularization."""
        self._selected_global_index = int(idx)
        model = self.fit.model
        setter = getattr(model, "set_reg_from_lcurve_index", None)
        if callable(setter):
            try:
                setter(int(idx))
            except Exception:
                pass
        self._highlight_global_index(int(idx))

    def _on_points_clicked(self, item, points):
        """Qt slot: handle click on an L-curve point."""
        try:
            if points is None or len(points) == 0:
                return
            pos = points[0].pos()
            self._select_global_index(self._nearest_global_index(float(pos.x()), float(pos.y())))
        except Exception:
            return

    def _on_scene_clicked(self, event):
        """Qt slot: handle click anywhere in the L-curve plot."""
        try:
            if event.button() != QtCore.Qt.LeftButton:
                return
            view_pos = self._plot_widget.getViewBox().mapSceneToView(event.scenePos())
            idx = self._nearest_global_index(float(view_pos.x()), float(view_pos.y()))
            if idx is not None:
                self._select_global_index(idx)
        except Exception:
            return


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
            cs.gui.plots.DistributionPlot,
            {
                "distribution_options": {
                    "MaxEnt tau_D": {
                        "attribute": "maxent_tauD_distribution",
                        "accessor": lambda x, **kwargs: x,
                        "accessor_kwargs": {"sort": False},
                        "curve_options": {
                            "stepMode": "right",
                            "connect": "all",
                            "symbol": "None",
                            "multi_curve": False,
                            "fillLevel": 0.0,
                            "fillBrush": cs.core.settings.gui["plot"]["colors"]["data"],
                        },
                    }
                },
                "scale_x": "log",
            },
        ),
        (cs.gui.plots.ResidualPlot, {}),
        (MaxEntFCSLCurvePlot, {}),
    ]

    def __init__(
        self,
        fit: cs.core.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs,
    ):
        """Initialize the MaxEnt FCS widget (GUI).

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

    def update_widgets(self) -> None:
        """Refresh parameter widgets from the current model state.

        This ensures that changes to the regularization parameter made via
        the L-curve (or other programmatic paths) are reflected in the GUI
        spinboxes without relying on a full Fit.update() call.
        """
        # Ensure parameters list is up to date
        try:
            self.find_parameters()
        except Exception:
            pass
        for p in getattr(self, 'parameters_all', []):
            ctrl = getattr(p, 'controller', None)
            if ctrl is not None:
                try:
                    ctrl.finalize()
                except Exception:
                    pass


class MaxEntRHModel(ModelCurve):
    """FCS model that reconstructs an rH distribution via MaxEnt.

    This model wraps :func:`cs.core.models.fcs.maxent.fcs_maxent_rh` and
    exposes both the reconstructed FCS curve and a hydrodynamic-radius
    distribution on ``rh_grid``.
    """

    name = "FCS MaxEnt rH"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs):
        """Initialize the MaxEnt rH model.

        Parameters
        ----------
        fit : cs.core.fitting.fit.Fit
            The fit this model belongs to.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        super().__init__(fit, **kwargs)

        self._reg = FittingParameter(
            name="reg",
            value=-4.0,
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
            value=500.0,
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
        self._l_curve_log10_reg = None
        self._l_curve_chi2 = None
        self._l_curve_solution_norm = None
        self._l_curve_corner_index = None

    @property
    def last_result(self) -> dict | None:
        """Return the last MaxEnt rH result or ``None``."""
        return self._result

    @property
    def maxent_rH_distribution(self):
        """Return the MaxEnt rH distribution as ``(y, x) = (p, rh_grid)``.

        This is used by :class:`cs.gui.plots.DistributionPlot` via the
        ``"maxent_rH_distribution"`` attribute.
        """
        if self._result is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        rh_grid = np.asarray(self._result["rh_grid"], dtype=float)
        p = np.asarray(self._result["p"], dtype=float)
        return p, rh_grid

    @property
    def l_curve_log10_reg(self):
        """L-curve log10(reg) grid points.

        Returns
        -------
        np.ndarray
            1-D array of log10(regularization) values, or empty.
        """
        if self._l_curve_log10_reg is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_log10_reg, dtype=float)

    @property
    def l_curve_reg(self):
        """L-curve regularization values (linear scale).

        Returns
        -------
        np.ndarray
            1-D array of regularization values, or empty.
        """
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return vals
        return 10.0 ** vals

    @property
    def l_curve_chi2(self):
        r"""L-curve chi-squared values per regularization point.

        Returns
        -------
        np.ndarray
            1-D array of :math:`\\chi^2_r` values, or empty.
        """
        if self._l_curve_chi2 is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_chi2, dtype=float)

    @property
    def l_curve_solution_norm(self):
        """L-curve solution norm (|p|) per regularization point.

        Returns
        -------
        np.ndarray
            1-D array of solution norms, or empty.
        """
        if self._l_curve_solution_norm is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_solution_norm, dtype=float)

    def compute_l_curve(
            self,
            n_points: int = 32,
            log10_min: float | None = None,
            log10_max: float | None = None,
    ) -> None:
        """Sweep the regularization parameter and build the L-curve.

        The numerical sweep is performed by
        :func:`chisurf.core.models.fcs.maxent.compute_fcs_maxent_rh_l_curve`;
        this method only adapts the current model parameters and fit-window
        settings to the core helper.

        Parameters
        ----------
        n_points : int, optional
            Number of grid points.
        log10_min, log10_max : float, optional
            log10(reg) range. Defaults to current value ± 2.
        """
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()
        if tau.size == 0 or g.size == 0:
            self._l_curve_log10_reg = np.array([], dtype=float)
            self._l_curve_chi2 = np.array([], dtype=float)
            self._l_curve_solution_norm = np.array([], dtype=float)
            self._l_curve_corner_index = None
            return

        if n_points is None or int(n_points) <= 1:
            n_points = 2
        current_log10 = float(self._reg.value)
        try:
            lb, ub = [float(b) for b in self._reg.bounds]
        except Exception:
            lb, ub = float("-inf"), float("inf")
        if log10_min is None:
            log10_min = max(lb, current_log10 - 2.0)
        if log10_max is None:
            log10_max = min(ub, current_log10 + 2.0)
        if log10_min > log10_max:
            log10_min, log10_max = log10_max, log10_min

        temp_c = float(self._temp.value)
        result = compute_fcs_maxent_rh_l_curve(
            tau=tau,
            g=g,
            y_error=getattr(data, "ey", None),
            log10_min=float(log10_min),
            log10_max=float(log10_max),
            n_points=int(n_points),
            xmin=getattr(self.fit, "xmin", 0),
            xmax=getattr(self.fit, "xmax", None),
            mask=getattr(self.fit, "mask", None),
            n_free=int(self.n_free),
            rh_min=float(self._rh_min.value),
            rh_max=float(self._rh_max.value),
            n_rh=int(self._n_rh.value) if int(self._n_rh.value) > 0 else 80,
            w0=float(self._w0.value) * 1.0e-3,
            s=float(self._s.value),
            baseline=float(self._b.value),
            temperature=temp_c + 273.15,
            num_iter=60,
        )
        self._l_curve_log10_reg = result.log10_reg
        self._l_curve_chi2 = result.chi2r
        self._l_curve_solution_norm = result.solution_norm
        self._l_curve_corner_index = result.corner_index
        self.update_model()

    def l_curve_corner_index(self) -> int | None:
        """Locate the L-curve corner index automatically.

        Returns
        -------
        int or None
            Global index into the L-curve arrays, or ``None``.
        """
        if self._l_curve_corner_index is not None:
            return int(self._l_curve_corner_index)
        rho = self.l_curve_chi2
        eta = self.l_curve_solution_norm
        if rho.size < 3 or eta.size < 3:
            return None
        idx = cs.core.math.regularization.discrete_lcurve_corner(rho, eta)
        if idx is not None:
            self._l_curve_corner_index = int(idx)
        return int(idx) if idx is not None else None

    def set_reg_from_lcurve_index(self, idx: int) -> None:
        """Set the regularization parameter from an L-curve index.

        Parameters
        ----------
        idx : int
            Index into the global L-curve arrays.
        """
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return
        if idx < 0 or idx >= vals.size:
            return
        new_log10 = float(vals[idx])
        self._reg.value = new_log10
        fc = get_fitting_client()
        if fc is not None:
            fc.set_parameter_value(
                parameter_name=str(self._reg.name),
                value=new_log10,
                fit_index=getattr(self.fit, "fit_idx", None),
            )
        update = getattr(self, "update", None)
        if callable(update):
            try:
                update()
                return
            except Exception:
                pass
        try:
            self.update_model()
        except Exception:
            pass

    def update_model(self, **kwargs) -> None:
        """Run MaxEnt in rH-space on the current FCS dataset.

        The experimental data are taken from ``self.fit.data``. The
        hydrodynamic-radius grid, beam waist, temperature and other options
        are derived from the internal :class:`FittingParameter` instances
        and passed to :func:`cs.core.models.fcs.maxent.fcs_maxent_rh`.
        The reconstructed curve is stored in ``self.y`` and the full
        result dictionary in ``self._result``.
        """
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
    """GUI model widget for hydrodynamic-radius MaxEnt FCS analysis."""

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
            cs.gui.plots.DistributionPlot,
            {
                "distribution_options": {
                    "MaxEnt rH": {
                        "attribute": "maxent_rH_distribution",
                        "accessor": lambda x, **kwargs: x,
                        "accessor_kwargs": {"sort": False},
                        "curve_options": {
                            "stepMode": "right",
                            "connect": "all",
                            "symbol": "None",
                            "multi_curve": False,
                            "fillLevel": 0.0,
                            "fillBrush": cs.core.settings.gui["plot"]["colors"]["data"],
                        },
                    }
                },
                "scale_x": "log",
            },
        ),
        (cs.gui.plots.ResidualPlot, {}),
        (MaxEntFCSLCurvePlot, {}),
    ]

    def __init__(
        self,
        fit: cs.core.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs,
    ):
        """Initialize the MaxEnt rH widget (GUI).

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
