from __future__ import annotations

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import chisurf
import chisurf.fitting
from chisurf.models.model import ModelWidget, ModelCurve
from chisurf.fitting.parameter import FittingParameter
from chisurf.models.fcs.maxent import fcs_maxent, fcs_maxent_rh
from chisurf import plots
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
import chisurf.math.regularization


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

    @property
    def last_result(self) -> dict | None:
        """Return the last MaxEnt result dictionary or ``None``.

        The structure matches the output of
        :func:`chisurf.models.fcs.maxent.fcs_maxent` and contains the
        reconstructed curve, diffusion-time grid and MaxEnt distribution.
        """
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

    @property
    def l_curve_log10_reg(self):
        if self._l_curve_log10_reg is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_log10_reg, dtype=float)

    @property
    def l_curve_reg(self):
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return vals
        return 10.0 ** vals

    @property
    def l_curve_chi2(self):
        if self._l_curve_chi2 is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_chi2, dtype=float)

    @property
    def l_curve_solution_norm(self):
        if self._l_curve_solution_norm is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_solution_norm, dtype=float)

    def compute_l_curve(
            self,
            n_points: int = 32,
            log10_min: float | None = None,
            log10_max: float | None = None,
    ) -> None:
        print("MaxEntFCSModel.compute_l_curve: called")
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()
        print(f"MaxEntFCSModel.compute_l_curve: tau.size={tau.size}, g.size={g.size}")
        if tau.size == 0 or g.size == 0:
            self._l_curve_log10_reg = np.array([], dtype=float)
            self._l_curve_chi2 = np.array([], dtype=float)
            self._l_curve_solution_norm = np.array([], dtype=float)
            print("MaxEntFCSModel.compute_l_curve: empty data, clearing L-curve arrays")
            return
        if n_points is None or n_points <= 1:
            n_points = 2
        current_log10 = float(self._reg.value)
        try:
            lb, ub = [float(b) for b in self._reg.bounds]
        except Exception as e:
            print(f"MaxEntFCSModel.compute_l_curve: failed to read bounds from _reg: {e}")
            lb, ub = float("-inf"), float("inf")
        if log10_min is None:
            log10_min = max(lb, current_log10 - 2.0)
        if log10_max is None:
            log10_max = min(ub, current_log10 + 2.0)
        if log10_min > log10_max:
            log10_min, log10_max = log10_max, log10_min
        grid = np.linspace(log10_min, log10_max, int(n_points))
        print(f"MaxEntFCSModel.compute_l_curve: grid size={grid.size}, range=[{float(grid[0])}, {float(grid[-1])}]")
        chi2_vals = np.empty_like(grid, dtype=float)
        sol_vals = np.empty_like(grid, dtype=float)
        old_log10 = current_log10
        for i, v in enumerate(grid):
            try:
                self._reg.value = float(v)
                self.update_model()
                chi2_vals[i] = float(self.fit.chi2r)
                result = self._result
                if result is None:
                    sol_vals[i] = np.nan
                else:
                    p = np.asarray(result.get("p", []), dtype=float).ravel()
                    if p.size == 0:
                        sol_vals[i] = np.nan
                    else:
                        sol_vals[i] = float(np.linalg.norm(p))
            except Exception as e:
                print(f"MaxEntFCSModel.compute_l_curve: exception at index {i}, v={float(v)}: {e}")
                chi2_vals[i] = np.nan
                sol_vals[i] = np.nan
        self._l_curve_log10_reg = grid
        self._l_curve_chi2 = chi2_vals
        self._l_curve_solution_norm = sol_vals
        finite_mask = np.isfinite(chi2_vals) & np.isfinite(sol_vals)
        print(f"MaxEntFCSModel.compute_l_curve: done, n={grid.size}, n_finite={int(finite_mask.sum())}")
        self._reg.value = old_log10
        try:
            self.update_model()
        except Exception as e:
            print(f"MaxEntFCSModel.compute_l_curve: final update_model() failed: {e}")
            pass

    def l_curve_corner_index(self) -> int | None:
        """Return index of the automatically detected L-curve corner.

        The index refers to the *global* L-curve arrays stored on the
        model (``l_curve_log10_reg``, ``l_curve_chi2``,
        ``l_curve_solution_norm``).  ``None`` is returned if no suitable
        corner can be determined.
        """
        rho = self.l_curve_chi2
        eta = self.l_curve_solution_norm
        if rho.size < 3 or eta.size < 3:
            return None
        idx = chisurf.math.regularization.discrete_lcurve_corner(rho, eta)
        return int(idx) if idx is not None else None

    def set_reg_from_lcurve_index(self, idx: int) -> None:
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return
        if idx < 0 or idx >= vals.size:
            return
        new_log10 = float(vals[idx])
        print(f"MaxEntFCSModel.set_reg_from_lcurve_index: idx={idx}, log10_reg={new_log10}")
        self._reg.value = new_log10
        # For plain model instances, just refresh the MaxEnt result; for
        # widgets (ModelWidget subclasses) the GUI layer will call
        # update_widgets and update_plots around this.
        try:
            self.update_model()
        except Exception as e2:
            print(f"MaxEntFCSModel.set_reg_from_lcurve_index: update_model() failed: {e2}")

    def on_auto_fit_range_completed(self) -> None:
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
                chisurf.logging.warning(
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

    def __init__(self, parent_plot):
        super().__init__(parent_plot)
        self._plot = parent_plot
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Axis scale controls
        scale_layout = QtWidgets.QHBoxLayout()
        self._cb_logx = QtWidgets.QCheckBox("log Chi2r")
        self._cb_logy = QtWidgets.QCheckBox("log |p|")
        self._cb_logx.setChecked(True)
        self._cb_logy.setChecked(True)
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
        self._sb_logmin.setPrefix("log10 min = ")
        self._sb_logmin.setDecimals(2)
        self._sb_logmin.setSingleStep(0.5)
        self._sb_logmin.setRange(-12.0, 12.0)
        self._sb_logmin.setValue(-3.0)

        self._sb_logmax = QtWidgets.QDoubleSpinBox()
        self._sb_logmax.setPrefix("log10 max = ")
        self._sb_logmax.setDecimals(2)
        self._sb_logmax.setSingleStep(0.5)
        self._sb_logmax.setRange(-12.0, 12.0)
        self._sb_logmax.setValue(3.0)

        self._sb_npoints = QtWidgets.QSpinBox()
        self._sb_npoints.setPrefix("N = ")
        self._sb_npoints.setRange(2, 256)
        self._sb_npoints.setSingleStep(2)
        self._sb_npoints.setValue(32)

        range_layout.addWidget(self._sb_logmin)
        range_layout.addWidget(self._sb_logmax)
        range_layout.addWidget(self._sb_npoints)

        self._btn_compute = QtWidgets.QPushButton("Compute L-Curve")
        self._btn_compute.clicked.connect(self._on_compute_clicked)
        range_layout.addWidget(self._btn_compute)
        range_layout.addStretch(1)

        layout.addLayout(range_layout)

    def _on_scale_changed(self, *args):
        self._plot.set_log_mode(
            logx=self._cb_logx.isChecked(),
            logy=self._cb_logy.isChecked(),
        )

    @property
    def log10_min(self) -> float | None:
        return float(self._sb_logmin.value())

    @property
    def log10_max(self) -> float | None:
        return float(self._sb_logmax.value())

    @property
    def n_points(self) -> int:
        return int(self._sb_npoints.value())

    def _on_compute_clicked(self, *args):
        # Trigger a recomputation and plot update using the current
        # controller settings for the L-curve range.
        try:
            self._plot.update_all()
        except Exception:
            pass


class MaxEntFCSLCurvePlot(plots.Plot):

    name = "L-Curve"

    def __init__(self, fit: chisurf.fitting.fit.FitGroup, **kwargs):
        super().__init__(fit=fit, **kwargs)
        self._plot_widget = pg.PlotWidget()
        self.layout.addWidget(self._plot_widget)
        # Main L-curve trace: misfit vs solution norm
        self._curve = self._plot_widget.plot([], [], pen=None, symbol="o")
        # Highlight for the currently selected regularization point
        self._selected_point = self._plot_widget.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolBrush="r",
            symbolPen="r",
            symbolSize=14,
        )
        # Start with no selection visible
        try:
            self._selected_point.hide()
        except Exception:
            pass
        self._plot_widget.setLabel("bottom", "Chi2r")
        self._plot_widget.setLabel("left", "|p|")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._logx = True
        self._logy = True
        self._last_indices = np.array([], dtype=int)
        self.set_log_mode(self._logx, self._logy)
        self.plot_controller = MaxEntFCSLCurveController(self)
        try:
            self._curve.sigPointsClicked.connect(self._on_points_clicked)
        except Exception:
            pass

    def set_log_mode(self, logx: bool, logy: bool) -> None:
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

    def update_all(self, *args, **kwargs) -> None:
        print("MaxEntFCSLCurvePlot.update_all: called")
        model = self.fit.model
        # Obtain L-curve range configuration from the controller, if present
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
            except Exception as e:
                print(f"MaxEntFCSLCurvePlot.update_all: compute_l_curve raised {e}")
                pass
        try:
            chi2 = np.asarray(model.l_curve_chi2, dtype=float)
            sol = np.asarray(model.l_curve_solution_norm, dtype=float)
            print(f"MaxEntFCSLCurvePlot.update_all: chi2.size={chi2.size}, sol.size={sol.size}")
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot.update_all: failed to read L-curve arrays: {e}")
            chi2 = np.array([], dtype=float)
            sol = np.array([], dtype=float)
        if chi2.size == 0 or sol.size == 0:
            print("MaxEntFCSLCurvePlot.update_all: empty chi2 or sol, clearing plot")
            self._last_indices = np.array([], dtype=int)
            self._curve.setData([], [])
            return
        mask = np.isfinite(chi2) & np.isfinite(sol)
        if not np.any(mask):
            print("MaxEntFCSLCurvePlot.update_all: no finite points in L-curve, clearing plot")
            self._last_indices = np.array([], dtype=int)
            self._curve.setData([], [])
            return
        self._last_indices = np.nonzero(mask)[0]
        chi2_plot = chi2[mask]
        sol_plot = sol[mask]
        self._curve.setData(chi2_plot, sol_plot)
        print(f"MaxEntFCSLCurvePlot.update_all: plotted {chi2_plot.size} points")
        self._plot_widget.enableAutoRange()
        # Hide any previous selection highlight; it will be re-applied on
        # the next point click or auto-corner detection.
        try:
            self._selected_point.hide()
        except Exception:
            pass
        # Auto-select a recommended regularization via L-curve corner
        # detection, if available, and highlight it.  This preserves the
        # full L-curve but starts the user at a sensible point.
        try:
            corner_idx_global = getattr(model, "l_curve_corner_index", lambda: None)()
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot.update_all: l_curve_corner_index failed: {e}")
            corner_idx_global = None
        if corner_idx_global is not None and self._last_indices.size > 0:
            try:
                # Map global index to local masked index, if present.
                matches = np.nonzero(self._last_indices == int(corner_idx_global))[0]
                if matches.size > 0:
                    local_idx = int(matches[0])
                    print(f"MaxEntFCSLCurvePlot.update_all: auto-selected corner at global {corner_idx_global}, local {local_idx}")
                    # Update model reg and highlight via the same machinery
                    # as a manual click.
                    setter = getattr(model, "set_reg_from_lcurve_index", None)
                    if callable(setter):
                        try:
                            setter(int(corner_idx_global))
                        except Exception as e:
                            print(f"MaxEntFCSLCurvePlot.update_all: set_reg_from_lcurve_index for corner raised {e}")
                    # Visually highlight
                    if 0 <= local_idx < chi2_plot.size:
                        self._selected_point.setData(
                            [float(chi2_plot[local_idx])],
                            [float(sol_plot[local_idx])],
                        )
                        try:
                            self._selected_point.show()
                        except Exception:
                            pass
                else:
                    print(f"MaxEntFCSLCurvePlot.update_all: corner index {corner_idx_global} not in finite mask")
            except Exception as e:
                print(f"MaxEntFCSLCurvePlot.update_all: failed to auto-highlight corner: {e}")
        self.set_log_mode(self._logx, self._logy)

    def _on_points_clicked(self, item, points):
        print("MaxEntFCSLCurvePlot._on_points_clicked: called")
        print(f"MaxEntFCSLCurvePlot._on_points_clicked: item={item}, type(points)={type(points)}")
        # ``points`` may be a list-like or numpy array; avoid ambiguous
        # truth-value checks and fall back gracefully if empty.
        try:
            n_points = len(points)
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: len(points) failed: {e}")
            n_points = 0
        if points is None or n_points == 0:
            print("MaxEntFCSLCurvePlot._on_points_clicked: no points, returning")
            return

        # Determine the curve index whose (chi2, |p|) point is closest to the
        # clicked position in the current plot coordinates.
        try:
            click_pos = points[0].pos()
            click_x = float(click_pos.x())
            click_y = float(click_pos.y())
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: failed to read click position: {e}")
            return
        try:
            x_data, y_data = self._curve.getData()
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: getData failed: {e}")
            return
        if x_data is None or y_data is None:
            print("MaxEntFCSLCurvePlot._on_points_clicked: no curve data, returning")
            return
        x_arr = np.asarray(x_data, dtype=float).ravel()
        y_arr = np.asarray(y_data, dtype=float).ravel()
        if x_arr.size == 0 or y_arr.size == 0:
            print("MaxEntFCSLCurvePlot._on_points_clicked: empty curve data, returning")
            return
        try:
            x_ref = click_x
            y_ref = click_y
            d2 = (x_arr - x_ref) ** 2 + (y_arr - y_ref) ** 2
            local_idx = int(np.argmin(d2))
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: nearest local_idx={local_idx}")
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: distance-based index selection failed: {e}")
            return
        if local_idx < 0 or local_idx >= self._last_indices.size:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: local_idx {local_idx} out of range for _last_indices size {self._last_indices.size}")
            return
        global_idx = int(self._last_indices[local_idx])
        print(f"MaxEntFCSLCurvePlot._on_points_clicked: mapped local_idx={local_idx} to global_idx={global_idx}")
        model = self.fit.model
        setter = getattr(model, "set_reg_from_lcurve_index", None)
        if callable(setter):
            try:
                print("MaxEntFCSLCurvePlot._on_points_clicked: calling set_reg_from_lcurve_index")
                setter(global_idx)
            except Exception as e:
                print(f"MaxEntFCSLCurvePlot._on_points_clicked: set_reg_from_lcurve_index raised {e}")
        else:
            print("MaxEntFCSLCurvePlot._on_points_clicked: model has no set_reg_from_lcurve_index")
        # Visually highlight the selected point as a larger red circle
        try:
            x_data, y_data = self._curve.getData()
            if x_data is not None and y_data is not None:
                if 0 <= local_idx < len(x_data):
                    sel_x = [float(x_data[local_idx])]
                    sel_y = [float(y_data[local_idx])]
                    self._selected_point.setData(sel_x, sel_y)
                    try:
                        self._selected_point.show()
                    except Exception:
                        pass
        except Exception as e:
            print(f"MaxEntFCSLCurvePlot._on_points_clicked: failed to update selected_point marker: {e}")


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
                            "stepMode": "right",
                            "connect": "all",
                            "symbol": "None",
                            "multi_curve": False,
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    }
                },
                "scale_x": "log",
            },
        ),
        (chisurf.plots.ResidualPlot, {}),
        (MaxEntFCSLCurvePlot, {}),
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

    This model wraps :func:`chisurf.models.fcs.maxent.fcs_maxent_rh` and
    exposes both the reconstructed FCS curve and a hydrodynamic-radius
    distribution on ``rh_grid``.
    """

    name = "FCS MaxEnt rH"

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
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

    @property
    def last_result(self) -> dict | None:
        return self._result

    @property
    def maxent_rH_distribution(self):
        """Return the MaxEnt rH distribution as ``(y, x) = (p, rh_grid)``.

        This is used by :class:`chisurf.plots.DistributionPlot` via the
        ``"maxent_rH_distribution"`` attribute.
        """
        if self._result is None:
            return np.array([], dtype=float), np.array([], dtype=float)
        rh_grid = np.asarray(self._result["rh_grid"], dtype=float)
        p = np.asarray(self._result["p"], dtype=float)
        return p, rh_grid

    @property
    def l_curve_log10_reg(self):
        if self._l_curve_log10_reg is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_log10_reg, dtype=float)

    @property
    def l_curve_reg(self):
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return vals
        return 10.0 ** vals

    @property
    def l_curve_chi2(self):
        if self._l_curve_chi2 is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_chi2, dtype=float)

    @property
    def l_curve_solution_norm(self):
        if self._l_curve_solution_norm is None:
            return np.array([], dtype=float)
        return np.asarray(self._l_curve_solution_norm, dtype=float)

    def compute_l_curve(
            self,
            n_points: int = 32,
            log10_min: float | None = None,
            log10_max: float | None = None,
    ) -> None:
        data = self.fit.data
        tau = np.asarray(data.x, dtype=float).ravel()
        g = np.asarray(data.y, dtype=float).ravel()
        if tau.size == 0 or g.size == 0:
            self._l_curve_log10_reg = np.array([], dtype=float)
            self._l_curve_chi2 = np.array([], dtype=float)
            self._l_curve_solution_norm = np.array([], dtype=float)
            return
        if n_points is None or n_points <= 1:
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
        grid = np.linspace(log10_min, log10_max, int(n_points))
        chi2_vals = np.empty_like(grid, dtype=float)
        sol_vals = np.empty_like(grid, dtype=float)
        old_log10 = current_log10
        for i, v in enumerate(grid):
            try:
                self._reg.value = float(v)
                self.update_model()
                chi2_vals[i] = float(self.fit.chi2r)
                result = self._result
                if result is None:
                    sol_vals[i] = np.nan
                else:
                    p = np.asarray(result.get("p", []), dtype=float).ravel()
                    if p.size == 0:
                        sol_vals[i] = np.nan
                    else:
                        sol_vals[i] = float(np.linalg.norm(p))
            except Exception:
                chi2_vals[i] = np.nan
                sol_vals[i] = np.nan
        self._l_curve_log10_reg = grid
        self._l_curve_chi2 = chi2_vals
        self._l_curve_solution_norm = sol_vals
        self._reg.value = old_log10
        try:
            self.update_model()
        except Exception:
            pass

    def l_curve_corner_index(self) -> int | None:
        rho = self.l_curve_chi2
        eta = self.l_curve_solution_norm
        if rho.size < 3 or eta.size < 3:
            return None
        idx = chisurf.math.regularization.discrete_lcurve_corner(rho, eta)
        return int(idx) if idx is not None else None

    def set_reg_from_lcurve_index(self, idx: int) -> None:
        vals = self.l_curve_log10_reg
        if vals.size == 0:
            return
        if idx < 0 or idx >= vals.size:
            return
        new_log10 = float(vals[idx])
        self._reg.value = new_log10
        try:
            self.fit.update()
        except Exception:
            try:
                self.update_model()
            except Exception:
                pass

    def update_model(self, **kwargs) -> None:
        """Run MaxEnt in rH-space on the current FCS dataset.

        The experimental data are taken from ``self.fit.data``. The
        hydrodynamic-radius grid, beam waist, temperature and other options
        are derived from the internal :class:`FittingParameter` instances
        and passed to :func:`chisurf.models.fcs.maxent.fcs_maxent_rh`.
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
                            "stepMode": "right",
                            "connect": "all",
                            "symbol": "None",
                            "multi_curve": False,
                            "fillLevel": 0.0,
                            "fillBrush": chisurf.settings.gui["plot"]["colors"]["data"],
                        },
                    }
                },
                "scale_x": "log",
            },
        ),
        (chisurf.plots.ResidualPlot, {}),
        (MaxEntFCSLCurvePlot, {}),
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
