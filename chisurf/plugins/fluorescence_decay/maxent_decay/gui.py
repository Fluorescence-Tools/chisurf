"""Qt/pyqtgraph GUI front-end for the MaxEnt TCSPC lifetime plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import json
import numpy as np

from .fmem.core import mem_vin4_lifetime, mem_vin4_fret

# Lazy-loaded Qt stack ------------------------------------------------------
pg = None
QtWidgets = None
QtCore = None
chisurf = None
ExperimentalDataSelector = None


def _lazy_import_qt_stack():
    global pg, QtWidgets, QtCore, chisurf, ExperimentalDataSelector

    if QtWidgets is None or QtCore is None:
        from qtpy import QtWidgets as _QtWidgets, QtCore as _QtCore  # type: ignore

        QtWidgets = _QtWidgets
        QtCore = _QtCore

    if pg is None:
        import pyqtgraph as _pg  # type: ignore

        pg = _pg

    if chisurf is None:
        import chisurf as _chisurf  # type: ignore

        chisurf = _chisurf

    if ExperimentalDataSelector is None:
        from chisurf.gui.widgets.experiments import (
            ExperimentalDataSelector as _ExperimentalDataSelector,
        )  # type: ignore

        ExperimentalDataSelector = _ExperimentalDataSelector


_lazy_import_qt_stack()


class MaxentDecayWidget(QtWidgets.QMainWindow):  # type: ignore[misc]
    """GUI front-end for the MaxEnt lifetime MEM analysis."""

    def __init__(self):
        _lazy_import_qt_stack()
        super().__init__()
        self.setWindowTitle("MaxEnt TCSPC lifetime (dev)")
        self.resize(900, 700)

        self._irf_dataset = None
        self._prior_vec: Optional[np.ndarray] = None
        self._donly_vec: Optional[np.ndarray] = None
        self._dist_prior_vec: Optional[np.ndarray] = None
        self._last_result = None

        self._t_axis = None
        self._fit_range = None

        self._mode_fret = False

        self._init_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        ctrl = QtWidgets.QWidget(self)
        # Keep the control panel narrow so that plots have more
        # horizontal space.
        try:
            ctrl.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
            ctrl.setMaximumWidth(250)
        except Exception:
            pass
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(0)

        data_group = QtWidgets.QGroupBox("Data", ctrl)
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(0)

        self.label_data_source = QtWidgets.QLabel(
            "Using cs.current_fit.data (not yet checked)", data_group
        )
        self.btn_refresh_data = QtWidgets.QToolButton(data_group)
        self.btn_refresh_data.setText("Refresh from current fit")
        # Emphasize data refresh as a prominent action.
        try:
            self.btn_refresh_data.setStyleSheet(
                "QToolButton { background-color: #c67e3b; color: white; font-weight: bold; }"
            )
        except Exception:
            pass
        self.btn_refresh_data.clicked.connect(self._on_refresh_data)

        data_layout.addWidget(self.label_data_source)
        data_layout.addWidget(self.btn_refresh_data)
        ctrl_layout.addWidget(data_group)

        irf_group = QtWidgets.QGroupBox("IRF", ctrl)
        irf_layout = QtWidgets.QVBoxLayout(irf_group)
        irf_layout.setContentsMargins(0, 0, 0, 0)
        irf_layout.setSpacing(0)

        self.label_irf_source = QtWidgets.QLabel("IRF: model.convolve.irf (default)", irf_group)
        self.btn_select_irf = QtWidgets.QToolButton(irf_group)
        self.btn_select_irf.setText("Select IRF dataset")
        self.btn_clear_irf = QtWidgets.QToolButton(irf_group)
        self.btn_clear_irf.setText("Clear IRF selection")

        # Make IRF selection buttons visually stand out.
        try:
            red_style = "QToolButton { background-color: #c67e3b; color: white; }"
            self.btn_select_irf.setStyleSheet(red_style)
            self.btn_clear_irf.setStyleSheet(red_style)
        except Exception:
            pass

        self.btn_select_irf.clicked.connect(self._on_select_irf_clicked)
        self.btn_clear_irf.clicked.connect(self._on_clear_irf_clicked)

        irf_layout.addWidget(self.label_irf_source)
        irf_layout.addWidget(self.btn_select_irf)
        irf_layout.addWidget(self.btn_clear_irf)
        ctrl_layout.addWidget(irf_group)

        mem_group = QtWidgets.QGroupBox("MEM settings", ctrl)
        mem_layout = QtWidgets.QFormLayout(mem_group)
        mem_layout.setContentsMargins(0, 0, 0, 0)
        mem_layout.setSpacing(0)

        self.combo_mode = QtWidgets.QComboBox(mem_group)
        self.combo_mode.addItem("Lifetime")
        self.combo_mode.addItem("FRET")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mem_layout.addRow("Mode", self.combo_mode)

        self.spin_nu = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_nu.setDecimals(6)
        self.spin_nu.setRange(1e-8, 1.0)
        self.spin_nu.setSingleStep(1e-3)
        self.spin_nu.setValue(1e-3)
        row_nu = QtWidgets.QHBoxLayout()
        row_nu.setContentsMargins(0, 0, 0, 0)
        row_nu.setSpacing(0)
        row_nu.addWidget(self.spin_nu)
        self.btn_lcurve = QtWidgets.QToolButton(mem_group)
        self.btn_lcurve.setText("L-curve")
        # Color L-curve button blue to distinguish it from Run MEM.
        try:
            self.btn_lcurve.setStyleSheet(
                "QToolButton { background-color: #295f9f; color: white; }"
            )
        except Exception:
            pass
        self.btn_lcurve.clicked.connect(self._on_lcurve_clicked)
        row_nu.addWidget(self.btn_lcurve)
        mem_layout.addRow("nu (reg)", row_nu)

        self.spin_lcurve_dec_left = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lcurve_dec_left.setDecimals(2)
        self.spin_lcurve_dec_left.setRange(0.0, 6.0)
        self.spin_lcurve_dec_left.setSingleStep(0.5)
        self.spin_lcurve_dec_left.setValue(2.0)
        self.spin_lcurve_dec_left.setToolTip(
            "Decades below the current nu used when scanning the L-curve."
        )
        self.spin_lcurve_dec_right = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lcurve_dec_right.setDecimals(2)
        self.spin_lcurve_dec_right.setRange(0.0, 6.0)
        self.spin_lcurve_dec_right.setSingleStep(0.5)
        self.spin_lcurve_dec_right.setValue(2.0)
        self.spin_lcurve_dec_right.setToolTip(
            "Decades above the current nu used when scanning the L-curve."
        )
        row_lc = QtWidgets.QHBoxLayout()
        row_lc.setContentsMargins(0, 0, 0, 0)
        row_lc.setSpacing(0)
        row_lc.addWidget(self.spin_lcurve_dec_left)
        row_lc.addWidget(self.spin_lcurve_dec_right)
        mem_layout.addRow("L-curve span [dec]", row_lc)

        self.spin_tau_min = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau_min.setDecimals(3)
        self.spin_tau_min.setRange(1e-3, 1e3)
        self.spin_tau_min.setValue(0.001)
        self.spin_tau_max = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau_max.setDecimals(3)
        self.spin_tau_max.setRange(1e-3, 1e3)
        self.spin_tau_max.setValue(10.0)
        self.spin_tau_step = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau_step.setDecimals(3)
        self.spin_tau_step.setRange(1e-4, 10.0)
        self.spin_tau_step.setValue(0.02)

        grid_tau = QtWidgets.QVBoxLayout()
        grid_tau.setContentsMargins(0, 0, 0, 0)
        grid_tau.setSpacing(0)

        row_tau_min = QtWidgets.QHBoxLayout()
        row_tau_min.setContentsMargins(0, 0, 0, 0)
        row_tau_min.setSpacing(0)
        row_tau_min.addWidget(self.spin_tau_min)

        row_tau_max = QtWidgets.QHBoxLayout()
        row_tau_max.setContentsMargins(0, 0, 0, 0)
        row_tau_max.setSpacing(0)
        row_tau_max.addWidget(self.spin_tau_max)

        row_tau_step = QtWidgets.QHBoxLayout()
        row_tau_step.setContentsMargins(0, 0, 0, 0)
        row_tau_step.setSpacing(0)
        row_tau_step.addWidget(self.spin_tau_step)

        grid_tau.addLayout(row_tau_min)
        grid_tau.addLayout(row_tau_max)
        grid_tau.addLayout(row_tau_step)
        mem_layout.addRow("tau grid [ns]", grid_tau)
        self._tau_grid_layout = grid_tau
        try:
            self._tau_grid_label = mem_layout.labelForField(grid_tau)
        except Exception:
            self._tau_grid_label = None

        self.spin_tau0 = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau0.setDecimals(3)
        self.spin_tau0.setRange(0.01, 100.0)
        self.spin_tau0.setValue(4.1)
        self.spin_R0 = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_R0.setRange(10.0, 100.0)
        self.spin_R0.setSingleStep(1.0)
        self.spin_R0.setValue(50.0)
        self.spin_period = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_period.setRange(0.1, 1000.0)
        self.spin_period.setSingleStep(1.0)
        self.spin_period.setValue(10.0)
        try:
            cfg = getattr(chisurf.settings, "fret", {}) or {}
            self.spin_tau0.setValue(float(cfg.get("tau0", self.spin_tau0.value())))
            self.spin_R0.setValue(float(cfg.get("forster_radius", self.spin_R0.value())))
        except Exception:
            pass
        mem_layout.addRow("tau0 [ns]", self.spin_tau0)
        mem_layout.addRow("R0 [\u00c5]", self.spin_R0)
        try:
            self._tau0_label = mem_layout.labelForField(self.spin_tau0)
        except Exception:
            self._tau0_label = None
        try:
            self._R0_label = mem_layout.labelForField(self.spin_R0)
        except Exception:
            self._R0_label = None
        mem_layout.addRow("Period [ns]", self.spin_period)
        # Store the label widget for the Period row so we can show/hide it.
        try:
            self._period_label = mem_layout.labelForField(self.spin_period)
        except Exception:
            self._period_label = None
        # Shared periodic-convolution toggle for both modes.
        self.chk_use_periodic = QtWidgets.QCheckBox("Periodic convolution", mem_group)
        # Disabled by default; user can enable periodic excitation explicitly.
        self.chk_use_periodic.setChecked(False)
        # React to changes by updating which widgets are visible.
        try:
            self.chk_use_periodic.toggled.connect(lambda _checked: self._update_mode_ui())
        except Exception:
            pass
        mem_layout.addRow(self.chk_use_periodic)

        # FRET distance range (RDA grid).
        r_min_default = 18.0
        r_max_default = 120.0
        r_res_default = 96
        try:
            cfg = getattr(chisurf.settings, "fret", {}) or {}
            r_min_default = float(cfg.get("rda_min", r_min_default))
            r_max_default = float(cfg.get("rda_max", r_max_default))
            r_res_default = int(cfg.get("rda_resolution", r_res_default))
        except Exception:
            pass
        if r_res_default < 2:
            r_res_default = 2

        grid_r = QtWidgets.QVBoxLayout()
        grid_r.setContentsMargins(0, 0, 0, 0)
        grid_r.setSpacing(0)

        row_r_min = QtWidgets.QHBoxLayout()
        row_r_min.setContentsMargins(0, 0, 0, 0)
        row_r_min.setSpacing(0)
        self.spin_R_min = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_R_min.setDecimals(1)
        self.spin_R_min.setRange(1.0, 500.0)
        self.spin_R_min.setValue(r_min_default)
        row_r_min.addWidget(self.spin_R_min)

        row_r_max = QtWidgets.QHBoxLayout()
        row_r_max.setContentsMargins(0, 0, 0, 0)
        row_r_max.setSpacing(0)
        self.spin_R_max = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_R_max.setDecimals(1)
        self.spin_R_max.setRange(1.0, 500.0)
        self.spin_R_max.setValue(r_max_default)
        row_r_max.addWidget(self.spin_R_max)

        row_r_pts = QtWidgets.QHBoxLayout()
        row_r_pts.setContentsMargins(0, 0, 0, 0)
        row_r_pts.setSpacing(0)
        self.spin_R_points = QtWidgets.QSpinBox(mem_group)
        self.spin_R_points.setRange(2, 10000)
        self.spin_R_points.setValue(r_res_default)
        row_r_pts.addWidget(self.spin_R_points)

        grid_r.addLayout(row_r_min)
        grid_r.addLayout(row_r_max)
        grid_r.addLayout(row_r_pts)
        mem_layout.addRow("RDA range [\u00c5]", grid_r)
        self._rda_grid_layout = grid_r
        try:
            self._rda_grid_label = mem_layout.labelForField(grid_r)
        except Exception:
            self._rda_grid_label = None

        self.btn_load_donor = QtWidgets.QToolButton(mem_group)
        self.btn_load_donor.setText("Load donor spectrum")
        self.btn_load_donor.clicked.connect(self._on_load_donor_clicked)
        self.btn_load_donor_fit = QtWidgets.QToolButton(mem_group)
        self.btn_load_donor_fit.setText("Load donor from fit")
        self.btn_load_donor_fit.clicked.connect(self._on_load_donor_from_fit_clicked)
        donor_btns = QtWidgets.QHBoxLayout()
        donor_btns.setContentsMargins(0, 0, 0, 0)
        donor_btns.setSpacing(0)
        donor_btns.addWidget(self.btn_load_donor)
        donor_btns.addWidget(self.btn_load_donor_fit)
        self.label_donor_info = QtWidgets.QLabel("Donor spectrum: tau0 only", mem_group)
        mem_layout.addRow(donor_btns)
        mem_layout.addRow(self.label_donor_info)

        self.spin_x_donly = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_x_donly.setDecimals(3)
        self.spin_x_donly.setRange(0.0, 1.0)
        self.spin_x_donly.setSingleStep(0.05)
        self.spin_x_donly.setValue(0.0)
        mem_layout.addRow("donor-only fraction", self.spin_x_donly)
        try:
            self._x_donly_label = mem_layout.labelForField(self.spin_x_donly)
        except Exception:
            self._x_donly_label = None

        self.spin_start_frac = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_start_frac.setDecimals(2)
        self.spin_start_frac.setRange(0.1, 1.0)
        self.spin_start_frac.setSingleStep(0.05)
        self.spin_start_frac.setValue(0.9)
        mem_layout.addRow("start @ frac of peak", self.spin_start_frac)

        self.spin_timeshift = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_timeshift.setDecimals(4)
        self.spin_timeshift.setRange(-100.0, 100.0)
        self.spin_timeshift.setSingleStep(0.01)
        self.spin_timeshift.setValue(0.0)
        # Timeshift is expressed in detector channels (samples), not ns.
        mem_layout.addRow("timeshift [ch]", self.spin_timeshift)

        self.spin_background = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_background.setDecimals(1)
        self.spin_background.setRange(0.0, 1e9)
        self.spin_background.setSingleStep(1.0)
        self.spin_background.setValue(0.0)
        mem_layout.addRow("background [cts]", self.spin_background)

        self.spin_irf_bg = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_irf_bg.setDecimals(3)
        self.spin_irf_bg.setRange(0.0, 1e9)
        self.spin_irf_bg.setSingleStep(1.0)
        self.spin_irf_bg.setValue(0.0)
        mem_layout.addRow("IRF background [cts]", self.spin_irf_bg)

        self.spin_lamp_scatter = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lamp_scatter.setDecimals(4)
        self.spin_lamp_scatter.setRange(0.0, 1e3)
        self.spin_lamp_scatter.setSingleStep(0.001)
        self.spin_lamp_scatter.setValue(0.0)
        mem_layout.addRow("lamp scatter", self.spin_lamp_scatter)

        self.chk_fit_nuisance = QtWidgets.QCheckBox("Fit nuisance (ts/bg/IRF BG)", mem_group)
        self.chk_fit_nuisance.setChecked(False)
        mem_layout.addRow(self.chk_fit_nuisance)

        # Single prior loading button: in Lifetime mode it loads the
        # lifetime prior, in FRET mode it loads the distance prior.
        self.btn_load_prior = QtWidgets.QToolButton(mem_group)
        self.btn_load_prior.setText("Load prior")
        self.btn_load_prior.clicked.connect(self._on_load_prior_any_clicked)
        self.label_prior_info = QtWidgets.QLabel("Prior: default 1/tau", mem_group)
        self.label_dist_prior_info = QtWidgets.QLabel("Distance prior: default flat", mem_group)
        mem_layout.addRow(self.btn_load_prior)
        mem_layout.addRow(self.label_prior_info)
        mem_layout.addRow(self.label_dist_prior_info)

        ctrl_layout.addWidget(mem_group)

        self.btn_run = QtWidgets.QToolButton(ctrl)
        self.btn_run.setText("Run MEM")
        # Make Run MEM the primary, wide green button.
        try:
            self.btn_run.setStyleSheet(
                "QToolButton { background-color: #2e8b57; color: white; font-weight: bold; }"
            )
            self.btn_run.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred,
            )
        except Exception:
            pass
        self.btn_run.clicked.connect(self._on_run_clicked)

        self.btn_save = QtWidgets.QToolButton(ctrl)
        self.btn_save.setText("Save result")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.setEnabled(False)

        self.btn_help = QtWidgets.QToolButton(ctrl)
        self.btn_help.setText("?")
        self.btn_help.setToolTip("Open MaxEnt README")
        self.btn_help.clicked.connect(self._on_help_clicked)

        row_buttons = QtWidgets.QHBoxLayout()
        row_buttons.setContentsMargins(0, 0, 0, 0)
        row_buttons.setSpacing(0)
        row_buttons.addWidget(self.btn_run)
        row_buttons.addWidget(self.btn_save)
        row_buttons.addWidget(self.btn_help)
        try:
            row_buttons.setStretch(0, 1)
            row_buttons.setStretch(1, 0)
            row_buttons.setStretch(2, 0)
        except Exception:
            pass
        ctrl_layout.addLayout(row_buttons)

        ctrl_layout.addStretch(1)

        _lazy_import_qt_stack()
        self.plot_widget = pg.GraphicsLayoutWidget(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.addWidget(ctrl)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.plot_decay = self.plot_widget.addPlot(row=0, col=0, title="Decay / fit / IRF")
        self.plot_decay.setLabel("bottom", "time", units="ns")
        self.plot_decay.setLabel("left", "counts")
        try:
            self.plot_decay.setLogMode(x=False, y=True)
        except Exception:
            pass

        self._fit_region = pg.LinearRegionItem()
        self._fit_region.setZValue(10)
        self._fit_region.sigRegionChanged.connect(self._on_fit_region_changed)
        self.plot_decay.addItem(self._fit_region)

        self.plot_wres = self.plot_widget.addPlot(row=1, col=0, title="Weighted residuals")
        self.plot_wres.setLabel("bottom", "time", units="ns")
        self.plot_wres.setLabel("left", "wres")
        try:
            self.plot_wres.setXLink(self.plot_decay)
        except Exception:
            pass

        self.plot_dist = self.plot_widget.addPlot(row=2, col=0, title="Lifetime distribution")
        self.plot_dist.setLabel("bottom", "lifetime", units="ns")
        self.plot_dist.setLabel("left", "probability")

        self.plot_lcurve = self.plot_widget.addPlot(row=3, col=0, title="L-curve (chi² vs |p|)")
        self.plot_lcurve.setLabel("bottom", "chi²")
        self.plot_lcurve.setLabel("left", "|p|")
        try:
            self.plot_lcurve.setLogMode(x=True, y=True)
        except Exception:
            pass
        try:
            self.plot_lcurve.showGrid(x=True, y=True, alpha=0.3)
        except Exception:
            pass
        self._lcurve_curve = self.plot_lcurve.plot([], [], pen=None, symbol="o")
        self._lcurve_corner = self.plot_lcurve.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolBrush="r",
            symbolPen="r",
            symbolSize=12,
        )
        try:
            self._lcurve_corner.hide()
        except Exception:
            pass

        self._update_mode_ui()

    # ------------------------------------------------------------------
    # Helpers to access current fit and IRF
    # ------------------------------------------------------------------

    def _current_fit(self):
        _lazy_import_qt_stack()
        try:
            return chisurf.cs.current_fit
        except Exception:
            return None

    def _describe_current_data(self) -> str:
        fit = self._current_fit()
        if fit is None or getattr(fit, "data", None) is None:
            return "No current fit / data"
        data = fit.data
        try:
            name = getattr(data, "name", None) or getattr(data, "filename", None)
        except Exception:
            name = None
        n_points = len(getattr(data, "y", []))
        return f"Current fit data: {name or 'unnamed'} (N={n_points})"

    def _on_refresh_data(self) -> None:
        # Changing the sample should clear any existing MEM results so the
        # user does not accidentally interpret old distributions or L-curve
        # diagnostics as belonging to the new data.
        try:
            self._reset_mem_result_state()
        except Exception:
            pass
        self.label_data_source.setText(self._describe_current_data())
        try:
            decay, dt, t = self._get_decay_and_dt()
        except Exception:
            return
        fit = self._current_fit()
        if fit is None:
            return

        try:
            xmin = int(getattr(fit, "xmin", 0))
            xmax = int(getattr(fit, "xmax", decay.size - 1))
        except Exception:
            xmin = 0
            xmax = decay.size - 1
        if decay.size > 0:
            xmin = max(0, min(xmin, decay.size - 1))
            xmax = max(xmin, min(xmax, decay.size - 1))
            self._fit_range = (xmin, xmax)
            if getattr(self, "_fit_region", None) is not None:
                self._fit_region.blockSignals(True)
                self._fit_region.setRegion((float(t[xmin]), float(t[xmax])))
                self._fit_region.blockSignals(False)

        try:
            model = fit.model
        except Exception:
            model = None
        if model is not None and hasattr(model, "convolve"):
            irf_curve = None
            try:
                convolve = model.convolve
                # Prefer the unnormalized IRF used for plotting in the
                # standard lifetime views; fall back to the normalized IRF
                # if needed.
                if hasattr(convolve, "unnormalized_irf"):
                    irf_curve = convolve.unnormalized_irf
                elif hasattr(convolve, "irf"):
                    irf_curve = convolve.irf
            except Exception:
                irf_curve = None

            if irf_curve is not None:
                self._irf_dataset = irf_curve
                try:
                    name = getattr(irf_curve, "name", None) or getattr(irf_curve, "filename", None)
                except Exception:
                    name = None
                if name:
                    self.label_irf_source.setText(f"IRF: model.convolve.irf ({name})")
                else:
                    self.label_irf_source.setText("IRF: model.convolve.irf (from current fit)")

        if model is not None:
            convolve = getattr(model, "convolve", None)
            generic = getattr(model, "generic", None)

            if convolve is not None:
                try:
                    ts_val = float(convolve.timeshift)
                except Exception:
                    ts_val = None
                if ts_val is not None:
                    try:
                        self.spin_timeshift.setValue(ts_val)
                    except Exception:
                        pass

                try:
                    irf_bg_val = float(convolve.lamp_background)
                except Exception:
                    irf_bg_val = None
                if irf_bg_val is not None:
                    try:
                        self.spin_irf_bg.setValue(irf_bg_val)
                    except Exception:
                        pass

            if generic is not None:
                try:
                    bg_val = float(generic.background)
                except Exception:
                    bg_val = None
                if bg_val is not None:
                    try:
                        self.spin_background.setValue(bg_val)
                    except Exception:
                        pass

                try:
                    scatter_val = float(generic.scatter)
                except Exception:
                    scatter_val = None
                if scatter_val is not None:
                    try:
                        self.spin_lamp_scatter.setValue(scatter_val)
                    except Exception:
                        pass

        try:
            lamp = self._build_irf_array(decay.size, t, dt)
        except Exception:
            lamp = None
        try:
            self._plot_decay_and_irf(decay, t, lamp)
        except Exception:
            pass
        if lamp is not None:
            try:
                fwhm = self._estimate_irf_fwhm(t, lamp)
            except Exception:
                fwhm = None
            if fwhm is not None and fwhm > 0.0:
                try:
                    self.spin_tau_min.setValue(float(fwhm))
                except Exception:
                    pass

        # For FRET mode, if no explicit period has been set by the user,
        # initialize the period spin box from the decay time axis
        # (approximate excitation period as the acquisition window).
        try:
            if t is not None and np.asarray(t, dtype=float).size > 1:
                t_arr = np.asarray(t, dtype=float).ravel()
                period_guess = float(t_arr[-1] - t_arr[0])
                if period_guess > 0.0:
                    self.spin_period.setValue(period_guess)
        except Exception:
            pass

    def _reset_mem_result_state(self) -> None:
        if getattr(self, "_lcurve_curve", None) is not None:
            try:
                self._lcurve_curve.setData([], [])
            except Exception:
                pass
        if getattr(self, "_lcurve_corner", None) is not None:
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
        if getattr(self, "plot_dist", None) is not None:
            try:
                self.plot_dist.clear()
            except Exception:
                pass
        if getattr(self, "plot_wres", None) is not None:
            try:
                self.plot_wres.clear()
            except Exception:
                pass
        self._last_result = None
        if getattr(self, "btn_save", None) is not None:
            try:
                self.btn_save.setEnabled(False)
            except Exception:
                pass

    def _ensure_irf_selector(self):
        _lazy_import_qt_stack()
        if getattr(self, "_irf_selector", None) is not None:
            return
        self._irf_selector = ExperimentalDataSelector(
            parent=None,
            change_event=self._on_irf_selection_changed,
            context_menu_enabled=False,
        )

    def _on_select_irf_clicked(self) -> None:
        self._ensure_irf_selector()
        try:
            self._irf_selector.show()
        except Exception:
            pass

    def _on_clear_irf_clicked(self) -> None:
        self._irf_dataset = None
        self.label_irf_source.setText("IRF: model.convolve.irf (default)")

    def _on_irf_selection_changed(self) -> None:
        try:
            ds = self._irf_selector.selected_dataset
        except Exception:
            ds = None
        self._irf_dataset = ds
        if ds is None:
            self.label_irf_source.setText("IRF: model.convolve.irf (default)")
        else:
            try:
                name = getattr(ds, "name", None) or getattr(ds, "filename", None)
            except Exception:
                name = None
            self.label_irf_source.setText(f"IRF: {name or 'dataset'}")

    def _on_mode_changed(self, index: int) -> None:
        self._mode_fret = bool(index == 1)
        self._update_mode_ui()

    def _update_mode_ui(self) -> None:
        is_fret = bool(self._mode_fret)
        lifetime_visible = not is_fret
        fret_visible = is_fret
        use_periodic = bool(self.chk_use_periodic.isChecked())

        # Lifetime-specific controls: tau grid.
        for w in (
            self.spin_tau_min,
            self.spin_tau_max,
            self.spin_tau_step,
        ):
            w.setVisible(lifetime_visible)
        label_tau = getattr(self, "_tau_grid_label", None)
        if label_tau is not None:
            label_tau.setVisible(lifetime_visible)

        # FRET-specific controls (excluding lamp scatter, which is
        # meaningful in both modes).
        for w in (
            self.spin_tau0,
            self.spin_R0,
            self.spin_R_min,
            self.spin_R_max,
            self.spin_R_points,
            self.btn_load_donor,
            self.btn_load_donor_fit,
            self.label_donor_info,
            self.spin_x_donly,
        ):
            w.setVisible(fret_visible)
        label_rda = getattr(self, "_rda_grid_label", None)
        if label_rda is not None:
            label_rda.setVisible(fret_visible)

        # tau0/R0 labels should also be hidden in lifetime mode.
        label_tau0 = getattr(self, "_tau0_label", None)
        if label_tau0 is not None:
            label_tau0.setVisible(fret_visible)
        label_R0 = getattr(self, "_R0_label", None)
        if label_R0 is not None:
            label_R0.setVisible(fret_visible)
        label_x_donly = getattr(self, "_x_donly_label", None)
        if label_x_donly is not None:
            label_x_donly.setVisible(fret_visible)

        # Period spinbox is relevant whenever periodic convolution is
        # enabled, independent of mode.
        period_visible = bool(use_periodic)
        try:
            self.spin_period.setVisible(period_visible)
        except Exception:
            pass
        label_period = getattr(self, "_period_label", None)
        if label_period is not None:
            label_period.setVisible(period_visible)

        # Combined prior button is always visible; the two labels are
        # mode specific for clarity.
        self.btn_load_prior.setVisible(True)
        self.label_prior_info.setVisible(lifetime_visible)
        self.label_dist_prior_info.setVisible(fret_visible)

    def _on_fit_region_changed(self) -> None:
        if getattr(self, "_fit_region", None) is None:
            return
        if self._t_axis is None:
            return
        t = self._t_axis
        if t.size == 0:
            return
        lb, ub = self._fit_region.getRegion()
        start = int(np.searchsorted(t, lb, side="left"))
        stop = int(np.searchsorted(t, ub, side="right") - 1)
        n = t.size
        if n <= 0:
            return
        start = max(0, min(start, n - 1))
        stop = max(start, min(stop, n - 1))
        self._fit_range = (start, stop)

    # ------------------------------------------------------------------
    # Prior loading
    # ------------------------------------------------------------------

    def _on_load_prior_any_clicked(self) -> None:
        """Load prior depending on current mode.

        - Lifetime mode: load lifetime prior over tau.
        - FRET mode: load distance prior over R.
        """
        if bool(self._mode_fret):
            self._on_load_dist_prior_clicked()
        else:
            self._on_load_prior_clicked()

    def _on_load_prior_clicked(self) -> None:
        _lazy_import_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load MEM prior (text/CSV)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=1)
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("empty prior")
            self._prior_vec = arr
            self.label_prior_info.setText(
                f"Prior: loaded {arr.size} values from '{Path(fn).name}'"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading prior",
                f"Failed to load prior from '{fn}':\n{exc}",
            )

    def _on_load_donor_clicked(self) -> None:
        _lazy_import_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load donor spectrum (amp/tau pairs)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=2)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            if arr.shape[1] < 2:
                raise ValueError("Donor spectrum must contain amplitude/lifetime pairs")
            flat = arr[:, :2].astype(float).ravel()
            if flat.size % 2 != 0:
                raise ValueError("Donor spectrum must contain amplitude/lifetime pairs")
            self._donly_vec = flat
            n_pairs = flat.size // 2
            self.label_donor_info.setText(
                f"Donor spectrum: {n_pairs} amp/tau pairs from '{Path(fn).name}'"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading donor spectrum",
                f"Failed to load donor spectrum from '{fn}':\n{exc}",
            )

    def _on_load_donor_from_fit_clicked(self) -> None:
        _lazy_import_qt_stack()
        try:
            import chisurf as _cs  # type: ignore
        except Exception:
            QtWidgets.QMessageBox.warning(
                self,
                "Load donor spectrum",
                "chisurf module is not available.",
            )
            return

        try:
            from chisurf.models.tcspc.lifetime import LifetimeModel  # type: ignore
        except Exception:
            LifetimeModel = None  # type: ignore[assignment]
        try:
            from chisurf.models.tcspc.fret import FRETModel  # type: ignore
        except Exception:
            FRETModel = None  # type: ignore[assignment]

        candidates = []
        labels = []

        try:
            fit_list = list(getattr(_cs, "fits", []) or [])
        except Exception:
            fit_list = []

        for fg in fit_list:
            try:
                for fit in fg:
                    model = getattr(fit, "model", None)
                    if model is None:
                        continue
                    arr = None
                    try:
                        if LifetimeModel is not None and isinstance(model, LifetimeModel):
                            arr = np.asarray(model.lifetime_spectrum, dtype=float).ravel()
                        elif FRETModel is not None and isinstance(model, FRETModel):
                            arr = np.asarray(model.donor_lifetime_spectrum, dtype=float).ravel()
                    except Exception:
                        arr = None
                    if arr is None or arr.size < 2 or arr.size % 2 != 0:
                        continue
                    labels.append(getattr(fit, "name", None) or "Fit")
                    candidates.append(arr)
            except Exception:
                continue

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "Load donor spectrum",
                "No lifetime/FRET fits with a donor spectrum were found.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select donor lifetime fit")
        layout = QtWidgets.QVBoxLayout(dialog)

        combo = QtWidgets.QComboBox(dialog)
        for label, arr in zip(labels, candidates):
            combo.addItem(label, arr)
        layout.addWidget(combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        idx = combo.currentIndex()
        if idx < 0:
            return

        try:
            arr = np.asarray(combo.itemData(idx), dtype=float).ravel()
        except Exception:
            arr = np.zeros(0, dtype=float)
        if arr.size < 2 or arr.size % 2 != 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Load donor spectrum",
                "Selected fit does not provide a valid (amplitude, lifetime) donor spectrum.",
            )
            return

        self._donly_vec = arr
        n_pairs = arr.size // 2
        label = combo.currentText()
        self.label_donor_info.setText(
            f"Donor spectrum: {n_pairs} amp/tau pairs from fit '{label}'"
        )

    def _on_load_dist_prior_clicked(self) -> None:
        _lazy_import_qt_stack()
        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load distance prior (text/CSV)",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not fn:
            return
        try:
            arr = np.loadtxt(fn, dtype=float, ndmin=1)
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("empty prior")
            self._dist_prior_vec = arr
            self.label_dist_prior_info.setText(
                f"Distance prior: loaded {arr.size} values from '{Path(fn).name}'"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading distance prior",
                f"Failed to load distance prior from '{fn}':\n{exc}",
            )

    def _build_tau_grid(self) -> np.ndarray:
        tmin = float(self.spin_tau_min.value())
        tmax = float(self.spin_tau_max.value())
        step = float(self.spin_tau_step.value())
        if tmax <= tmin or step <= 0.0:
            raise ValueError("Invalid tau grid parameters")
        n_steps = int(np.floor((tmax - tmin) / step)) + 1
        grid = tmin + step * np.arange(n_steps, dtype=float)
        return grid

    def _get_decay_and_dt(self) -> Tuple[np.ndarray, float, np.ndarray]:
        fit = self._current_fit()
        if fit is None or getattr(fit, "data", None) is None:
            raise RuntimeError("No current fit / data available")
        data = fit.data
        y = np.asarray(getattr(data, "y", []), dtype=float).ravel()
        if y.size == 0:
            raise RuntimeError("Current data has no counts")
        try:
            x = np.asarray(getattr(data, "x", None), dtype=float).ravel()
        except Exception:
            x = np.zeros_like(y)
        if x.size != y.size or x.size == 0:
            try:
                dx = float(np.asarray(data.dx).ravel()[0])
            except Exception:
                dx = 1.0
            x = dx * np.arange(y.size, dtype=float)
        else:
            dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
        self._t_axis = x
        return y, dx, x

    def _build_irf_array(self, n: int, t: np.ndarray, dt: float) -> np.ndarray:
        if self._irf_dataset is not None:
            try:
                x_irf = np.asarray(getattr(self._irf_dataset, "x", []), dtype=float).ravel()
                y_irf = np.asarray(getattr(self._irf_dataset, "y", []), dtype=float).ravel()
            except Exception:
                x_irf = np.zeros(0, dtype=float)
                y_irf = np.zeros(0, dtype=float)
            if x_irf.size > 1 and y_irf.size == x_irf.size:
                order = np.argsort(x_irf)
                x_sorted = x_irf[order]
                y_sorted = y_irf[order]
                lamp = np.interp(t, x_sorted, y_sorted, left=0.0, right=0.0)
                return lamp.astype(float)

        fit = self._current_fit()
        if fit is not None:
            try:
                model = fit.model
                if hasattr(model, "convolve"):
                    convolve = model.convolve
                    irf_curve = None
                    if hasattr(convolve, "unnormalized_irf"):
                        irf_curve = convolve.unnormalized_irf
                    elif hasattr(convolve, "irf"):
                        irf_curve = convolve.irf
                    if irf_curve is not None:
                        y_irf = np.asarray(irf_curve.y, dtype=float).ravel()
                        if y_irf.size > 0:
                            lamp = np.resize(y_irf, n)
                            return lamp
            except Exception:
                pass

        lamp = np.zeros(n, dtype=float)
        lamp[0] = 1.0
        return lamp

    def _estimate_irf_fwhm(self, t: np.ndarray, lamp: np.ndarray) -> Optional[float]:
        t_arr = np.asarray(t, dtype=float).ravel()
        lamp_arr = np.asarray(lamp, dtype=float).ravel()
        if t_arr.size != lamp_arr.size or t_arr.size < 3:
            return None
        max_val = float(np.max(lamp_arr))
        if not np.isfinite(max_val) or max_val <= 0.0:
            return None
        half = 0.5 * max_val
        above = np.nonzero(lamp_arr >= half)[0]
        if above.size < 2:
            if t_arr.size >= 2:
                dt_local = float(np.mean(np.diff(t_arr)))
                return dt_local if dt_local > 0.0 else None
            return None
        fwhm = float(t_arr[above[-1]] - t_arr[above[0]]) / 2.0
        if fwhm <= 0.0 and t_arr.size >= 2:
            dt_local = float(np.mean(np.diff(t_arr)))
            fwhm = dt_local
        return fwhm if fwhm > 0.0 else None

    def _plot_decay_and_irf(
        self,
        decay: np.ndarray,
        t: np.ndarray,
        lamp: Optional[np.ndarray],
    ) -> None:
        _lazy_import_qt_stack()

        self.plot_decay.clear()
        if getattr(self, "_fit_region", None) is not None:
            self.plot_decay.addItem(self._fit_region)

        decay_plot = np.maximum(np.asarray(decay, dtype=float).ravel(), 1.0)

        irf_plot = None
        if lamp is not None:
            lamp_arr = np.asarray(lamp, dtype=float).ravel()
            if lamp_arr.size == decay_plot.size and np.any(lamp_arr > 0.0):
                scale = float(np.max(decay_plot)) / float(np.max(lamp_arr))
                irf_plot = lamp_arr * scale

        self.plot_decay.plot(t, decay_plot, pen="w", name="data")
        if irf_plot is not None:
            self.plot_decay.plot(t, np.maximum(irf_plot, 1.0), pen="r", name="IRF")

        self.plot_wres.clear()
        self.plot_dist.clear()

    def _on_help_clicked(self) -> None:
        _lazy_import_qt_stack()
        try:
            readme_path = Path(__file__).with_name("README.md")
        except Exception:
            readme_path = None

        if readme_path is None or not readme_path.is_file():
            try:
                QtWidgets.QMessageBox.information(
                    self,
                    "MaxEnt help",
                    "README.md not found next to the plugin.",
                )
            except Exception:
                pass
            return

        try:
            text = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            try:
                QtWidgets.QMessageBox.critical(
                    self,
                    "MaxEnt help",
                    f"Failed to read README.md:\n{exc}",
                )
            except Exception:
                pass
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("MaxEnt MEM README")
        layout = QtWidgets.QVBoxLayout(dialog)

        text_edit = QtWidgets.QPlainTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        try:
            dialog.resize(800, 600)
        except Exception:
            pass
        dialog.exec_()

    def _on_run_clicked(self) -> None:
        try:
            self._run_mem()
        except Exception as exc:
            _lazy_import_qt_stack()
            if str(exc) == "MEM computation cancelled":
                return
            QtWidgets.QMessageBox.critical(self, "MEM error", str(exc))

    def _on_save_clicked(self) -> None:
        _lazy_import_qt_stack()
        if self._last_result is None or self._t_axis is None:
            QtWidgets.QMessageBox.warning(self, "Save MEM result", "No MEM result available to save.")
            return

        result = self._last_result

        try:
            decay, dt, t = self._get_decay_and_dt()
        except Exception:
            decay = None
            dt = None
            t = self._t_axis

        try:
            fit = self._current_fit()
        except Exception:
            fit = None

        start_dir = ""
        try:
            start_dir = str(getattr(chisurf, "working_path", "") or "")
        except Exception:
            start_dir = ""

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder to save MEM result",
            start_dir,
        )
        if not target_dir:
            return

        out_dir = Path(target_dir)

        try:
            dist_axis = np.asarray(result.get("R", result.get("tau")), dtype=float).ravel()
            p = np.asarray(result["p"], dtype=float).ravel()
        except Exception:
            dist_axis = np.zeros(0, dtype=float)
            p = np.zeros(0, dtype=float)

        try:
            Fi = np.asarray(result["Fi"], dtype=float)
            y_seg = np.asarray(result["y"], dtype=float).ravel()
            sigma = np.asarray(result["sigma"], dtype=float).ravel()
            fitstart, fitstop = result["fitrange"]
            fit_seg = (Fi @ p) * sigma
            wres = (y_seg - fit_seg) / sigma
        except Exception:
            Fi = None
            y_seg = None
            sigma = None
            fitstart = fitstop = None
            fit_seg = None
            wres = None

        decay_arr = None
        if decay is not None and t is not None:
            decay_arr = np.asarray(decay, dtype=float).ravel()

        try:
            lamp = self._build_irf_array(decay_arr.size if decay_arr is not None else result.get("y", np.zeros(0)).size, t, float(dt) if dt is not None else 1.0)
        except Exception:
            lamp = None

        try:
            meta = {
                "mode": "FRET" if "R" in result else "lifetime",
                "nu": float(result.get("nu", result.get("nu_input", 0.0))),
                "chisq": float(result.get("chisq", 0.0)),
                "S": float(result.get("S", 0.0)),
                "Q": float(result.get("Q", 0.0)),
                "fitrange": [int(x) for x in result.get("fitrange", (0, 0))],
                "dt": float(result.get("dt", dt if dt is not None else 0.0)),
                "tau0": float(result.get("tau0", self.spin_tau0.value())),
                "R0": float(result.get("R0", self.spin_R0.value())),
                "R_min": float(self.spin_R_min.value()),
                "R_max": float(self.spin_R_max.value()),
                "R_points": int(self.spin_R_points.value()),
                "tau_min": float(self.spin_tau_min.value()),
                "tau_max": float(self.spin_tau_max.value()),
                "tau_step": float(self.spin_tau_step.value()),
                "fit_start_fraction": float(self.spin_start_frac.value()),
                "fit_nuisance": bool(self.chk_fit_nuisance.isChecked()),
            }
        except Exception:
            meta = {}

        try:
            if dist_axis.size and p.size:
                np.savetxt(out_dir / "distribution.txt", np.column_stack([dist_axis, p]), header="axis  p")
        except Exception:
            pass

        try:
            if decay_arr is not None and t is not None:
                if fitstart is not None and fitstop is not None and fit_seg is not None:
                    full_fit = np.zeros_like(decay_arr)
                    full_fit[fitstart : fitstop + 1] = fit_seg
                    arr = np.column_stack([t, decay_arr, full_fit])
                    header = "time  decay  mem_fit"
                else:
                    arr = np.column_stack([t, decay_arr])
                    header = "time  decay"
                np.savetxt(out_dir / "decay_fit.txt", arr, header=header)
        except Exception:
            pass

        try:
            if lamp is not None and t is not None:
                lamp_arr = np.asarray(lamp, dtype=float).ravel()
                if lamp_arr.size == np.asarray(t, dtype=float).ravel().size:
                    np.savetxt(out_dir / "irf.txt", np.column_stack([t, lamp_arr]), header="time  irf")
        except Exception:
            pass

        try:
            if wres is not None and t is not None and fitstart is not None and fitstop is not None:
                t_seg = np.asarray(t, dtype=float).ravel()[fitstart : fitstop + 1]
                wres_arr = np.asarray(wres, dtype=float).ravel()
                if t_seg.size == wres_arr.size:
                    np.savetxt(out_dir / "wres.txt", np.column_stack([t_seg, wres_arr]), header="time  wres")
        except Exception:
            pass

        try:
            with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception:
            pass

        QtWidgets.QMessageBox.information(self, "Save MEM result", f"Saved MEM result to:\n{out_dir}")

    def _on_lcurve_clicked(self) -> None:
        try:
            self._run_lcurve()
        except Exception as exc:
            _lazy_import_qt_stack()
            QtWidgets.QMessageBox.critical(self, "L-curve error", str(exc))

    def _run_lcurve(self) -> None:
        _lazy_import_qt_stack()
        decay, dt, t = self._get_decay_and_dt()
        lamp = self._build_irf_array(decay.size, t, dt)

        is_fret = bool(self._mode_fret)

        fit_start_fraction = float(self.spin_start_frac.value())
        use_periodic = bool(self.chk_use_periodic.isChecked())

        fitrange = self._fit_range
        if fitrange is None:
            fit = self._current_fit()
            if fit is not None:
                try:
                    xmin = int(getattr(fit, "xmin", 0))
                    xmax = int(getattr(fit, "xmax", decay.size - 1))
                    fitrange = (xmin, xmax)
                except Exception:
                    fitrange = None

        fitrange_arg = None
        if fitrange is not None and decay.size > 0:
            fs, fe = int(fitrange[0]), int(fitrange[1])
            if fe < fs:
                fe = fs
            n = decay.size
            fs = max(0, min(fs, n - 1))
            fe = max(fs, min(fe, n - 1))
            fitrange_arg = (fs, fe)

        r_axis = None
        tau0_val = None
        R0_val = None
        period_val = None
        x_donly_val = None
        donly_vec = None
        tau = None
        prior_vec: Optional[Sequence[float]] = None

        if is_fret:
            tau0_val = float(self.spin_tau0.value())
            R0_val = float(self.spin_R0.value())
            period_val = max(float(self.spin_period.value()), 1e-3)
            x_donly_val = float(self.spin_x_donly.value())
            r_min = float(self.spin_R_min.value())
            r_max = float(self.spin_R_max.value())
            n_points_r = int(self.spin_R_points.value())
            if n_points_r < 2:
                n_points_r = 2
            if r_max <= r_min:
                r_max = r_min + 1e-3
            r_axis = np.linspace(r_min, r_max, n_points_r, dtype=float)
            if self._donly_vec is not None:
                donly_vec = self._donly_vec
            else:
                donly_vec = np.array([1.0, float(tau0_val)], dtype=float)
            prior = self._dist_prior_vec if self._dist_prior_vec is not None and self._dist_prior_vec.size == r_axis.size else None
        else:
            tau = self._build_tau_grid()
            if self._prior_vec is not None and self._prior_vec.size == tau.size:
                prior_vec = self._prior_vec

        optimize_nuisance = bool(self.chk_fit_nuisance.isChecked()) if not is_fret else False

        nu0 = float(self.spin_nu.value())
        if nu0 <= 0.0:
            nu0 = 1e-5
        log10_center = float(np.log10(nu0))
        try:
            dec_left = float(self.spin_lcurve_dec_left.value())
            dec_right = float(self.spin_lcurve_dec_right.value())
        except Exception:
            dec_left = 2.0
            dec_right = 2.0
        if dec_left < 0.0:
            dec_left = 0.0
        if dec_right < 0.0:
            dec_right = 0.0
        log10_min = log10_center - dec_left
        log10_max = log10_center + dec_right
        if (not np.isfinite(log10_min)) or (not np.isfinite(log10_max)) or log10_min >= log10_max:
            log10_min = log10_center - 2.0
            log10_max = log10_center + 2.0
        n_nu = 16
        log_grid = np.linspace(log10_min, log10_max, n_nu)
        nu_grid = 10.0 ** log_grid

        chi2_vals = np.empty_like(nu_grid, dtype=float)
        sol_vals = np.empty_like(nu_grid, dtype=float)

        progress = QtWidgets.QProgressDialog("Computing L-curve...", "Cancel", 0, int(n_nu), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            for i, nu_val in enumerate(nu_grid):
                if progress.wasCanceled():
                    raise RuntimeError("L-curve computation cancelled")
                try:
                    if is_fret:
                        res = mem_vin4_fret(
                            decay=decay,
                            lamp=lamp,
                            dt=dt,
                            period=period_val,
                            use_periodic=use_periodic,
                            R=r_axis,
                            tau0=float(tau0_val),
                            R0=float(R0_val),
                            donly=donly_vec,
                            x_donly=float(x_donly_val) if x_donly_val is not None else 0.0,
                            fitrange=fitrange_arg,
                            fit_start_fraction=fit_start_fraction,
                            nu=float(nu_val),
                            max_iter=200,
                            prior=prior,
                            progress_cb=None,
                        )
                    else:
                        res = mem_vin4_lifetime(
                            decay=decay,
                            lamp=lamp,
                            dt=dt,
                            tau=tau,
                            lamp_scatter=float(self.spin_lamp_scatter.value()),
                            use_periodic=use_periodic,
                            fitrange=fitrange_arg,
                            fit_start_fraction=fit_start_fraction,
                            nu=float(nu_val),
                            optimize_nuisance=optimize_nuisance,
                            prior=prior_vec,
                            max_iter=200,
                            progress_cb=None,
                        )
                    chi2_vals[i] = float(res.get("chisq", np.nan))
                    p_res = np.asarray(res.get("p", []), dtype=float).ravel()
                    if p_res.size == 0:
                        sol_vals[i] = np.nan
                    else:
                        sol_vals[i] = float(np.linalg.norm(p_res))
                except Exception:
                    chi2_vals[i] = np.nan
                    sol_vals[i] = np.nan
                progress.setValue(int(i + 1))
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
        finally:
            progress.close()

        corner_idx = None
        try:
            mask = np.isfinite(chi2_vals) & np.isfinite(sol_vals)
            if np.any(mask) and getattr(chisurf, "math", None) is not None:
                try:
                    corner_idx = chisurf.math.regularization.discrete_lcurve_corner(chi2_vals, sol_vals)
                except Exception:
                    corner_idx = None
        except Exception:
            corner_idx = None

        if corner_idx is not None and 0 <= int(corner_idx) < nu_grid.size:
            try:
                self.spin_nu.setValue(float(nu_grid[int(corner_idx)]))
            except Exception:
                pass

        self._update_lcurve_plot(chi2_vals, sol_vals, nu_grid, corner_idx)

    def _update_lcurve_plot(
        self,
        chi2_vals: np.ndarray,
        sol_vals: np.ndarray,
        nu_grid: np.ndarray,
        corner_idx: Optional[int],
    ) -> None:
        if getattr(self, "plot_lcurve", None) is None:
            return
        try:
            chi2_arr = np.asarray(chi2_vals, dtype=float).ravel()
            sol_arr = np.asarray(sol_vals, dtype=float).ravel()
        except Exception:
            chi2_arr = np.array([], dtype=float)
            sol_arr = np.array([], dtype=float)
        if chi2_arr.size == 0 or sol_arr.size == 0:
            self._lcurve_curve.setData([], [])
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
            return
        mask = np.isfinite(chi2_arr) & np.isfinite(sol_arr) & (chi2_arr > 0.0) & (sol_arr > 0.0)
        if not np.any(mask):
            self._lcurve_curve.setData([], [])
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
            return
        idx_all = np.nonzero(mask)[0]
        chi2_plot = chi2_arr[mask]
        sol_plot = sol_arr[mask]
        eps = np.finfo(float).tiny
        chi2_plot = np.clip(chi2_plot, eps, np.inf)
        sol_plot = np.clip(sol_plot, eps, np.inf)
        self._lcurve_curve.setData(chi2_plot, sol_plot)

        try:
            self._lcurve_corner.hide()
        except Exception:
            pass

        if corner_idx is None:
            return
        try:
            matches = np.nonzero(idx_all == int(corner_idx))[0]
            if matches.size == 0:
                return
            local_idx = int(matches[0])
            if 0 <= local_idx < chi2_plot.size:
                self._lcurve_corner.setData(
                    [float(chi2_plot[local_idx])],
                    [float(sol_plot[local_idx])],
                )
                try:
                    self._lcurve_corner.show()
                except Exception:
                    pass
        except Exception:
            return

    def _run_mem(self) -> None:
        _lazy_import_qt_stack()
        decay, dt, t = self._get_decay_and_dt()
        lamp = self._build_irf_array(decay.size, t, dt)

        is_fret = bool(self._mode_fret)

        if not is_fret:
            tau = self._build_tau_grid()
        nu = float(self.spin_nu.value())
        fit_start_fraction = float(self.spin_start_frac.value())
        use_periodic = bool(self.chk_use_periodic.isChecked())

        prior_vec: Optional[Sequence[float]] = None
        if (not is_fret) and self._prior_vec is not None:
            if self._prior_vec.size != tau.size:
                raise ValueError(
                    f"Prior length {self._prior_vec.size} does not match tau grid length {tau.size}"
                )
            prior_vec = self._prior_vec

        optimize_nuisance = bool(self.chk_fit_nuisance.isChecked())
        max_iter = 200

        fitrange = self._fit_range
        if fitrange is None:
            fit = self._current_fit()
            if fit is not None:
                try:
                    xmin = int(getattr(fit, "xmin", 0))
                    xmax = int(getattr(fit, "xmax", decay.size - 1))
                    fitrange = (xmin, xmax)
                except Exception:
                    fitrange = None

        fitrange_arg = None
        if fitrange is not None and decay.size > 0:
            fs, fe = int(fitrange[0]), int(fitrange[1])
            if fe < fs:
                fe = fs
            n = decay.size
            fs = max(0, min(fs, n - 1))
            fe = max(fs, min(fe, n - 1))
            fitrange_arg = (fs, fe)

        r_axis = None
        tau0_val = None
        R0_val = None
        period_val = None
        x_donly_val = None
        if is_fret:
            tau0_val = float(self.spin_tau0.value())
            R0_val = float(self.spin_R0.value())
            period_val = max(float(self.spin_period.value()), 1e-3)
            x_donly_val = float(self.spin_x_donly.value())
            r_min = float(self.spin_R_min.value())
            r_max = float(self.spin_R_max.value())
            n_points = int(self.spin_R_points.value())
            if n_points < 2:
                n_points = 2
            if r_max <= r_min:
                r_max = r_min + 1e-3
            r_axis = np.linspace(r_min, r_max, n_points, dtype=float)

        # Timeshift spinbox is in detector channels; convert to ns for the
        # MEM core, which expects a time shift in the same units as ``dt``.
        ts_channels = float(self.spin_timeshift.value())
        ts_val = float(ts_channels * dt)
        bg_val = float(self.spin_background.value())
        irf_bg_input = float(self.spin_irf_bg.value())
        irf_bg_arg: Optional[float]
        if irf_bg_input > 0.0:
            irf_bg_arg = irf_bg_input
        else:
            irf_bg_arg = None
        lamp_scatter_val = float(self.spin_lamp_scatter.value())

        progress = QtWidgets.QProgressDialog("Running MEM...", "Cancel", 0, max_iter, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def _progress_cb(iter_idx, chisq, S, Q, dgrad):
            val = int(iter_idx)
            if val < 0:
                val = 0
            if val > max_iter:
                val = max_iter
            progress.setValue(val)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            if progress.wasCanceled():
                raise RuntimeError("MEM computation cancelled")

        try:
            if is_fret:
                if self._donly_vec is not None:
                    donly_vec = self._donly_vec
                else:
                    donly_vec = np.array([1.0, float(tau0_val)], dtype=float)
                prior = self._dist_prior_vec if self._dist_prior_vec is not None and self._dist_prior_vec.size == r_axis.size else None
                result = mem_vin4_fret(
                    decay=decay,
                    lamp=lamp,
                    dt=dt,
                    period=period_val,
                    use_periodic=use_periodic,
                    R=r_axis,
                    tau0=float(tau0_val),
                    R0=float(R0_val),
                    donly=donly_vec,
                    x_donly=float(x_donly_val) if x_donly_val is not None else 0.0,
                    timeshift=float(ts_val),
                    background=float(bg_val),
                    lamp_scatter=float(lamp_scatter_val),
                    irf_background=irf_bg_arg,
                    fitrange=fitrange_arg,
                    fit_start_fraction=fit_start_fraction,
                    nu=nu,
                    max_iter=max_iter,
                    prior=prior,
                    progress_cb=_progress_cb,
                    optimize_nuisance=optimize_nuisance,
                )
            else:
                tau = self._build_tau_grid()
                result = mem_vin4_lifetime(
                    decay=decay,
                    lamp=lamp,
                    dt=dt,
                    tau=tau,
                    lamp_scatter=lamp_scatter_val,
                    use_periodic=use_periodic,
                    timeshift=float(ts_val),
                    background=float(bg_val),
                    irf_background=irf_bg_arg,
                    fitrange=fitrange_arg,
                    fit_start_fraction=fit_start_fraction,
                    nu=nu,
                    optimize_nuisance=optimize_nuisance,
                    prior=prior_vec,
                    max_iter=max_iter,
                    progress_cb=_progress_cb,
                )
        finally:
            progress.close()
        self._last_result = result
        if getattr(self, "btn_save", None) is not None:
            self.btn_save.setEnabled(True)
        try:
            if not is_fret:
                if "timeshift" in result:
                    # Convert MEM timeshift (ns) back to channels for the GUI.
                    self.spin_timeshift.setValue(float(result["timeshift"]) / float(dt))
                if "background" in result:
                    self.spin_background.setValue(float(result["background"]))
                if "irf_background" in result:
                    self.spin_irf_bg.setValue(float(result["irf_background"]))
            else:
                if "timeshift" in result:
                    # Convert MEM timeshift (ns) back to channels for the GUI.
                    self.spin_timeshift.setValue(float(result["timeshift"]) / float(dt))
                if "background" in result:
                    self.spin_background.setValue(float(result["background"]))
                if "lamp_scatter" in result:
                    self.spin_lamp_scatter.setValue(float(result["lamp_scatter"]))
                if "irf_background" in result:
                    self.spin_irf_bg.setValue(float(result["irf_background"]))
        except Exception:
            pass
        self._update_plots_from_result(decay, t, result)

    def _update_plots_from_result(
        self,
        decay: np.ndarray,
        t: np.ndarray,
        result: dict,
    ) -> None:
        _lazy_import_qt_stack()

        if "R" in result:
            dist_axis = np.asarray(result["R"], dtype=float).ravel()
            self.plot_dist.setLabel("bottom", "distance", units="\u00c5")
            self.plot_dist.setTitle("Distance distribution")
        else:
            dist_axis = np.asarray(result["tau"], dtype=float).ravel()
            self.plot_dist.setLabel("bottom", "lifetime", units="ns")
            self.plot_dist.setTitle("Lifetime distribution")

        p = np.asarray(result["p"], dtype=float).ravel()
        Fi = np.asarray(result["Fi"], dtype=float)
        y_seg = np.asarray(result["y"], dtype=float).ravel()
        sigma = np.asarray(result["sigma"], dtype=float).ravel()
        fitstart, fitstop = result["fitrange"]

        s = float(p.sum())
        if s > 0.0:
            p_norm = p / s
        else:
            p_norm = p

        fit_seg = (Fi @ p) * sigma
        wres = (y_seg - fit_seg) / sigma

        t_seg = t[fitstart : fitstop + 1]

        self.plot_decay.clear()
        if getattr(self, "_fit_region", None) is not None:
            self.plot_decay.addItem(self._fit_region)
        decay_plot = np.maximum(decay, 1.0)
        fit_plot = np.zeros_like(decay_plot)
        fit_plot[fitstart : fitstop + 1] = np.maximum(fit_seg, 1.0)

        self.plot_decay.plot(t, decay_plot, pen="w", name="data")
        self.plot_decay.plot(t, fit_plot, pen="y", name="MEM fit")

        self.plot_wres.clear()
        self.plot_wres.plot(t_seg, wres, pen="c")
        self.plot_wres.addLine(y=0.0, pen=pg.mkPen("w", width=1))

        self.plot_dist.clear()
        self.plot_dist.plot(dist_axis, p_norm, pen="m", symbol="o", symbolSize=4)


__all__ = ["MaxentDecayWidget"]
