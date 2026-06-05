from __future__ import annotations

import sys

import numpy as np

from chisurf.plugins.fluorescence_decay.maxent_decay.fmem import settings as maxent_settings
from .qt_stack import ensure_qt_stack


class _MaxentUIMixin:
    def _init_ui(self) -> None:
        pg, QtWidgets, QtCore, chisurf, _ = ensure_qt_stack()

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        ctrl = QtWidgets.QWidget(self)
        # Keep the control panel narrow so that plots have more
        # horizontal space.
        try:
            ctrl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            ctrl.setFixedWidth(250)
        except Exception:
            pass
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(0)

        data_group = QtWidgets.QGroupBox("Data / IRF", ctrl)
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(4, 4, 4, 4)
        data_layout.setSpacing(2)

        self.label_data_source = QtWidgets.QLabel(
            "Using cs.current_fit.data (not yet checked)", data_group
        )
        self.btn_refresh_data = QtWidgets.QToolButton(data_group)
        self.btn_refresh_data.setText("Read from fit")
        # Emphasize data refresh as a prominent action.
        try:
            self.btn_refresh_data.setStyleSheet(
                "QToolButton { background-color: #c67e3b; color: white; font-weight: bold; }"
            )
        except Exception:
            pass
        self.btn_refresh_data.clicked.connect(self._on_refresh_data)

        # IRF controls live in the same group as the data controls so
        # everything related to the current fit is in one place.
        self.label_irf_source = QtWidgets.QLabel("IRF: model.convolve.irf (default)", data_group)
        self.btn_select_irf = QtWidgets.QToolButton(data_group)
        self.btn_select_irf.setText("Select IRF dataset")

        # Make IRF selection buttons visually stand out.
        try:
            red_style = "QToolButton { background-color: #c67e3b; color: white; }"
            self.btn_select_irf.setStyleSheet(red_style)
        except Exception:
            pass

        self.btn_select_irf.clicked.connect(self._on_select_irf_clicked)

        # Button to edit the underlying JSON settings file with a simple
        # text editor. This allows advanced users to tweak defaults that are
        # not directly exposed in the GUI.
        self.btn_edit_settings = QtWidgets.QToolButton(data_group)
        self.btn_edit_settings.setText("Edit JSON settings")
        try:
            self.btn_edit_settings.setToolTip(
                f"Open MaxEnt JSON settings file:\n{maxent_settings.get_settings_file()}"
            )
        except Exception:
            pass
        self.btn_edit_settings.clicked.connect(self._on_edit_settings_clicked)

        # Bundle the top actions (data/IRF/JSON) into a single row so the
        # control panel reads more compactly.
        grid_top_btns = QtWidgets.QGridLayout()
        grid_top_btns.setContentsMargins(0, 0, 0, 0)
        grid_top_btns.setHorizontalSpacing(4)
        grid_top_btns.setVerticalSpacing(2)
        grid_top_btns.addWidget(self.btn_refresh_data, 0, 0)
        grid_top_btns.addWidget(self.btn_select_irf, 0, 1)
        grid_top_btns.addWidget(self.btn_edit_settings, 1, 0, 1, 2)
        try:
            grid_top_btns.setColumnStretch(0, 1)
            grid_top_btns.setColumnStretch(1, 1)
        except Exception:
            pass

        data_layout.addWidget(self.label_data_source)
        data_layout.addLayout(grid_top_btns)
        # IRF label below the action row.
        data_layout.addWidget(self.label_irf_source)
        ctrl_layout.addWidget(data_group)

        mem_group = QtWidgets.QGroupBox("MEM settings", ctrl)
        mem_layout = QtWidgets.QFormLayout(mem_group)
        mem_layout.setContentsMargins(0, 0, 0, 0)
        mem_layout.setSpacing(0)
        try:
            mem_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        except Exception:
            pass

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
        try:
            row_nu.setStretch(0, 1)
            row_nu.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("nu (reg)", row_nu)

        self.spin_lcurve_dec_left = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lcurve_dec_left.setDecimals(2)
        self.spin_lcurve_dec_left.setRange(0.0, 6.0)
        self.spin_lcurve_dec_left.setSingleStep(0.5)
        try:
            lc_left = float(
                ((self._settings.get("lcurve_span_decades", {}) or {}).get("left", 2.0))
            )
        except Exception:
            lc_left = 2.0
        self.spin_lcurve_dec_left.setValue(lc_left)
        self.spin_lcurve_dec_left.setToolTip(
            "Decades below the current nu used when scanning the L-curve."
        )
        self.spin_lcurve_dec_right = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lcurve_dec_right.setDecimals(2)
        self.spin_lcurve_dec_right.setRange(0.0, 6.0)
        self.spin_lcurve_dec_right.setSingleStep(0.5)
        try:
            lc_right = float(
                ((self._settings.get("lcurve_span_decades", {}) or {}).get("right", 2.0))
            )
        except Exception:
            lc_right = 2.0
        self.spin_lcurve_dec_right.setValue(lc_right)
        self.spin_lcurve_dec_right.setToolTip(
            "Decades above the current nu used when scanning the L-curve."
        )
        row_lc = QtWidgets.QHBoxLayout()
        row_lc.setContentsMargins(0, 0, 0, 0)
        row_lc.setSpacing(0)
        row_lc.addWidget(self.spin_lcurve_dec_left)
        row_lc.addWidget(self.spin_lcurve_dec_right)
        try:
            row_lc.setStretch(0, 1)
            row_lc.setStretch(1, 1)
        except Exception:
            pass
        mem_layout.addRow("L-curve span [dec]", row_lc)

        self.spin_tau_min = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau_min.setDecimals(3)
        self.spin_tau_min.setRange(1e-3, 1e3)
        try:
            tau_min = float(
                ((self._settings.get("tau_grid", {}) or {}).get("min", 0.01))
            )
        except Exception:
            tau_min = 0.01
        self.spin_tau_min.setValue(tau_min)
        self.spin_tau_max = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_tau_max.setDecimals(3)
        self.spin_tau_max.setRange(1e-3, 1e3)
        try:
            tau_max = float(
                ((self._settings.get("tau_grid", {}) or {}).get("max", 6.0))
            )
        except Exception:
            tau_max = 6.0
        self.spin_tau_max.setValue(tau_max)
        self.spin_tau_bins = QtWidgets.QSpinBox(mem_group)
        self.spin_tau_bins.setRange(2, 10000)
        try:
            tg = (self._settings.get("tau_grid", {}) or {})
            if "bins" in tg:
                tau_bins = int(tg.get("bins", 192))
            else:
                step_val = float(tg.get("step", 0.02))
                if step_val > 0.0 and tau_max > tau_min:
                    tau_bins = int(np.floor((tau_max - tau_min) / step_val)) + 1
                else:
                    tau_bins = 192
        except Exception:
            tau_bins = 192
        if tau_bins < 2:
            tau_bins = 2
        self.spin_tau_bins.setValue(tau_bins)

        grid_tau = QtWidgets.QVBoxLayout()
        grid_tau.setContentsMargins(0, 0, 0, 0)
        grid_tau.setSpacing(0)

        row_tau_min = QtWidgets.QHBoxLayout()
        row_tau_min.setContentsMargins(0, 0, 0, 0)
        row_tau_min.setSpacing(0)
        row_tau_min.addWidget(self.spin_tau_min)
        try:
            row_tau_min.setStretch(0, 1)
        except Exception:
            pass

        row_tau_max = QtWidgets.QHBoxLayout()
        row_tau_max.setContentsMargins(0, 0, 0, 0)
        row_tau_max.setSpacing(0)
        row_tau_max.addWidget(self.spin_tau_max)
        try:
            row_tau_max.setStretch(0, 1)
        except Exception:
            pass

        row_tau_bins = QtWidgets.QHBoxLayout()
        row_tau_bins.setContentsMargins(0, 0, 0, 0)
        row_tau_bins.setSpacing(0)
        row_tau_bins.addWidget(self.spin_tau_bins)
        try:
            row_tau_bins.setStretch(0, 1)
        except Exception:
            pass

        grid_tau.addLayout(row_tau_min)
        grid_tau.addLayout(row_tau_max)
        grid_tau.addLayout(row_tau_bins)
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
        # First take MaxEnt-specific settings from our JSON file; fall back
        # to global chisurf.core.settings.fret when available.
        try:
            fret_cfg = (self._settings.get("fret", {}) or {})
            if "tau0" in fret_cfg:
                self.spin_tau0.setValue(float(fret_cfg["tau0"]))
            if "R0" in fret_cfg:
                self.spin_R0.setValue(float(fret_cfg["R0"]))
            if "period_ns" in fret_cfg:
                self.spin_period.setValue(float(fret_cfg["period_ns"]))
            use_periodic = bool(fret_cfg.get("use_periodic", False))
        except Exception:
            fret_cfg = {}
            use_periodic = False
        try:
            cfg = getattr(chisurf.core.settings, "fret", {}) or {}
            if "tau0" not in fret_cfg:
                self.spin_tau0.setValue(float(cfg.get("tau0", self.spin_tau0.value())))
            if "R0" not in fret_cfg:
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
        try:
            self.chk_use_periodic.setChecked(bool(use_periodic))
        except Exception:
            self.chk_use_periodic.setChecked(False)
        # React to changes by updating which widgets are visible.
        try:
            self.chk_use_periodic.toggled.connect(lambda _checked: self._update_mode_ui())
        except Exception:
            pass
        mem_layout.addRow(self.chk_use_periodic)

        # FRET distance range (R/R0 grid fractions).
        r_min_default = 0.1
        r_max_default = 3.0
        r_res_default = 96
        try:
            if "r_min_frac" in fret_cfg:
                r_min_default = float(fret_cfg["r_min_frac"])
            if "r_max_frac" in fret_cfg:
                r_max_default = float(fret_cfg["r_max_frac"])
            if "r_bins" in fret_cfg:
                r_res_default = int(fret_cfg["r_bins"])
        except Exception:
            pass
        try:
            cfg = getattr(chisurf.core.settings, "fret", {}) or {}
            if "r_bins" not in fret_cfg:
                r_res_default = int(cfg.get("rda_resolution", r_res_default))
            if "r_min_frac" not in fret_cfg and "r_max_frac" not in fret_cfg:
                r0_val = float(self.spin_R0.value())
                rda_min = cfg.get("rda_min", None)
                rda_max = cfg.get("rda_max", None)
                if rda_min is not None and rda_max is not None and np.isfinite(r0_val) and r0_val > 0.0:
                    rda_min = float(rda_min)
                    rda_max = float(rda_max)
                    if np.isfinite(rda_min) and np.isfinite(rda_max) and rda_min > 0.0 and rda_max > rda_min:
                        r_min_default = rda_min / r0_val
                        r_max_default = rda_max / r0_val
        except Exception:
            pass
        if not np.isfinite(r_min_default) or r_min_default <= 0.0:
            r_min_default = 0.1
        if not np.isfinite(r_max_default) or r_max_default <= r_min_default:
            r_max_default = max(3.0, r_min_default + 1e-3)
        if r_res_default < 2:
            r_res_default = 2

        grid_r = QtWidgets.QVBoxLayout()
        grid_r.setContentsMargins(0, 0, 0, 0)
        grid_r.setSpacing(0)

        row_r_min = QtWidgets.QHBoxLayout()
        row_r_min.setContentsMargins(0, 0, 0, 0)
        row_r_min.setSpacing(0)
        self.spin_R_min = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_R_min.setDecimals(3)
        self.spin_R_min.setRange(0.001, 10.0)
        self.spin_R_min.setSingleStep(0.05)
        self.spin_R_min.setValue(r_min_default)
        try:
            self.spin_R_min.setToolTip("Lower bound as fraction of R0. The distance grid is built as R = linspace(R_min_frac*R0, R_max_frac*R0, N).")
        except Exception:
            pass
        row_r_min.addWidget(self.spin_R_min)
        try:
            row_r_min.setStretch(0, 1)
        except Exception:
            pass

        row_r_max = QtWidgets.QHBoxLayout()
        row_r_max.setContentsMargins(0, 0, 0, 0)
        row_r_max.setSpacing(0)
        self.spin_R_max = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_R_max.setDecimals(3)
        self.spin_R_max.setRange(0.001, 10.0)
        self.spin_R_max.setSingleStep(0.05)
        self.spin_R_max.setValue(r_max_default)
        try:
            self.spin_R_max.setToolTip("Upper bound as fraction of R0. The distance grid is built as R = linspace(R_min_frac*R0, R_max_frac*R0, N).")
        except Exception:
            pass
        row_r_max.addWidget(self.spin_R_max)
        try:
            row_r_max.setStretch(0, 1)
        except Exception:
            pass

        row_r_pts = QtWidgets.QHBoxLayout()
        row_r_pts.setContentsMargins(0, 0, 0, 0)
        row_r_pts.setSpacing(0)
        self.spin_R_points = QtWidgets.QSpinBox(mem_group)
        self.spin_R_points.setRange(2, 10000)
        self.spin_R_points.setValue(r_res_default)
        try:
            self.spin_R_points.setToolTip("Number of points in the distance grid.")
        except Exception:
            pass
        row_r_pts.addWidget(self.spin_R_points)
        try:
            row_r_pts.setStretch(0, 1)
        except Exception:
            pass

        grid_r.addLayout(row_r_min)
        grid_r.addLayout(row_r_max)
        grid_r.addLayout(row_r_pts)
        mem_layout.addRow("R/R0 range", grid_r)
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
        try:
            self._donor_btn_style_normal = self.btn_load_donor.styleSheet()
            self._donor_btn_fit_style_normal = self.btn_load_donor_fit.styleSheet()
        except Exception:
            self._donor_btn_style_normal = None
            self._donor_btn_fit_style_normal = None
        donor_btns = QtWidgets.QHBoxLayout()
        donor_btns.setContentsMargins(0, 0, 0, 0)
        donor_btns.setSpacing(0)
        donor_btns.addWidget(self.btn_load_donor)
        donor_btns.addWidget(self.btn_load_donor_fit)
        try:
            donor_btns.setStretch(0, 1)
            donor_btns.setStretch(1, 1)
        except Exception:
            pass
        self.label_donor_info = QtWidgets.QLabel("Donor spectrum: tau0 only", mem_group)
        try:
            self._donor_label_style_normal = self.label_donor_info.styleSheet()
        except Exception:
            self._donor_label_style_normal = None
        mem_layout.addRow(donor_btns)
        mem_layout.addRow(self.label_donor_info)

        self.spin_x_donly = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_x_donly.setDecimals(3)
        self.spin_x_donly.setRange(0.0, 1.0)
        self.spin_x_donly.setSingleStep(0.05)
        self.spin_x_donly.setValue(0.0)
        self.chk_fix_x_donly = QtWidgets.QCheckBox("fix", mem_group)
        try:
            self.chk_fix_x_donly.setToolTip("Keep donor-only fraction fixed during nuisance fitting")
        except Exception:
            pass
        self._row_x_donly = QtWidgets.QWidget(mem_group)
        row_x_donly_layout = QtWidgets.QHBoxLayout(self._row_x_donly)
        row_x_donly_layout.setContentsMargins(0, 0, 0, 0)
        row_x_donly_layout.setSpacing(4)
        row_x_donly_layout.addWidget(self.spin_x_donly)
        row_x_donly_layout.addWidget(self.chk_fix_x_donly)
        try:
            row_x_donly_layout.setStretch(0, 1)
            row_x_donly_layout.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("donor-only fraction", self._row_x_donly)
        try:
            self._x_donly_label = mem_layout.labelForField(self._row_x_donly)
        except Exception:
            self._x_donly_label = None

        self.spin_start_frac = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_start_frac.setDecimals(2)
        self.spin_start_frac.setRange(0.1, 1.0)
        self.spin_start_frac.setSingleStep(0.05)
        self.spin_start_frac.setValue(0.9)
        mem_layout.addRow("start @ frac of peak", self.spin_start_frac)
        try:
            self._start_frac_label = mem_layout.labelForField(self.spin_start_frac)
        except Exception:
            self._start_frac_label = None

        self.spin_timeshift = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_timeshift.setDecimals(4)
        self.spin_timeshift.setRange(-100.0, 100.0)
        self.spin_timeshift.setSingleStep(0.01)
        self.spin_timeshift.setValue(0.0)
        # Timeshift is expressed in tttr_channeldefinition channels (samples), not ns.
        self.chk_fix_timeshift = QtWidgets.QCheckBox("fix", mem_group)
        try:
            self.chk_fix_timeshift.setToolTip("Keep timeshift fixed during nuisance fitting")
        except Exception:
            pass
        self._row_timeshift = QtWidgets.QWidget(mem_group)
        row_timeshift_layout = QtWidgets.QHBoxLayout(self._row_timeshift)
        row_timeshift_layout.setContentsMargins(0, 0, 0, 0)
        row_timeshift_layout.setSpacing(4)
        row_timeshift_layout.addWidget(self.spin_timeshift)
        row_timeshift_layout.addWidget(self.chk_fix_timeshift)
        try:
            row_timeshift_layout.setStretch(0, 1)
            row_timeshift_layout.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("timeshift [ch]", self._row_timeshift)

        self.spin_background = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_background.setDecimals(1)
        self.spin_background.setRange(0.0, 1e9)
        self.spin_background.setSingleStep(1.0)
        self.spin_background.setValue(0.0)
        self.chk_fix_background = QtWidgets.QCheckBox("fix", mem_group)
        try:
            self.chk_fix_background.setToolTip("Keep background fixed during nuisance fitting")
        except Exception:
            pass
        self._row_background = QtWidgets.QWidget(mem_group)
        row_background_layout = QtWidgets.QHBoxLayout(self._row_background)
        row_background_layout.setContentsMargins(0, 0, 0, 0)
        row_background_layout.setSpacing(4)
        row_background_layout.addWidget(self.spin_background)
        row_background_layout.addWidget(self.chk_fix_background)
        try:
            row_background_layout.setStretch(0, 1)
            row_background_layout.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("background [cts]", self._row_background)

        self.spin_irf_bg = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_irf_bg.setDecimals(3)
        self.spin_irf_bg.setRange(0.0, 1e9)
        self.spin_irf_bg.setSingleStep(1.0)
        self.spin_irf_bg.setValue(0.0)
        self.chk_fix_irf_bg = QtWidgets.QCheckBox("fix", mem_group)
        try:
            self.chk_fix_irf_bg.setToolTip("Keep IRF background fixed during nuisance fitting")
        except Exception:
            pass
        self._row_irf_bg = QtWidgets.QWidget(mem_group)
        row_irf_bg_layout = QtWidgets.QHBoxLayout(self._row_irf_bg)
        row_irf_bg_layout.setContentsMargins(0, 0, 0, 0)
        row_irf_bg_layout.setSpacing(4)
        row_irf_bg_layout.addWidget(self.spin_irf_bg)
        row_irf_bg_layout.addWidget(self.chk_fix_irf_bg)
        try:
            row_irf_bg_layout.setStretch(0, 1)
            row_irf_bg_layout.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("IRF background [cts]", self._row_irf_bg)

        self.spin_lamp_scatter = QtWidgets.QDoubleSpinBox(mem_group)
        self.spin_lamp_scatter.setDecimals(4)
        self.spin_lamp_scatter.setRange(0.0, 1e3)
        self.spin_lamp_scatter.setSingleStep(0.001)
        self.spin_lamp_scatter.setValue(0.0)
        self.chk_fix_lamp_scatter = QtWidgets.QCheckBox("fix", mem_group)
        try:
            self.chk_fix_lamp_scatter.setToolTip("Keep lamp scatter fixed during nuisance fitting")
        except Exception:
            pass
        self._row_lamp_scatter = QtWidgets.QWidget(mem_group)
        row_lamp_scatter_layout = QtWidgets.QHBoxLayout(self._row_lamp_scatter)
        row_lamp_scatter_layout.setContentsMargins(0, 0, 0, 0)
        row_lamp_scatter_layout.setSpacing(4)
        row_lamp_scatter_layout.addWidget(self.spin_lamp_scatter)
        row_lamp_scatter_layout.addWidget(self.chk_fix_lamp_scatter)
        try:
            row_lamp_scatter_layout.setStretch(0, 1)
            row_lamp_scatter_layout.setStretch(1, 0)
        except Exception:
            pass
        mem_layout.addRow("lamp scatter", self._row_lamp_scatter)

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

        sampling_group = QtWidgets.QGroupBox("Sampling", ctrl)
        sampling_layout = QtWidgets.QFormLayout(sampling_group)
        try:
            sampling_layout.setContentsMargins(4, 4, 4, 4)
            sampling_layout.setHorizontalSpacing(4)
            sampling_layout.setVerticalSpacing(2)
        except Exception:
            pass

        self.spin_sample_steps = QtWidgets.QSpinBox(sampling_group)
        self.spin_sample_steps.setRange(10, 1000000)
        self.spin_sample_steps.setSingleStep(100)
        self.spin_sample_steps.setValue(500)
        sampling_layout.addRow("Q-MCMC steps", self.spin_sample_steps)

        self.spin_sample_thin = QtWidgets.QSpinBox(sampling_group)
        self.spin_sample_thin.setRange(1, 1000)
        self.spin_sample_thin.setSingleStep(1)
        self.spin_sample_thin.setValue(5)
        sampling_layout.addRow("Q-MCMC thinning", self.spin_sample_thin)

        self.spin_sample_walkers = QtWidgets.QSpinBox(sampling_group)
        self.spin_sample_walkers.setRange(0, 1000000)
        self.spin_sample_walkers.setSingleStep(10)
        self.spin_sample_walkers.setValue(0)
        sampling_layout.addRow("Walkers (0=auto)", self.spin_sample_walkers)

        self.spin_sample_substeps = QtWidgets.QSpinBox(sampling_group)
        self.spin_sample_substeps.setRange(1, 1000000)
        self.spin_sample_substeps.setSingleStep(10)
        self.spin_sample_substeps.setValue(50)
        sampling_layout.addRow("Chunk size (substeps)", self.spin_sample_substeps)

        self.spin_sample_nprocs = QtWidgets.QSpinBox(sampling_group)
        self.spin_sample_nprocs.setRange(0, 64)
        self.spin_sample_nprocs.setSingleStep(1)
        self.spin_sample_nprocs.setValue(0)
        sampling_layout.addRow("CPUs (0=auto)", self.spin_sample_nprocs)

        # Toggle between vectorized (single-process, NumPy/BLAS parallel) and
        # multiprocessing-based sampling. On Windows the default is
        # vectorized, which avoids the high IPC overhead of spawn-based
        # multiprocessing.
        self.chk_sample_vectorized = QtWidgets.QCheckBox(sampling_group)
        try:
            is_win = sys.platform.startswith("win")
        except Exception:
            is_win = False
        self.chk_sample_vectorized.setChecked(bool(is_win))
        sampling_layout.addRow("Vectorized sampling", self.chk_sample_vectorized)

        self.btn_sample = QtWidgets.QToolButton(sampling_group)
        self.btn_sample.setText("Sample Q-MCMC")
        try:
            self.btn_sample.setStyleSheet(
                "QToolButton { background-color: #7b3fa7; color: white; }"
            )
        except Exception:
            pass
        self.btn_sample.clicked.connect(self._on_sample_clicked)
        sampling_layout.addRow(self.btn_sample)

        try:
            for w in self.findChildren(QtWidgets.QDoubleSpinBox):
                w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                w.setMinimumWidth(120)
            for w in self.findChildren(QtWidgets.QSpinBox):
                w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                w.setMinimumWidth(120)
        except Exception:
            pass

        ctrl_layout.addWidget(sampling_group)

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

        self.plot_widget = pg.GraphicsLayoutWidget(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.addWidget(ctrl)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        try:
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(2)
            splitter.setSizes([250, 750])
        except Exception:
            pass
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
        self._sample_band_lower = None
        self._sample_band_upper = None
        self._sample_band_fill = None
        self._sample_hist_item = None

        self.plot_lcurve = self.plot_widget.addPlot(row=3, col=0, title="L-curve (chi\u00b2 vs |p|)")
        self.plot_lcurve.setLabel("bottom", "chi\u00b2")
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
        self._update_donor_requirement_ui()
