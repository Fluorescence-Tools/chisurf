"""Qt/pyqtgraph GUI front-end for the MaxEnt TCSPC lifetime/FRET MEM plugin.

New-style plugin with DockArea-based layout and menu bar,
following the burst selection plugin pattern.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.plugins.fluorescence_decay.maxent_decay.core.settings import (
    get_settings_file,
    load_maxent_settings as maxent_load_settings,
)

from .gui_actions import _MaxentActionsMixin
from .gui_data import _MaxentDataMixin
from .gui_helpers import _MaxentHelpersMixin
from .gui_mode import _MaxentModeMixin
from .gui_priors import _MaxentPriorsMixin
from .gui_plotting import _MaxentPlottingMixin
from .gui_run import _MaxentRunMixin

try:
    from qtpy import QtWidgets, QtCore, QtGui
except Exception:
    QtWidgets = QtCore = QtGui = None


class HelpDialog(QtWidgets.QDialog):
    """Help dialog with description and CLI reference."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About MaxEnt MEM")
        self.resize(640, 520)
        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QTextEdit(self)
        text.setReadOnly(True)

        cli_text = ""
        try:
            from ..cli.cli import cli
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(cli, ["--help"])
            cli_text = "<pre>\n" + result.output + "</pre>"
        except Exception as exc:
            cli_text = f"<p>CLI help unavailable: {exc}</p>"

        text.setHtml(
            """
            <h2>MaxEnt TCSPC lifetime/FRET MEM</h2>
            <p>This plugin performs maximum entropy analysis of time-correlated single photon counting (TCSPC) data to recover fluorescence lifetime distributions.</p>

            <h3>How it works</h3>
            <ol>
              <li>Load decay and IRF data</li>
              <li>Configure analysis parameters (regularization, optimization method)</li>
              <li>Optionally load priors or donor spectra for FRET analysis</li>
              <li>Click <b>Run</b> to perform the MEM optimization</li>
              <li>Optionally run sampling or analyze the L-curve</li>
            </ol>

            <h3>Output</h3>
            <p>Results include the recovered lifetime distribution, goodness-of-fit statistics, and optional samples or prior distributions.</p>

            <hr>
            <h3>CLI Reference</h3>
            """
            + cli_text
        )
        layout.addWidget(text, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

try:
    import pyqtgraph as pg
except Exception:
    pg = None

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

SETTINGS_ORG = "chisurf"
SETTINGS_APP = "MaxentDecayWidget"


@persist_plugin_state("maxent_decay")
class MaxentDecayWidget(
    _MaxentActionsMixin,
    _MaxentDataMixin,
    _MaxentHelpersMixin,
    _MaxentModeMixin,
    _MaxentPriorsMixin,
    _MaxentRunMixin,
    _MaxentPlottingMixin,
    QtWidgets.QMainWindow,
):
    """GUI front-end for the MaxEnt lifetime/FRET MEM analysis.

    Uses a DockArea layout with the control panel and individual plot docks
    for the decay, residuals, distribution, and L-curve.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MaxEnt MEM")
        self.resize(1200, 800)

        self._irf_dataset = None
        self._prior_vec: Optional[np.ndarray] = None
        self._donly_vec: Optional[np.ndarray] = None
        self._dist_prior_vec: Optional[np.ndarray] = None
        self._last_result = None
        self._sample_stats = None
        self._t_axis = None
        self._fit_range = None
        self._mode_fret = False

        self._donor_style_missing = (
            "QToolButton { color: #ff6b6b; font-weight: bold; }"
        )
        self._donor_label_style_missing = (
            "QLabel { color: #ff6b6b; font-weight: bold; }"
        )
        self._donor_btn_style_normal = None
        self._donor_btn_fit_style_normal = None
        self._donor_label_style_normal = None

        self._sampling_thread = None
        self._sampling_worker = None

        try:
            self._settings = maxent_load_settings()
        except Exception:
            self._settings = {}

        self._create_plot_widgets()
        self._init_ui()

    # ------------------------------------------------------------------ #
    #  Plot widget creation (called early, like burst selector pattern)   #
    # ------------------------------------------------------------------ #

    def _create_plot_widgets(self) -> None:
        self.plot_decay = pg.PlotWidget(title="Decay / fit / IRF")
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

        self.plot_wres = pg.PlotWidget(title="Weighted residuals")
        self.plot_wres.setLabel("bottom", "time", units="ns")
        self.plot_wres.setLabel("left", "wres")

        self.plot_dist = pg.PlotWidget(title="Lifetime / Distance distribution")
        self.plot_dist.setLabel("bottom", "lifetime", units="ns")
        self.plot_dist.setLabel("left", "probability")
        self._sample_band_lower = None
        self._sample_band_upper = None
        self._sample_band_fill = None
        self._sample_hist_item = None

        self.plot_lcurve = pg.PlotWidget(title="L-curve (chi\u00b2 vs |p|)")
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
            [],
            [],
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

    # ------------------------------------------------------------------ #
    #  UI construction: DockArea + docks (burst selector pattern)         #
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._setup_toolbar()
        self.dock_area = DockArea(central)
        self._build_docks()
        layout.addWidget(self.dock_area, 1)
        self._setup_menu()
        self._restore_geometry()
        self._restore_dock_layout()
        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        self._update_mode_ui()
        self._update_donor_requirement_ui()

    def _build_docks(self) -> None:
        ctrl = self._build_control_panel()
        self.dock_area.addTab(ctrl, "Settings", close_mode="hide")
        self.dock_area.addTab(self.plot_decay, "Decay / fit / IRF", close_mode="hide")
        self.dock_area.addTab(self.plot_wres, "Weighted residuals", close_mode="hide")
        self.dock_area.addTab(self.plot_dist, "Distribution", close_mode="hide")
        self.dock_area.addTab(self.plot_lcurve, "L-curve", close_mode="hide")

    # ------------------------------------------------------------------ #
    #  Control panel (Data/IRF, MEM settings, Sampling, buttons)          #
    # ------------------------------------------------------------------ #

    def _build_control_panel(self) -> QtWidgets.QWidget:
        ctrl = QtWidgets.QWidget()
        try:
            ctrl.setMinimumWidth(280)
            ctrl.setMaximumWidth(400)
        except Exception:
            pass

        layout = QtWidgets.QVBoxLayout(ctrl)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- Data / IRF group ---- #
        data_group = QtWidgets.QGroupBox("Data / IRF")
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(4, 4, 4, 4)
        data_layout.setSpacing(2)

        self.label_data_source = QtWidgets.QLabel("No data loaded")
        self.label_irf_source = QtWidgets.QLabel("IRF: model.convolve.irf (default)")
        data_layout.addWidget(self.label_data_source)
        data_layout.addWidget(self.label_irf_source)
        layout.addWidget(data_group)

        # ---- MEM settings group ---- #
        mem_group = QtWidgets.QGroupBox("MEM settings")
        mem_layout = QtWidgets.QFormLayout(mem_group)
        mem_layout.setContentsMargins(4, 4, 4, 4)
        mem_layout.setSpacing(2)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Lifetime")
        self.combo_mode.addItem("FRET")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mem_layout.addRow("Mode", self.combo_mode)

        self.spin_nu = QtWidgets.QDoubleSpinBox()
        self.spin_nu.setDecimals(6)
        self.spin_nu.setRange(1e-8, 1.0)
        self.spin_nu.setSingleStep(1e-3)
        self.spin_nu.setValue(1e-3)
        mem_layout.addRow("nu (reg)", self.spin_nu)

        self.spin_lcurve_dec_left = QtWidgets.QDoubleSpinBox()
        self.spin_lcurve_dec_left.setDecimals(2)
        self.spin_lcurve_dec_left.setRange(0.0, 6.0)
        self.spin_lcurve_dec_left.setSingleStep(0.5)
        lc_left = 2.0
        try:
            lc_left = float(
                ((self._settings.get("lcurve_span_decades", {}) or {}).get("left", 2.0))
            )
        except Exception:
            pass
        self.spin_lcurve_dec_left.setValue(lc_left)
        self.spin_lcurve_dec_left.setToolTip(
            "Decades below the current nu used when scanning the L-curve."
        )

        self.spin_lcurve_dec_right = QtWidgets.QDoubleSpinBox()
        self.spin_lcurve_dec_right.setDecimals(2)
        self.spin_lcurve_dec_right.setRange(0.0, 6.0)
        self.spin_lcurve_dec_right.setSingleStep(0.5)
        lc_right = 2.0
        try:
            lc_right = float(
                ((self._settings.get("lcurve_span_decades", {}) or {}).get("right", 2.0))
            )
        except Exception:
            pass
        self.spin_lcurve_dec_right.setValue(lc_right)
        self.spin_lcurve_dec_right.setToolTip(
            "Decades above the current nu used when scanning the L-curve."
        )

        row_lc = QtWidgets.QHBoxLayout()
        row_lc.setContentsMargins(0, 0, 0, 0)
        row_lc.setSpacing(4)
        row_lc.addWidget(self.spin_lcurve_dec_left)
        row_lc.addWidget(self.spin_lcurve_dec_right)
        mem_layout.addRow("L-curve span [dec]", row_lc)

        self.spin_tau_min = QtWidgets.QDoubleSpinBox()
        self.spin_tau_min.setDecimals(3)
        self.spin_tau_min.setRange(1e-3, 1e3)
        tau_min = 0.01
        try:
            tau_min = float(
                ((self._settings.get("tau_grid", {}) or {}).get("min", 0.01))
            )
        except Exception:
            pass
        self.spin_tau_min.setValue(tau_min)

        self.spin_tau_max = QtWidgets.QDoubleSpinBox()
        self.spin_tau_max.setDecimals(3)
        self.spin_tau_max.setRange(1e-3, 1e3)
        tau_max = 6.0
        try:
            tau_max = float(
                ((self._settings.get("tau_grid", {}) or {}).get("max", 6.0))
            )
        except Exception:
            pass
        self.spin_tau_max.setValue(tau_max)

        self.spin_tau_bins = QtWidgets.QSpinBox()
        self.spin_tau_bins.setRange(2, 10000)
        tau_bins = 192
        try:
            tg = self._settings.get("tau_grid", {}) or {}
            if "bins" in tg:
                tau_bins = int(tg.get("bins", 192))
            else:
                step_val = float(tg.get("step", 0.02))
                if step_val > 0.0 and tau_max > tau_min:
                    tau_bins = int(np.floor((tau_max - tau_min) / step_val)) + 1
        except Exception:
            pass
        tau_bins = max(2, tau_bins)
        self.spin_tau_bins.setValue(tau_bins)

        grid_tau = QtWidgets.QVBoxLayout()
        grid_tau.setContentsMargins(0, 0, 0, 0)
        grid_tau.setSpacing(2)
        grid_tau.addWidget(self.spin_tau_min)
        grid_tau.addWidget(self.spin_tau_max)
        grid_tau.addWidget(self.spin_tau_bins)
        mem_layout.addRow("tau grid [ns]", grid_tau)
        self._tau_grid_layout = grid_tau
        try:
            self._tau_grid_label = mem_layout.labelForField(grid_tau)
        except Exception:
            self._tau_grid_label = None

        self.spin_tau0 = QtWidgets.QDoubleSpinBox()
        self.spin_tau0.setDecimals(3)
        self.spin_tau0.setRange(0.01, 100.0)
        self.spin_tau0.setValue(4.1)

        self.spin_R0 = QtWidgets.QDoubleSpinBox()
        self.spin_R0.setRange(10.0, 100.0)
        self.spin_R0.setSingleStep(1.0)
        self.spin_R0.setValue(50.0)

        self.spin_period = QtWidgets.QDoubleSpinBox()
        self.spin_period.setRange(0.1, 1000.0)
        self.spin_period.setSingleStep(1.0)
        self.spin_period.setValue(10.0)

        fret_cfg = {}
        use_periodic = False
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
            pass

        try:
            import chisurf
            cfg = getattr(chisurf.core.settings, "fret", {}) or {}
            if "tau0" not in fret_cfg:
                self.spin_tau0.setValue(float(cfg.get("tau0", self.spin_tau0.value())))
            if "R0" not in fret_cfg:
                self.spin_R0.setValue(float(cfg.get("forster_radius", self.spin_R0.value())))
        except Exception:
            pass

        mem_layout.addRow("tau0 [ns]", self.spin_tau0)
        try:
            self._tau0_label = mem_layout.labelForField(self.spin_tau0)
        except Exception:
            self._tau0_label = None
        mem_layout.addRow("R0 [\u00c5]", self.spin_R0)
        try:
            self._R0_label = mem_layout.labelForField(self.spin_R0)
        except Exception:
            self._R0_label = None
        mem_layout.addRow("Period [ns]", self.spin_period)
        try:
            self._period_label = mem_layout.labelForField(self.spin_period)
        except Exception:
            self._period_label = None

        self.chk_use_periodic = QtWidgets.QCheckBox("Periodic convolution")
        self.chk_use_periodic.setChecked(use_periodic)
        self.chk_use_periodic.toggled.connect(lambda _: self._update_mode_ui())
        mem_layout.addRow(self.chk_use_periodic)

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

        self.spin_R_min = QtWidgets.QDoubleSpinBox()
        self.spin_R_min.setDecimals(3)
        self.spin_R_min.setRange(0.001, 10.0)
        self.spin_R_min.setSingleStep(0.05)
        self.spin_R_min.setValue(r_min_default)

        self.spin_R_max = QtWidgets.QDoubleSpinBox()
        self.spin_R_max.setDecimals(3)
        self.spin_R_max.setRange(0.001, 10.0)
        self.spin_R_max.setSingleStep(0.05)
        self.spin_R_max.setValue(r_max_default)

        self.spin_R_points = QtWidgets.QSpinBox()
        self.spin_R_points.setRange(2, 10000)
        self.spin_R_points.setValue(r_res_default)

        grid_r = QtWidgets.QVBoxLayout()
        grid_r.setContentsMargins(0, 0, 0, 0)
        grid_r.setSpacing(2)
        grid_r.addWidget(self.spin_R_min)
        grid_r.addWidget(self.spin_R_max)
        grid_r.addWidget(self.spin_R_points)
        mem_layout.addRow("R/R0 range", grid_r)
        self._rda_grid_layout = grid_r
        try:
            self._rda_grid_label = mem_layout.labelForField(grid_r)
        except Exception:
            self._rda_grid_label = None

        self.label_donor_info = QtWidgets.QLabel("Donor spectrum: tau0 only")
        try:
            self._donor_label_style_normal = self.label_donor_info.styleSheet()
        except Exception:
            pass
        mem_layout.addRow(self.label_donor_info)

        self.spin_x_donly = QtWidgets.QDoubleSpinBox()
        self.spin_x_donly.setDecimals(3)
        self.spin_x_donly.setRange(0.0, 1.0)
        self.spin_x_donly.setSingleStep(0.05)
        self.spin_x_donly.setValue(0.0)
        self.chk_fix_x_donly = QtWidgets.QCheckBox("fix")
        self._row_x_donly = QtWidgets.QWidget()
        row_x_donly_layout = QtWidgets.QHBoxLayout(self._row_x_donly)
        row_x_donly_layout.setContentsMargins(0, 0, 0, 0)
        row_x_donly_layout.setSpacing(4)
        row_x_donly_layout.addWidget(self.spin_x_donly)
        row_x_donly_layout.addWidget(self.chk_fix_x_donly)
        mem_layout.addRow("donor-only fraction", self._row_x_donly)
        try:
            self._x_donly_label = mem_layout.labelForField(self._row_x_donly)
        except Exception:
            self._x_donly_label = None

        self.spin_start_frac = QtWidgets.QDoubleSpinBox()
        self.spin_start_frac.setDecimals(2)
        self.spin_start_frac.setRange(0.1, 1.0)
        self.spin_start_frac.setSingleStep(0.05)
        self.spin_start_frac.setValue(0.9)
        mem_layout.addRow("start @ frac of peak", self.spin_start_frac)
        try:
            self._start_frac_label = mem_layout.labelForField(self.spin_start_frac)
        except Exception:
            self._start_frac_label = None

        self.spin_timeshift = QtWidgets.QDoubleSpinBox()
        self.spin_timeshift.setDecimals(4)
        self.spin_timeshift.setRange(-100.0, 100.0)
        self.spin_timeshift.setSingleStep(0.01)
        self.spin_timeshift.setValue(0.0)
        self.chk_fix_timeshift = QtWidgets.QCheckBox("fix")
        self._row_timeshift = QtWidgets.QWidget()
        row_timeshift_layout = QtWidgets.QHBoxLayout(self._row_timeshift)
        row_timeshift_layout.setContentsMargins(0, 0, 0, 0)
        row_timeshift_layout.setSpacing(4)
        row_timeshift_layout.addWidget(self.spin_timeshift)
        row_timeshift_layout.addWidget(self.chk_fix_timeshift)
        mem_layout.addRow("timeshift [ch]", self._row_timeshift)

        self.spin_background = QtWidgets.QDoubleSpinBox()
        self.spin_background.setDecimals(1)
        self.spin_background.setRange(0.0, 1e9)
        self.spin_background.setSingleStep(1.0)
        self.spin_background.setValue(0.0)
        self.chk_fix_background = QtWidgets.QCheckBox("fix")
        self._row_background = QtWidgets.QWidget()
        row_background_layout = QtWidgets.QHBoxLayout(self._row_background)
        row_background_layout.setContentsMargins(0, 0, 0, 0)
        row_background_layout.setSpacing(4)
        row_background_layout.addWidget(self.spin_background)
        row_background_layout.addWidget(self.chk_fix_background)
        mem_layout.addRow("background [cts]", self._row_background)

        self.spin_irf_bg = QtWidgets.QDoubleSpinBox()
        self.spin_irf_bg.setDecimals(3)
        self.spin_irf_bg.setRange(0.0, 1e9)
        self.spin_irf_bg.setSingleStep(1.0)
        self.spin_irf_bg.setValue(0.0)
        self.chk_fix_irf_bg = QtWidgets.QCheckBox("fix")
        self._row_irf_bg = QtWidgets.QWidget()
        row_irf_bg_layout = QtWidgets.QHBoxLayout(self._row_irf_bg)
        row_irf_bg_layout.setContentsMargins(0, 0, 0, 0)
        row_irf_bg_layout.setSpacing(4)
        row_irf_bg_layout.addWidget(self.spin_irf_bg)
        row_irf_bg_layout.addWidget(self.chk_fix_irf_bg)
        mem_layout.addRow("IRF background [cts]", self._row_irf_bg)

        self.spin_lamp_scatter = QtWidgets.QDoubleSpinBox()
        self.spin_lamp_scatter.setDecimals(4)
        self.spin_lamp_scatter.setRange(0.0, 1e3)
        self.spin_lamp_scatter.setSingleStep(0.001)
        self.spin_lamp_scatter.setValue(0.0)
        self.chk_fix_lamp_scatter = QtWidgets.QCheckBox("fix")
        self._row_lamp_scatter = QtWidgets.QWidget()
        row_lamp_scatter_layout = QtWidgets.QHBoxLayout(self._row_lamp_scatter)
        row_lamp_scatter_layout.setContentsMargins(0, 0, 0, 0)
        row_lamp_scatter_layout.setSpacing(4)
        row_lamp_scatter_layout.addWidget(self.spin_lamp_scatter)
        row_lamp_scatter_layout.addWidget(self.chk_fix_lamp_scatter)
        mem_layout.addRow("lamp scatter", self._row_lamp_scatter)

        self.chk_fit_nuisance = QtWidgets.QCheckBox("Fit nuisance (ts/bg/IRF BG)")
        self.chk_fit_nuisance.setChecked(False)
        mem_layout.addRow(self.chk_fit_nuisance)

        self.label_prior_info = QtWidgets.QLabel("Prior: default 1/tau")
        self.label_dist_prior_info = QtWidgets.QLabel("Distance prior: default flat")
        mem_layout.addRow(self.label_prior_info)
        mem_layout.addRow(self.label_dist_prior_info)

        layout.addWidget(mem_group)

        # ---- Sampling group ---- #
        sampling_group = QtWidgets.QGroupBox("Sampling")
        sampling_layout = QtWidgets.QFormLayout(sampling_group)
        sampling_layout.setContentsMargins(4, 4, 4, 4)
        sampling_layout.setSpacing(2)

        self.spin_sample_steps = QtWidgets.QSpinBox()
        self.spin_sample_steps.setRange(10, 1000000)
        self.spin_sample_steps.setSingleStep(100)
        self.spin_sample_steps.setValue(500)
        sampling_layout.addRow("Q-MCMC steps", self.spin_sample_steps)

        self.spin_sample_thin = QtWidgets.QSpinBox()
        self.spin_sample_thin.setRange(1, 1000)
        self.spin_sample_thin.setSingleStep(1)
        self.spin_sample_thin.setValue(5)
        sampling_layout.addRow("Q-MCMC thinning", self.spin_sample_thin)

        self.spin_sample_walkers = QtWidgets.QSpinBox()
        self.spin_sample_walkers.setRange(0, 1000000)
        self.spin_sample_walkers.setSingleStep(10)
        self.spin_sample_walkers.setValue(0)
        sampling_layout.addRow("Walkers (0=auto)", self.spin_sample_walkers)

        self.spin_sample_substeps = QtWidgets.QSpinBox()
        self.spin_sample_substeps.setRange(1, 1000000)
        self.spin_sample_substeps.setSingleStep(10)
        self.spin_sample_substeps.setValue(50)
        sampling_layout.addRow("Chunk size (substeps)", self.spin_sample_substeps)

        self.spin_sample_nprocs = QtWidgets.QSpinBox()
        self.spin_sample_nprocs.setRange(0, 64)
        self.spin_sample_nprocs.setSingleStep(1)
        self.spin_sample_nprocs.setValue(0)
        sampling_layout.addRow("CPUs (0=auto)", self.spin_sample_nprocs)

        import sys as _sys
        self.chk_sample_vectorized = QtWidgets.QCheckBox()
        is_win = _sys.platform.startswith("win")
        self.chk_sample_vectorized.setChecked(is_win)
        sampling_layout.addRow("Vectorized sampling", self.chk_sample_vectorized)

        layout.addWidget(sampling_group)

        layout.addStretch(1)

        try:
            for w in ctrl.findChildren(QtWidgets.QDoubleSpinBox):
                w.setMinimumWidth(100)
            for w in ctrl.findChildren(QtWidgets.QSpinBox):
                w.setMinimumWidth(100)
        except Exception:
            pass

        return ctrl

    # ------------------------------------------------------------------ #
    #  Toolbar (all action buttons, following burst selector pattern)     #
    # ------------------------------------------------------------------ #

    def _setup_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("maxentMainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setContentsMargins(4, 2, 4, 2)
        if toolbar.layout() is not None:
            toolbar.layout().setSpacing(6)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet("""
            QToolBar#maxentMainToolbar {
                background-color: transparent;
                border: none;
                padding: 3px 4px;
                spacing: 6px;
            }
            QToolBar#maxentMainToolbar::separator {
                width: 8px;
            }
            QToolBar#maxentMainToolbar QToolButton {
                background-color: #2a2a4a;
                border: 1px solid #5a5a8a;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 0px;
                font-weight: bold;
                font-size: 12px;
            }
            QToolBar#maxentMainToolbar QToolButton:hover {
                background-color: #3a3a6a;
                border-color: #8a8aba;
            }
            QToolBar#maxentMainToolbar QToolButton:pressed {
                background-color: #4a4a8a;
            }
            QToolBar#maxentMainToolbar QToolButton:disabled {
                color: #555;
            }
            QToolBar#maxentMainToolbar #maxentBtnRun {
                color: #4ade80;
            }
            QToolBar#maxentMainToolbar #maxentBtnSave {
                color: #93c5fd;
            }
            QToolBar#maxentMainToolbar #maxentBtnLcurve {
                color: #60a5fa;
            }
            QToolBar#maxentMainToolbar #maxentBtnSample {
                color: #c084fc;
            }
            QToolBar#maxentMainToolbar #maxentBtnRefresh {
                color: #fbbf24;
            }
            QToolBar#maxentMainToolbar #maxentBtnIrf {
                color: #fb923c;
            }
            QToolBar#maxentMainToolbar #maxentBtnJson {
                color: #a78bfa;
            }
            QToolBar#maxentMainToolbar #maxentBtnPrior {
                color: #67e8f9;
            }
            QToolBar#maxentMainToolbar #maxentBtnDonor {
                color: #f472b6;
            }
            QToolBar#maxentMainToolbar #maxentBtnDonorFit {
                color: #f472b6;
            }
            QToolBar#maxentMainToolbar #maxentBtnHelp {
                color: #9ca3af;
            }
        """)

        self.btn_run = QtWidgets.QToolButton()
        self.btn_run.setText("\U0001f3af Run")
        self.btn_run.setObjectName("maxentBtnRun")
        self.btn_run.setAutoRaise(True)
        self.btn_run.setToolTip("Run MaxEnt MEM optimization with current settings")
        self.btn_run.clicked.connect(self._on_run_clicked)
        toolbar.addWidget(self.btn_run)

        self.btn_save = QtWidgets.QToolButton()
        self.btn_save.setText("\U0001f4be Save")
        self.btn_save.setObjectName("maxentBtnSave")
        self.btn_save.setAutoRaise(True)
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Save MEM result to disk")
        self.btn_save.clicked.connect(self._on_save_clicked)
        toolbar.addWidget(self.btn_save)

        toolbar.addSeparator()

        self.btn_lcurve = QtWidgets.QToolButton()
        self.btn_lcurve.setText("\U0001f4c8 L-curve")
        self.btn_lcurve.setObjectName("maxentBtnLcurve")
        self.btn_lcurve.setAutoRaise(True)
        self.btn_lcurve.setToolTip("Scan regularization parameter and detect optimal nu via L-curve corner")
        self.btn_lcurve.clicked.connect(self._on_lcurve_clicked)
        toolbar.addWidget(self.btn_lcurve)

        self.btn_sample = QtWidgets.QToolButton()
        self.btn_sample.setText("\U0001f9ea Sample")
        self.btn_sample.setObjectName("maxentBtnSample")
        self.btn_sample.setAutoRaise(True)
        self.btn_sample.setToolTip("Run Q-MCMC sampling on the current MEM result")
        self.btn_sample.clicked.connect(self._on_sample_clicked)
        toolbar.addWidget(self.btn_sample)

        toolbar.addSeparator()

        self.btn_refresh_data = QtWidgets.QToolButton()
        self.btn_refresh_data.setText("\U0001f504 Refresh")
        self.btn_refresh_data.setObjectName("maxentBtnRefresh")
        self.btn_refresh_data.setAutoRaise(True)
        self.btn_refresh_data.setToolTip("Read decay and IRF from the current ChiSurf fit")
        self.btn_refresh_data.clicked.connect(self._on_refresh_data)
        toolbar.addWidget(self.btn_refresh_data)

        self.btn_select_irf = QtWidgets.QToolButton()
        self.btn_select_irf.setText("\U0001f50d IRF")
        self.btn_select_irf.setObjectName("maxentBtnIrf")
        self.btn_select_irf.setAutoRaise(True)
        self.btn_select_irf.setToolTip("Select a different IRF dataset from the project")
        self.btn_select_irf.clicked.connect(self._on_select_irf_clicked)
        toolbar.addWidget(self.btn_select_irf)

        self.btn_edit_settings = QtWidgets.QToolButton()
        self.btn_edit_settings.setText("\u2699\ufe0f JSON")
        self.btn_edit_settings.setObjectName("maxentBtnJson")
        self.btn_edit_settings.setAutoRaise(True)
        try:
            self.btn_edit_settings.setToolTip(
                f"Edit MaxEnt JSON settings file:\n{get_settings_file()}"
            )
        except Exception:
            self.btn_edit_settings.setToolTip("Edit MaxEnt JSON settings file")
        self.btn_edit_settings.clicked.connect(self._on_edit_settings_clicked)
        toolbar.addWidget(self.btn_edit_settings)

        toolbar.addSeparator()

        self.btn_load_prior = QtWidgets.QToolButton()
        self.btn_load_prior.setText("\U0001f4e4 Prior")
        self.btn_load_prior.setObjectName("maxentBtnPrior")
        self.btn_load_prior.setAutoRaise(True)
        self.btn_load_prior.setToolTip("Load prior distribution (lifetime or distance depending on mode)")
        self.btn_load_prior.clicked.connect(self._on_load_prior_any_clicked)
        toolbar.addWidget(self.btn_load_prior)

        self.btn_load_donor = QtWidgets.QToolButton()
        self.btn_load_donor.setText("\U0001f48e Donor")
        self.btn_load_donor.setObjectName("maxentBtnDonor")
        self.btn_load_donor.setAutoRaise(True)
        self.btn_load_donor.setToolTip("Load donor emission spectrum (amp/tau pairs) for FRET analysis")
        self.btn_load_donor.clicked.connect(self._on_load_donor_clicked)
        self.btn_load_donor_fit = QtWidgets.QToolButton()
        self.btn_load_donor_fit.setText("\U0001f3e0 Fit")
        self.btn_load_donor_fit.setObjectName("maxentBtnDonorFit")
        self.btn_load_donor_fit.setAutoRaise(True)
        self.btn_load_donor_fit.setToolTip("Load donor spectrum from current ChiSurf fit")
        self.btn_load_donor_fit.clicked.connect(self._on_load_donor_from_fit_clicked)
        try:
            self._donor_btn_style_normal = self.btn_load_donor.styleSheet()
            self._donor_btn_fit_style_normal = self.btn_load_donor_fit.styleSheet()
        except Exception:
            pass
        toolbar.addWidget(self.btn_load_donor)
        toolbar.addWidget(self.btn_load_donor_fit)

        toolbar.addSeparator()

        self.btn_help = QtWidgets.QToolButton()
        self.btn_help.setText("ℹ️ Help")
        self.btn_help.setObjectName("maxentBtnHelp")
        self.btn_help.setAutoRaise(True)
        self.btn_help.setToolTip("Show help and CLI reference")
        self.btn_help.clicked.connect(self._on_help_clicked)
        toolbar.addWidget(self.btn_help)

    # ------------------------------------------------------------------ #
    #  Menu bar  (burst selector pattern)                                 #
    # ------------------------------------------------------------------ #

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_action = QtGui.QAction("Save result", self)
        save_action.triggered.connect(self._on_save_clicked)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        settings_menu = menubar.addMenu("Settings")
        edit_action = QtGui.QAction("Edit JSON settings", self)
        edit_action.triggered.connect(self._on_edit_settings_clicked)
        settings_menu.addAction(edit_action)

        help_menu = menubar.addMenu("Help")
        readme_action = QtGui.QAction("README", self)
        readme_action.triggered.connect(self._on_help_clicked)
        help_menu.addAction(readme_action)

    def _on_help_clicked(self) -> None:
        """Show the help dialog."""
        dialog = HelpDialog(self)
        dialog.exec_()

        # ------------------------------------------------------------------ #
        #  Window / dock state persistence (burst selector pattern)           #
        # ------------------------------------------------------------------ #

    def _save_geometry(self) -> None:
        try:
            settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
            settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass

    def _restore_geometry(self) -> None:
        try:
            settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
            geom = settings.value("geometry")
            if geom is not None:
                self.restoreGeometry(geom)
        except Exception:
            pass

    def _save_dock_layout(self) -> None:
        try:
            settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
            layout = self.dock_area.get_layout_state()
            settings.setValue("dock_layout", layout)
        except Exception:
            pass

    def _restore_dock_layout(self) -> None:
        try:
            settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
            layout = settings.value("dock_layout")
            if layout is not None:
                self.dock_area.set_layout_state(layout)
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_geometry()
        self._save_dock_layout()
        super().closeEvent(event)


__all__ = ["MaxentDecayWidget"]
