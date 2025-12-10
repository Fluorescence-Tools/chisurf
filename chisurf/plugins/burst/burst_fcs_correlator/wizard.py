"""Wizard for burst-wise FCS correlation.

This wizard reuses the standard :class:`ChisurfFCSWizard` pages
to select detector setups, files and FCS correlator settings, but
changes the *Finish* step:

- Input: Burst-ID ``.bst`` files listed on the Files page.
- For each ``.bst`` file, the underlying TTTR container is resolved
  and each (start, stop) photon index range is treated as a burst.
- For each burst, a single FCS correlation curve is computed for the
  **currently selected correlator channels and settings** on the
  Correlator page (typically set via the FCS preset combobox).
- Each correlation curve is then fitted with :func:`chisurf.models.fcs.maxent.fcs_maxent`
  to obtain a diffusion-time distribution. Two summary diffusion
  times are stored per burst: the probability-weighted mean and the
  peak (mode) of the distribution.
- Results are written to ``td4`` folders next to the underlying TTTR
  files in a "b*4-like" zero-interleaved tabular format.
"""

import json
import pathlib
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tttrlib
import pyqtgraph as pg

from chisurf.gui import QtWidgets, QtCore
from chisurf.models.fcs.maxent import fcs_maxent
from chisurf.gui.widgets.wizard.tttr_correlator import WizardTTTRCorrelator
from chisurf.gui.widgets.wizard.tttr_channel_definition import load_detector_setups
from chisurf.settings.path_utils import get_path as _get_settings_path
from chisurf import settings as _cs_settings
from .file_list import BurstFileListWidget
from .helpers import (
    parse_bst_file,
    parse_bur_file,
    open_tttr,
    parse_channel_list,
    correlate_single_burst,
    fit_simple_diffusion,
)


class BurstWiseFCSWizard(QtWidgets.QDialog):
    """Burst-wise FCS correlator wizard.

    The regular FCS wizard flow is preserved (detector setup, file
    selection, optional photon filter, correlator page). When the user
    clicks *Finish*, this subclass performs a burst-wise FCS analysis
    on the selected ``.bst`` files instead of running the standard
    merger logic.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Burst-wise FCS Correlator")
        try:
            self.resize(800, 600)
        except Exception:
            pass
        self._detector_setups: Dict[str, Any] = {}
        self._pair_configs: List[Dict[str, Any]] = []
        self._cached_curves: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self) -> None:

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1) Detector setup selector at the very top
        top_row = QtWidgets.QHBoxLayout()
        lbl_setup = QtWidgets.QLabel("Detector setup:", self)
        self.combo_setup = QtWidgets.QComboBox(self)
        top_row.addWidget(lbl_setup)
        top_row.addWidget(self.combo_setup, 1)
        layout.addLayout(top_row)

        # 2) Internal correlator widget (hidden) to manage FCS presets/parameters
        self.correlator = WizardTTTRCorrelator()
        try:
            self.correlator.save_chunks_to_disk = False
        except Exception:
            pass

        # 2a/2b) Main splitter: processing (left) and browser (right)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Left side: processing widgets (pairs + file list)
        left_widget = QtWidgets.QWidget(main_splitter)
        proc_col = QtWidgets.QVBoxLayout(left_widget)
        proc_col.setContentsMargins(0, 0, 0, 0)
        proc_col.setSpacing(4)

        lbl_pairs = QtWidgets.QLabel("FCS channel pairs:", left_widget)
        proc_col.addWidget(lbl_pairs)
        self.list_pairs = QtWidgets.QListWidget(left_widget)
        self.list_pairs.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        try:
            row_h_pairs = self.list_pairs.fontMetrics().height() + 6
            self.list_pairs.setMaximumHeight(row_h_pairs * 5 + 8)
        except Exception:
            pass
        proc_col.addWidget(self.list_pairs)

        pad_row = QtWidgets.QHBoxLayout()
        lbl_pad = QtWidgets.QLabel("Photon padding around bursts (+/- ms):", left_widget)
        self.spin_padding_ms = QtWidgets.QDoubleSpinBox(left_widget)
        self.spin_padding_ms.setRange(0.0, 1e6)
        self.spin_padding_ms.setDecimals(1)
        self.spin_padding_ms.setSingleStep(0.1)
        self.spin_padding_ms.setValue(100.0)
        pad_row.addWidget(lbl_pad)
        pad_row.addWidget(self.spin_padding_ms, 1)
        proc_col.addLayout(pad_row)

        group_corr = QtWidgets.QGroupBox("FCS correlator (binning)", left_widget)
        corr_layout = QtWidgets.QVBoxLayout(group_corr)
        corr_layout.setContentsMargins(4, 4, 4, 4)
        corr_layout.setSpacing(2)

        bins_row = QtWidgets.QHBoxLayout()
        lbl_bins = QtWidgets.QLabel("FCS bins (B):", group_corr)
        self.spin_n_bins = QtWidgets.QSpinBox(group_corr)
        self.spin_n_bins.setRange(1, 65535)
        self.spin_n_bins.setValue(3)
        bins_row.addWidget(lbl_bins)
        bins_row.addWidget(self.spin_n_bins, 1)
        corr_layout.addLayout(bins_row)

        casc_row = QtWidgets.QHBoxLayout()
        lbl_casc = QtWidgets.QLabel("Number of cascades (n_casc):", group_corr)
        self.spin_n_casc = QtWidgets.QSpinBox(group_corr)
        self.spin_n_casc.setRange(1, 64)
        self.spin_n_casc.setValue(20)
        casc_row.addWidget(lbl_casc)
        casc_row.addWidget(self.spin_n_casc, 1)
        corr_layout.addLayout(casc_row)

        fine_row = QtWidgets.QHBoxLayout()
        self.check_fine = QtWidgets.QCheckBox("Fine correlation grid", group_corr)
        self.check_fine.setChecked(False)
        fine_row.addWidget(self.check_fine)
        fine_row.addStretch(1)
        corr_layout.addLayout(fine_row)

        proc_col.addWidget(group_corr)

        group_fit = QtWidgets.QGroupBox("Fitting / MaxEnt", left_widget)
        fit_layout = QtWidgets.QVBoxLayout(group_fit)
        fit_layout.setContentsMargins(4, 4, 4, 4)
        fit_layout.setSpacing(2)

        mode_row = QtWidgets.QHBoxLayout()
        self.radio_fit_simple = QtWidgets.QRadioButton("Simple diffusion", group_fit)
        self.radio_fit_maxent = QtWidgets.QRadioButton("MaxEnt", group_fit)
        try:
            self.radio_fit_simple.setChecked(True)
        except Exception:
            pass
        mode_row.addWidget(self.radio_fit_simple)
        mode_row.addWidget(self.radio_fit_maxent)
        mode_row.addStretch(1)
        fit_layout.addLayout(mode_row)

        maxent_grid = QtWidgets.QGridLayout()
        self.lbl_maxent_reg = QtWidgets.QLabel("MaxEnt reg (log10)", group_fit)
        self.spin_log10_reg = QtWidgets.QDoubleSpinBox(group_fit)
        self.spin_log10_reg.setDecimals(2)
        self.spin_log10_reg.setSingleStep(0.5)
        self.spin_log10_reg.setRange(-12.0, 6.0)
        self.spin_log10_reg.setValue(0.0)

        self.lbl_td_min = QtWidgets.QLabel("tau_D min (ms)", group_fit)
        self.spin_td_min = QtWidgets.QDoubleSpinBox(group_fit)
        self.spin_td_min.setDecimals(6)
        self.spin_td_min.setRange(1e-6, 1e6)
        self.spin_td_min.setSingleStep(1e-4)
        self.spin_td_min.setValue(1.0e-3)

        self.lbl_td_max = QtWidgets.QLabel("tau_D max (ms)", group_fit)
        self.spin_td_max = QtWidgets.QDoubleSpinBox(group_fit)
        self.spin_td_max.setDecimals(3)
        self.spin_td_max.setRange(1e-6, 1e9)
        self.spin_td_max.setSingleStep(0.1)
        self.spin_td_max.setValue(20.0)

        lbl_tmin = QtWidgets.QLabel("t_min (ms)", group_fit)
        self.spin_tmin_fit = QtWidgets.QDoubleSpinBox(group_fit)
        self.spin_tmin_fit.setDecimals(6)
        self.spin_tmin_fit.setRange(0.0, 1e9)
        self.spin_tmin_fit.setSingleStep(1e-4)
        self.spin_tmin_fit.setValue(0.002)

        lbl_tmax = QtWidgets.QLabel("t_max (ms)", group_fit)
        self.spin_tmax_fit = QtWidgets.QDoubleSpinBox(group_fit)
        self.spin_tmax_fit.setDecimals(3)
        self.spin_tmax_fit.setRange(0.0, 1e9)
        self.spin_tmax_fit.setSingleStep(0.1)
        self.spin_tmax_fit.setValue(0.0)

        maxent_grid.addWidget(self.lbl_maxent_reg, 0, 0)
        maxent_grid.addWidget(self.spin_log10_reg, 0, 1)
        maxent_grid.addWidget(self.lbl_td_min, 1, 0)
        maxent_grid.addWidget(self.spin_td_min, 1, 1)
        maxent_grid.addWidget(self.lbl_td_max, 2, 0)
        maxent_grid.addWidget(self.spin_td_max, 2, 1)
        maxent_grid.addWidget(lbl_tmin, 3, 0)
        maxent_grid.addWidget(self.spin_tmin_fit, 3, 1)
        maxent_grid.addWidget(lbl_tmax, 4, 0)
        maxent_grid.addWidget(self.spin_tmax_fit, 4, 1)
        fit_layout.addLayout(maxent_grid)

        proc_col.addWidget(group_fit)

        buttons_row = QtWidgets.QHBoxLayout()
        self.btn_show_pairs_json = QtWidgets.QToolButton(left_widget)
        self.btn_show_pairs_json.setText("FCS JSON")
        self.btn_load_settings = QtWidgets.QToolButton(left_widget)
        self.btn_load_settings.setText("Load JSON")
        self.btn_save_settings = QtWidgets.QToolButton(left_widget)
        self.btn_save_settings.setText("Save JSON")
        buttons_row.addWidget(self.btn_show_pairs_json)
        buttons_row.addWidget(self.btn_load_settings)
        buttons_row.addWidget(self.btn_save_settings)
        buttons_row.addStretch(1)
        proc_col.addLayout(buttons_row)

        lbl_files = QtWidgets.QLabel("Burst analysis folders or BUR/BID files:", left_widget)
        proc_col.addWidget(lbl_files)
        self.file_list = BurstFileListWidget(left_widget)
        proc_col.addWidget(self.file_list, 1)

        # Right side: integrated burst-wise FCS browser
        browser_box = QtWidgets.QGroupBox("Burst-wise FCS browser", main_splitter)
        browser_layout = QtWidgets.QVBoxLayout(browser_box)
        browser_layout.setContentsMargins(4, 4, 4, 4)
        browser_layout.setSpacing(4)

        browser_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, browser_box)

        # Top pane: filter + list of curves
        top_widget = QtWidgets.QWidget(browser_splitter)
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        self.line_burst_filter = QtWidgets.QLineEdit(top_widget)
        self.line_burst_filter.setPlaceholderText("Filter by file or pair name (e.g. 'GG')")
        top_layout.addWidget(self.line_burst_filter)
        self.list_browser = QtWidgets.QListWidget(top_widget)
        try:
            row_h_browser = self.list_browser.fontMetrics().height() + 6
            self.list_browser.setMaximumHeight(row_h_browser * 10 + 8)
        except Exception:
            pass
        top_layout.addWidget(self.list_browser)

        # Bottom pane: correlation + distribution plots with their own splitter
        bottom_widget = QtWidgets.QWidget(browser_splitter)
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, bottom_widget)
        self.plot_corr_view = pg.PlotWidget(plots_splitter)
        self.plot_corr_view.setLogMode(True, False)
        self.plot_corr_view.setLabel("bottom", "Correlation time, t_c (ms)")
        self.plot_corr_view.setLabel("left", "Correlation amplitude, G")
        self.curve_corr_data_view = self.plot_corr_view.plot(pen="w", symbol="o", symbolSize=4)
        self.curve_corr_fit_view = self.plot_corr_view.plot(pen="r")
        # Inset text for diffusion times (mean / fitted) in the correlation plot
        self.text_td_inset = pg.TextItem(color="y", anchor=(1, 1))
        try:
            self.text_td_inset.setText("")
        except Exception:
            pass
        try:
            self.plot_corr_view.addItem(self.text_td_inset, ignoreBounds=True)
        except Exception:
            pass

        self.plot_dist_view = pg.PlotWidget(plots_splitter)
        self.plot_dist_view.setLogMode(True, False)
        self.plot_dist_view.setLabel("bottom", "Diffusion time, tau_D (ms)")
        self.plot_dist_view.setLabel("left", "P(tau_D)")
        self.curve_dist_view = self.plot_dist_view.plot(pen="y")
        # Start with the diffusion-time distribution plot hidden; it is only
        # shown for entries that were fitted with MaxEnt.
        try:
            self.plot_dist_view.setVisible(False)
        except Exception:
            pass

        bottom_layout.addWidget(plots_splitter, 1)

        browser_splitter.addWidget(top_widget)
        browser_splitter.addWidget(bottom_widget)
        browser_splitter.setStretchFactor(0, 0)
        browser_splitter.setStretchFactor(1, 1)

        browser_layout.addWidget(browser_splitter, 1)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(browser_box)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        layout.addWidget(main_splitter, 1)

        btn_box = QtWidgets.QDialogButtonBox(self)
        self.btn_run = QtWidgets.QPushButton("Run burst-wise FCS", self)
        btn_close = btn_box.addButton(QtWidgets.QDialogButtonBox.Close)
        btn_box.addButton(self.btn_run, QtWidgets.QDialogButtonBox.ActionRole)
        layout.addWidget(btn_box)

        btn_close.clicked.connect(self.reject)
        self.btn_run.clicked.connect(self.onFinish)

        try:
            self.btn_show_pairs_json.clicked.connect(self._show_pair_config_dialog)
        except Exception:
            pass
        try:
            self.btn_load_settings.clicked.connect(self._on_load_settings_clicked)
            self.btn_save_settings.clicked.connect(self._on_save_settings_clicked)
        except Exception:
            pass

        # Update visibility of MaxEnt-specific controls when switching mode
        try:
            self.radio_fit_simple.toggled.connect(self._update_fit_mode_visibility)
            self.radio_fit_maxent.toggled.connect(self._update_fit_mode_visibility)
        except Exception:
            pass

        self._populate_detector_setups()
        self.line_burst_filter.textChanged.connect(self._update_browser_view)
        self.list_browser.currentItemChanged.connect(self._on_browser_selection_changed)

        # Ensure initial visibility matches the default fit mode
        try:
            self._update_fit_mode_visibility()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settings (save / load JSON)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_settings_json_paths() -> Tuple[pathlib.Path, pathlib.Path]:
        """Return (package_default_path, user_settings_path) for burst-FCS JSON.

        The default JSON lives next to other chisurf settings files and is copied
        into the user settings folder ("~/.chisurf") on first use.
        """

        # Package default: chisurf/settings/burst_fcs.settings.json
        pkg_root = pathlib.Path(__file__).resolve().parent.parent
        pkg_settings = pkg_root / "settings" / "burst_fcs.settings.json"

        # User settings root: ~/.chisurf
        user_root = _get_settings_path("settings")
        user_json = user_root / "burst_fcs.settings.json"
        return pkg_settings, user_json

    def _update_fit_mode_visibility(self) -> None:
        """Show/hide MaxEnt-only widgets based on selected fit mode.

        - When *MaxEnt* is selected, reg and tau_D min/max are visible/enabled.
        - When *Simple diffusion* is selected, they are hidden/disabled.
        The generic t_min/t_max window remains visible in both modes.
        """

        try:
            use_maxent = bool(self.radio_fit_maxent.isChecked())
        except Exception:
            use_maxent = False

        for w in (self.lbl_maxent_reg, self.spin_log10_reg,
                  self.lbl_td_min, self.spin_td_min,
                  self.lbl_td_max, self.spin_td_max):
            try:
                w.setVisible(use_maxent)
                w.setEnabled(use_maxent)
            except Exception:
                continue

    def _export_settings_dict(self) -> Dict[str, Any]:
        """Collect current UI state into a JSON-serializable dict."""

        data: Dict[str, Any] = {}

        # Correlator / burst padding
        try:
            data["padding_ms"] = float(self.spin_padding_ms.value())
        except Exception:
            data["padding_ms"] = 0.0

        try:
            data["n_bins"] = int(self.spin_n_bins.value())
        except Exception:
            data["n_bins"] = 3

        try:
            data["n_casc"] = int(self.spin_n_casc.value())
        except Exception:
            data["n_casc"] = 20

        try:
            data["make_fine"] = bool(self.check_fine.isChecked())
        except Exception:
            data["make_fine"] = False

        # Fitting / MaxEnt
        try:
            data["fit_mode"] = "maxent" if self.radio_fit_maxent.isChecked() else "simple"
        except Exception:
            data["fit_mode"] = "simple"

        try:
            data["log10_reg"] = float(self.spin_log10_reg.value())
        except Exception:
            data["log10_reg"] = 0.0

        try:
            data["td_min_ms"] = float(self.spin_td_min.value())
        except Exception:
            data["td_min_ms"] = 1.0e-3

        try:
            data["td_max_ms"] = float(self.spin_td_max.value())
        except Exception:
            data["td_max_ms"] = 20.0

        # Channel pairs: store logical names and checked state
        pairs: List[Dict[str, Any]] = []
        try:
            count = self.list_pairs.count()
        except Exception:
            count = 0
        for row in range(count):
            try:
                item = self.list_pairs.item(row)
            except Exception:
                item = None
            if item is None:
                continue
            pair_entry: Dict[str, Any] = {
                "label": str(item.text()),
                "checked": bool(item.checkState() == QtCore.Qt.Checked),
                "preset_index": int(item.data(QtCore.Qt.UserRole) or 0),
            }
            pairs.append(pair_entry)
        data["pairs"] = pairs

        return data

    def _apply_settings_dict(self, cfg: Dict[str, Any]) -> None:
        """Apply a previously stored settings dict to the current UI.

        Missing keys are ignored; invalid values are clamped by the widgets.
        """

        if not isinstance(cfg, dict):
            return

        try:
            if "padding_ms" in cfg:
                self.spin_padding_ms.setValue(float(cfg["padding_ms"]))
        except Exception:
            pass

        try:
            if "n_bins" in cfg:
                self.spin_n_bins.setValue(int(cfg["n_bins"]))
        except Exception:
            pass

        try:
            if "n_casc" in cfg:
                self.spin_n_casc.setValue(int(cfg["n_casc"]))
        except Exception:
            pass

        try:
            if "make_fine" in cfg:
                self.check_fine.setChecked(bool(cfg["make_fine"]))
        except Exception:
            pass

        try:
            mode = cfg.get("fit_mode", "simple")
            if mode == "maxent":
                self.radio_fit_maxent.setChecked(True)
            else:
                self.radio_fit_simple.setChecked(True)
        except Exception:
            pass

        try:
            if "log10_reg" in cfg:
                self.spin_log10_reg.setValue(float(cfg["log10_reg"]))
        except Exception:
            pass

        try:
            if "td_min_ms" in cfg:
                self.spin_td_min.setValue(float(cfg["td_min_ms"]))
        except Exception:
            pass

        try:
            if "td_max_ms" in cfg:
                self.spin_td_max.setValue(float(cfg["td_max_ms"]))
        except Exception:
            pass

        # Restore pair check states based on preset_index, if compatible
        try:
            pairs_cfg = cfg.get("pairs", [])
            if isinstance(pairs_cfg, list) and pairs_cfg:
                by_index: Dict[int, bool] = {}
                for entry in pairs_cfg:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        p_idx = int(entry.get("preset_index", 0))
                        checked = bool(entry.get("checked", True))
                    except Exception:
                        continue
                    by_index[p_idx] = checked

                for row in range(self.list_pairs.count()):
                    item = self.list_pairs.item(row)
                    idx_data = item.data(QtCore.Qt.UserRole)
                    try:
                        p_idx = int(idx_data)
                    except Exception:
                        continue
                    if p_idx in by_index:
                        item.setCheckState(QtCore.Qt.Checked if by_index[p_idx] else QtCore.Qt.Unchecked)
        except Exception:
            pass

    def _on_save_settings_clicked(self) -> None:
        """Save current burst-wise FCS settings to JSON in the user folder."""

        pkg_settings, user_json = self._get_settings_json_paths()
        try:
            user_json.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        data = self._export_settings_dict()
        try:
            with user_json.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4, sort_keys=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Burst-wise FCS settings",
                f"Could not save settings to '{user_json}':\n{e}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Burst-wise FCS settings",
            f"Settings saved to:\n{user_json}",
        )

    def _on_load_settings_clicked(self) -> None:
        """Load burst-wise FCS settings from JSON in the user folder.

        If the user file does not exist yet but a package default is present,
        the default file is copied to the user folder first.
        """

        pkg_settings, user_json = self._get_settings_json_paths()

        # Ensure there is a user JSON, copying from package default if available
        if not user_json.is_file() and pkg_settings.is_file():
            try:
                user_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(pkg_settings, user_json)
            except Exception:
                pass

        if not user_json.is_file():
            QtWidgets.QMessageBox.information(
                self,
                "Burst-wise FCS settings",
                "No burst-wise FCS settings JSON found.\n"
                "Create one with 'Save settings (JSON)' first.",
            )
            return

        try:
            with user_json.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh) or {}
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Burst-wise FCS settings",
                f"Could not read settings file '{user_json}':\n{e}",
            )
            return

        self._apply_settings_dict(cfg)

    # ------------------------------------------------------------------
    # Browser helpers
    # ------------------------------------------------------------------

    def _update_browser_view(self) -> None:

        try:
            self.list_browser.clear()
        except Exception:
            return

        if not self._cached_curves:
            return

        text = str(self.line_burst_filter.text() or "").strip().lower()
        for idx, entry in enumerate(self._cached_curves):
            try:
                index_file = entry.get("Index File") or entry.get("BST File") or entry.get("First File")
                burst_idx = entry.get("Burst Index")
                pair_name = entry.get("pair_name", "")
                path_obj = pathlib.Path(str(index_file)) if index_file else None
                stem = path_obj.stem if path_obj is not None else ""
                label = f"{stem} | Burst {burst_idx} | {pair_name}"
            except Exception:
                continue
            if text:
                haystack = f"{stem} {pair_name}".lower()
                if text not in haystack:
                    continue
            item = QtWidgets.QListWidgetItem(label, self.list_browser)
            if index_file:
                try:
                    item.setToolTip(str(index_file))
                except Exception:
                    pass
            item.setData(QtCore.Qt.UserRole, int(idx))

        if self.list_browser.count() > 0:
            self.list_browser.setCurrentRow(0)
        else:
            self._clear_browser_plots()

    def _clear_browser_plots(self) -> None:

        try:
            self.curve_corr_data_view.setData([], [])
            self.curve_corr_fit_view.setData([], [])
            self.curve_dist_view.setData([], [])
            try:
                # Clear inset text and hide distribution plot when nothing is selected
                if hasattr(self, "text_td_inset"):
                    self.text_td_inset.setText("")
                if hasattr(self, "plot_dist_view"):
                    self.plot_dist_view.setVisible(False)
            except Exception:
                pass
        except Exception:
            pass

    def _on_browser_selection_changed(self, current, _previous) -> None:

        if current is None:
            self._clear_browser_plots()
            return
        idx = current.data(QtCore.Qt.UserRole)
        try:
            i = int(idx)
        except Exception:
            self._clear_browser_plots()
            return
        if i < 0 or i >= len(self._cached_curves):
            self._clear_browser_plots()
            return
        entry = self._cached_curves[i]
        try:
            tau = np.asarray(entry.get("tau", []), dtype=float)
            g = np.asarray(entry.get("g", []), dtype=float)
            g_fit = np.asarray(entry.get("g_fit", []), dtype=float)
            td_grid = np.asarray(entry.get("td_grid", []), dtype=float)
            p = np.asarray(entry.get("p", []), dtype=float)
            td_mean_ms = float(entry.get("td_mean_ms", float("nan")))
            td_peak_ms = float(entry.get("td_peak_ms", float("nan")))
            fit_mode_entry = str(entry.get("fit_mode", "simple")).lower()
        except Exception:
            self._clear_browser_plots()
            return

        try:
            if tau.size and g.size:
                self.curve_corr_data_view.setData(tau, g)
            else:
                self.curve_corr_data_view.setData([], [])

            if tau.size and g_fit.size:
                self.curve_corr_fit_view.setData(tau, g_fit)
            else:
                self.curve_corr_fit_view.setData([], [])

            # Update diffusion-time distribution plot only for MaxEnt-fitted entries
            use_maxent_entry = (fit_mode_entry == "maxent")
            if use_maxent_entry and td_grid.size and p.size:
                self.curve_dist_view.setData(td_grid, p)
                try:
                    self.plot_dist_view.setVisible(True)
                except Exception:
                    pass
            else:
                self.curve_dist_view.setData([], [])
                try:
                    self.plot_dist_view.setVisible(False)
                except Exception:
                    pass

            # Update inset text with mean / peak diffusion times and position it
            try:
                if hasattr(self, "text_td_inset"):
                    parts = []
                    if np.isfinite(td_mean_ms):
                        parts.append(f"td_mean = {td_mean_ms:.3g} ms")
                    if np.isfinite(td_peak_ms):
                        parts.append(f"td_peak = {td_peak_ms:.3g} ms")
                    text = "\n".join(parts)
                    self.text_td_inset.setText(text)

                    # Place the inset near the top-right corner of the current data range
                    if tau.size and g.size and text:
                        try:
                            x = float(np.nanmax(tau))
                            y = float(np.nanmax(g))
                            self.text_td_inset.setPos(x, y)
                        except Exception:
                            pass
                    elif not text:
                        self.text_td_inset.setText("")
            except Exception:
                pass
        except Exception:
            self._clear_browser_plots()

    def _show_pair_config_dialog(self) -> None:

        presets = []
        try:
            if isinstance(getattr(self, "_pair_configs", None), list):
                for p in self._pair_configs:
                    if isinstance(p, dict):
                        presets.append(p)
        except Exception:
            presets = []
        if not presets:
            try:
                corr = getattr(self, "correlator", None)
                raw = getattr(corr, "_fcs_presets", []) if corr is not None else []
                if isinstance(raw, list):
                    for p in raw:
                        if isinstance(p, dict):
                            presets.append(p)
            except Exception:
                presets = []

        if not presets:
            QtWidgets.QMessageBox.information(
                self,
                "FCS channel presets",
                "No FCS channel presets are available for the current detector setup.",
            )
            return

        try:
            text = json.dumps(presets, indent=4, sort_keys=True)
        except Exception:
            text = str(presets)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("FCS channel configuration (JSON)")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        edit = QtWidgets.QPlainTextEdit(dlg)
        edit.setReadOnly(True)
        edit.setPlainText(text)
        layout.addWidget(edit, 1)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, dlg)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        try:
            dlg.resize(800, 500)
        except Exception:
            pass
        try:
            dlg.exec_()
        except Exception:
            dlg.close()

    def _populate_detector_setups(self) -> None:

        try:
            cfg = load_detector_setups()
            setups = cfg.get("setups", {}) if isinstance(cfg, dict) else {}
        except Exception:
            setups = {}

        self._detector_setups = setups if isinstance(setups, dict) else {}

        try:
            self.combo_setup.blockSignals(True)
        except Exception:
            pass
        self.combo_setup.clear()
        self.combo_setup.addItem("")
        for name in sorted(self._detector_setups.keys()):
            self.combo_setup.addItem(str(name))
        try:
            self.combo_setup.blockSignals(False)
        except Exception:
            pass

        self.combo_setup.currentTextChanged.connect(self._on_setup_changed)

    def _on_setup_changed(self, setup_name: str) -> None:

        if not setup_name:
            try:
                self.list_pairs.clear()
            except Exception:
                pass
            self._pair_configs = []
            return
        setups = self._detector_setups or {}
        data = setups.get(setup_name)
        if not isinstance(data, dict):
            return
        detectors = data.get("detectors", {}) or {}
        try:
            self.correlator.load_fcs_presets(setup_name, detectors)
        except Exception:
            pass

        # Rebuild the FCS channel-pair list from the correlator presets
        try:
            self.list_pairs.clear()
        except Exception:
            pass
        self._pair_configs = []

        presets = getattr(self.correlator, "_fcs_presets", []) or []
        for idx, pair in enumerate(presets):
            try:
                cha = str(pair.get("channel_a", ""))
                chb = str(pair.get("channel_b", ""))
                name = str(pair.get("name", ""))
            except Exception:
                continue
            if not name:
                if cha and chb:
                    name = f"{cha}×{chb}" if cha != chb else f"{cha}_ACF"
                else:
                    name = f"Pair {idx + 1}"
            try:
                item = QtWidgets.QListWidgetItem(name, self.list_pairs)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                item.setData(QtCore.Qt.UserRole, int(idx))
            except Exception:
                continue
            self._pair_configs.append(pair)

    # ------------------------------------------------------------------
    # Core burst-wise processing
    # ------------------------------------------------------------------

    def _run_burstwise_fcs(self):
        """Process selected burst analysis folders / .bst files.

        The Files page is expected to contain either:

        - Burst analysis folders (e.g. ``burstwise_All 0.2000#30``), each
          of which contains a ``BID`` subfolder with Seidel BID ``.bst``
          files listing (start, stop) photon indices per burst; or
        - Raw ``.bst`` files directly.

        All discovered BID ``.bst`` files are converted to (TTTR path,
        burst ranges) via :meth:`_parse_bst_file`.

        The correlator configuration (channels, micro-time ranges and
        correlator settings) is taken from the current state of the
        embedded WizardTTTRCorrelator on the Correlator page. Typically
        these fields are populated from an FCS preset.
        """

        pad_ms = 0.0
        spin = getattr(self, "spin_padding_ms", None)
        if spin is not None:
            try:
                pad_ms = float(spin.value())
            except Exception:
                pad_ms = 0.0

        override_n_bins = None
        override_n_casc = None
        override_make_fine = None
        sb_bins = getattr(self, "spin_n_bins", None)
        if sb_bins is not None:
            try:
                v = int(sb_bins.value())
                if v > 0:
                    override_n_bins = v
            except Exception:
                pass
        sb_casc = getattr(self, "spin_n_casc", None)
        if sb_casc is not None:
            try:
                v = int(sb_casc.value())
                if v > 0:
                    override_n_casc = v
            except Exception:
                pass

        try:
            cb_fine = getattr(self, "check_fine", None)
            if cb_fine is not None:
                override_make_fine = bool(cb_fine.isChecked())
        except Exception:
            override_make_fine = None

        fit_mode = "simple"
        try:
            rb_maxent = getattr(self, "radio_fit_maxent", None)
            rb_simple = getattr(self, "radio_fit_simple", None)
            if rb_maxent is not None and rb_maxent.isChecked():
                fit_mode = "maxent"
            elif rb_simple is not None and rb_simple.isChecked():
                fit_mode = "simple"
        except Exception:
            fit_mode = "simple"

        maxent_reg = None
        try:
            sb_reg = getattr(self, "spin_log10_reg", None)
            if sb_reg is not None:
                log10_reg = float(sb_reg.value())
                maxent_reg = float(10.0 ** log10_reg)
        except Exception:
            maxent_reg = None
        if maxent_reg is None or not np.isfinite(maxent_reg) or maxent_reg <= 0.0:
            maxent_reg = 0.1

        maxent_td_min = None
        maxent_td_max = None
        try:
            sb_td_min = getattr(self, "spin_td_min", None)
            if sb_td_min is not None:
                v = float(sb_td_min.value())
                if v > 0.0:
                    maxent_td_min = v
        except Exception:
            maxent_td_min = None
        try:
            sb_td_max = getattr(self, "spin_td_max", None)
            if sb_td_max is not None:
                v = float(sb_td_max.value())
                if v > 0.0:
                    maxent_td_max = v
        except Exception:
            maxent_td_max = None
        if maxent_td_min is not None and maxent_td_max is not None and maxent_td_max <= maxent_td_min:
            maxent_td_max = None

        # Optional analysis / fitting time-window in correlation time (ms)
        tmin_fit = None
        tmax_fit = None
        try:
            sb_tmin = getattr(self, "spin_tmin_fit", None)
            if sb_tmin is not None:
                v = float(sb_tmin.value())
                if v > 0.0:
                    tmin_fit = v
        except Exception:
            tmin_fit = None
        try:
            sb_tmax = getattr(self, "spin_tmax_fit", None)
            if sb_tmax is not None:
                v = float(sb_tmax.value())
                if v > 0.0:
                    tmax_fit = v
        except Exception:
            tmax_fit = None
        if tmin_fit is not None and tmax_fit is not None and tmax_fit <= tmin_fit:
            tmax_fit = None

        # 1) Collect burst analysis folders / BUR + BST files from the file list
        try:
            all_files = list(self.file_list.checked_files())
        except Exception:
            all_files = []

        bur_sources: List[Tuple[pathlib.Path, pathlib.Path]] = []  # (bur_path, analysis_folder)
        bst_sources: List[Tuple[pathlib.Path, pathlib.Path]] = []  # (bst_path, analysis_folder)

        for entry in all_files:
            try:
                p = pathlib.Path(entry)
            except Exception:
                continue

            if p.is_dir():
                # Treat directory as a burstwise analysis folder. Prefer bi4_bur/bur
                # with .bur files; fall back to BID/*.bst if no .bur is found.
                analysis_root = p
                found_bur = False

                for sub_name in ("bi4_bur", "bur"):
                    subdir = analysis_root / sub_name
                    if not subdir.is_dir():
                        continue
                    try:
                        for bur in sorted(subdir.glob("*.bur")):
                            bur_sources.append((bur, analysis_root))
                            found_bur = True
                    except Exception:
                        continue

                if not found_bur:
                    bid_dir = analysis_root / "BID"
                    if bid_dir.is_dir():
                        try:
                            for bst in sorted(bid_dir.glob("*.bst")):
                                bst_sources.append((bst, analysis_root))
                        except Exception:
                            continue

            elif p.is_file():
                suffix = p.suffix.lower()
                if suffix == ".bur":
                    if p.parent.name in ("bi4_bur", "bur"):
                        analysis_root = p.parent.parent
                    else:
                        analysis_root = p.parent
                    bur_sources.append((p, analysis_root))
                elif suffix == ".bst":
                    bst_sources.append((p, p.parent))

        if not bur_sources and not bst_sources:
            QtWidgets.QMessageBox.information(
                self,
                "Burst-wise FCS",
                "No burst analysis folders or BUR/BID files selected.",
            )
            return None

        # 2) Build list of selected FCS channel pairs from presets
        corr = self.correlator

        presets = getattr(corr, "_fcs_presets", []) or []
        if not presets:
            QtWidgets.QMessageBox.warning(
                self,
                "Burst-wise FCS",
                "No FCS channel pairs are defined for the selected detector setup.\n"
                "Use the 'FCS Channel Definitions' plugin to create them.",
            )
            return None

        selected_cfgs: List[Dict[str, Any]] = []
        cb_preset = getattr(corr, "comboBox_fcs_preset", None)

        for row in range(getattr(self, "list_pairs", QtWidgets.QListWidget()).count()):
            try:
                item = self.list_pairs.item(row)
            except Exception:
                item = None
            if item is None or item.checkState() != QtCore.Qt.Checked:
                continue
            idx_data = item.data(QtCore.Qt.UserRole)
            try:
                p_idx = int(idx_data)
            except Exception:
                continue
            if p_idx < 0 or p_idx >= len(presets):
                continue

            pair = presets[p_idx]
            try:
                cha_name = str(pair.get("channel_a", ""))
                chb_name = str(pair.get("channel_b", ""))
            except Exception:
                continue

            try:
                pair_name = str(pair.get("name", ""))
            except Exception:
                pair_name = ""
            if not pair_name:
                if cha_name and chb_name:
                    pair_name = f"{cha_name}×{chb_name}" if cha_name != chb_name else f"{cha_name}_ACF"
                else:
                    pair_name = f"Pair {p_idx + 1}"

            # Let the correlator apply this preset so that channels,
            # micro-time ranges and correlator settings are updated consistently.
            try:
                if cb_preset is not None:
                    cb_preset.setCurrentIndex(p_idx + 1)
            except Exception:
                pass

            ch_a_text = str(getattr(corr, 'lineEdit', None).text()) if getattr(corr, 'lineEdit', None) is not None else ""
            ch_b_text = str(getattr(corr, 'lineEdit_2', None).text()) if getattr(corr, 'lineEdit_2', None) is not None else ""
            chs_a = parse_channel_list(ch_a_text)
            chs_b = parse_channel_list(ch_b_text)
            if not chs_a or not chs_b:
                continue

            try:
                micro_a = corr.microtime_range_a
                micro_b = corr.microtime_range_b
            except Exception:
                micro_a = []
                micro_b = []

            try:
                n_bins = int(corr.correlation_nbins)
                n_casc = int(corr.correlation_ncasc)
                make_fine = bool(corr.correlation_is_fine)
            except Exception:
                from chisurf import settings as _cs_settings
                n_bins = int(_cs_settings.cs_settings['correlator']['B'])
                n_casc = int(_cs_settings.cs_settings['correlator']['number_of_cascades'])
                make_fine = bool(_cs_settings.cs_settings['correlator']['fine'])

            if override_n_bins is not None:
                try:
                    n_bins = int(override_n_bins)
                except Exception:
                    pass
            if override_n_casc is not None:
                try:
                    n_casc = int(override_n_casc)
                except Exception:
                    pass

            if override_make_fine is not None:
                try:
                    make_fine = bool(override_make_fine)
                except Exception:
                    pass

            selected_cfgs.append({
                "preset_index": p_idx,
                "pair_name": pair_name,
                "channel_a_logical": cha_name,
                "channel_b_logical": chb_name,
                "chs_a": chs_a,
                "chs_b": chs_b,
                "micro_a": micro_a,
                "micro_b": micro_b,
                "n_bins": n_bins,
                "n_casc": n_casc,
                "make_fine": make_fine,
            })

        if not selected_cfgs:
            QtWidgets.QMessageBox.warning(
                self,
                "Burst-wise FCS",
                "No FCS channel pairs are selected. Enable at least one pair in the list.",
            )
            return None

        # Reset caches for this run
        self._pair_configs = selected_cfgs
        self._cached_curves = []

        # 3) Iterate over all bursts in all discovered BUR/BST files
        #    and compute per-burst FCS + diffusion times for each pair
        rows: List[Dict[str, Any]] = []

        # Estimate total number of bursts for progress dialog
        total_bursts = 0
        burst_info: List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path, List[Tuple[int, int]]]] = []

        for p_bur, analysis_root in bur_sources:
            tttr_path, ranges = parse_bur_file(p_bur, analysis_root)
            if tttr_path is None or not ranges:
                continue
            burst_info.append((p_bur, analysis_root, tttr_path, ranges))
            total_bursts += len(ranges)

        for p_bst, analysis_root in bst_sources:
            tttr_path, ranges = parse_bst_file(p_bst)
            if tttr_path is None or not ranges:
                continue
            burst_info.append((p_bst, analysis_root, tttr_path, ranges))
            total_bursts += len(ranges)

        if total_bursts == 0:
            QtWidgets.QMessageBox.information(self, "Burst-wise FCS", "No bursts found in the selected burst files.")
            return None

        progress = QtWidgets.QProgressDialog("Computing burst-wise FCS...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Burst-wise FCS")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        current = 0
        for index_path, analysis_root, tttr_path, ranges in burst_info:
            # Open underlying TTTR file once per index file (.bur or .bst)
            # We rely on open_tttr's internal heuristics and optional inference
            filetype = None
            tttr = open_tttr(tttr_path, filetype)
            if tttr is None:
                continue
            n_events = None
            try:
                n_events = len(tttr)
            except Exception:
                n_events = None

            macro_res_s = 0.0
            mt = None
            pad_ticks = 0.0
            if pad_ms > 0.0:
                try:
                    macro_res_s = float(tttr.header.macro_time_resolution)
                except Exception:
                    try:
                        macro_res_s = float(getattr(tttr, "macro_time_resolution", 0.0))
                    except Exception:
                        macro_res_s = 0.0
                if macro_res_s > 0.0:
                    try:
                        macro_t = tttr.macro_times
                    except Exception:
                        macro_t = None
                    if macro_t is not None:
                        try:
                            mt = np.asarray(macro_t, dtype=float)
                        except Exception:
                            mt = None
                    if mt is not None and mt.size > 0:
                        pad_ticks = (pad_ms / 1000.0) / macro_res_s

            for local_idx, (start, stop) in enumerate(ranges):
                if progress.wasCanceled():
                    progress.close()
                    return None
                current += 1
                progress.setValue(current)
                if (current % 25) == 0:
                    QtWidgets.QApplication.processEvents()

                try:
                    s = int(start)
                    e = int(stop)
                except Exception:
                    continue
                if n_events is not None:
                    if s < 0:
                        s = 0
                    if e >= n_events:
                        e = n_events - 1
                    if e < s:
                        continue

                if pad_ms > 0.0 and mt is not None and mt.size > 0 and pad_ticks > 0.0:
                    try:
                        t_s = float(mt[s])
                        t_e = float(mt[e])
                        t_min = max(0.0, t_s - pad_ticks)
                        t_max = t_e + pad_ticks
                        s_pad = int(np.searchsorted(mt, t_min, side="left"))
                        e_pad = int(np.searchsorted(mt, t_max, side="right") - 1)
                        if s_pad < 0:
                            s_pad = 0
                        if e_pad >= mt.size:
                            e_pad = mt.size - 1
                        if e_pad >= s_pad:
                            s, e = s_pad, e_pad
                    except Exception:
                        pass

                try:
                    # Inclusive range: [s, e]
                    tttr_burst = tttr[s:e + 1]
                except Exception:
                    continue

                for cfg in selected_cfgs:
                    tau, g = correlate_single_burst(
                        tttr_burst,
                        chs_a=cfg["chs_a"],
                        chs_b=cfg["chs_b"],
                        micro_a=cfg["micro_a"],
                        micro_b=cfg["micro_b"],
                        n_bins=cfg["n_bins"],
                        n_casc=cfg["n_casc"],
                        make_fine=cfg["make_fine"],
                    )
                    if tau is None or g is None:
                        continue

                    td_mean = float('nan')
                    td_peak = float('nan')
                    fit_result = None
                    tau_arr = np.asarray(tau, dtype=float)
                    g_arr = np.asarray(g, dtype=float)

                    # Apply user-defined analysis window in correlation time
                    if tmin_fit is not None or tmax_fit is not None:
                        try:
                            mask = np.isfinite(tau_arr) & np.isfinite(g_arr)
                            if tmin_fit is not None:
                                mask &= tau_arr >= tmin_fit
                            if tmax_fit is not None:
                                mask &= tau_arr <= tmax_fit
                            tau_arr = tau_arr[mask]
                            g_arr = g_arr[mask]
                        except Exception:
                            pass

                    tau_fit = tau_arr
                    g_data = g_arr
                    g_fit = np.asarray([], dtype=float)
                    td_grid = np.asarray([], dtype=float)
                    p = np.asarray([], dtype=float)

                    if fit_mode == "maxent":
                        try:
                            fit_result = fcs_maxent(
                                tau=tau_arr,
                                g=g_arr,
                                td_min=maxent_td_min,
                                td_max=maxent_td_max,
                                reg=maxent_reg,
                            )
                        except Exception:
                            fit_result = None

                        if isinstance(fit_result, dict):
                            try:
                                tau_fit = np.asarray(fit_result.get("tau", tau_arr), dtype=float)
                                g_data = np.asarray(fit_result.get("g", g_arr), dtype=float)
                                g_fit = np.asarray(fit_result.get("g_fit", []), dtype=float)
                                td_grid = np.asarray(fit_result.get("td_grid", []), dtype=float)
                                p = np.asarray(fit_result.get("p", []), dtype=float)
                            except Exception:
                                tau_fit = tau_arr
                                g_data = g_arr
                                g_fit = np.asarray([], dtype=float)
                                td_grid = np.asarray([], dtype=float)
                                p = np.asarray([], dtype=float)

                            if td_grid.size > 0 and p.size > 0:
                                try:
                                    p_clip = np.clip(p, 0.0, np.inf)
                                    if np.any(p_clip > 0.0):
                                        td_mean = float(np.sum(td_grid * p_clip) / np.sum(p_clip))
                                        td_peak = float(td_grid[np.argmax(p_clip)])
                                    else:
                                        td_peak = float(td_grid[np.argmax(p_clip)])
                                except Exception:
                                    td_mean = float('nan')
                                    td_peak = float('nan')
                    else:
                        td_est, tau_used, g_used, g_fit_arr = fit_simple_diffusion(tau_arr, g_arr)
                        if np.isfinite(td_est):
                            td_mean = float(td_est)
                            td_peak = float(td_est)
                            # Provide a delta-like distribution for the diffusion-time plot
                            try:
                                td_grid = np.asarray([td_est], dtype=float)
                                p = np.asarray([1.0], dtype=float)
                            except Exception:
                                td_grid = np.asarray([], dtype=float)
                                p = np.asarray([], dtype=float)

                        # Use the same (tau, g) grid as was used for the fit so that
                        # the data and fitted curve have matching lengths.
                        if tau_used is not None and np.size(tau_used) and g_used is not None and np.size(g_used):
                            try:
                                tau_fit = np.asarray(tau_used, dtype=float)
                                g_data = np.asarray(g_used, dtype=float)
                            except Exception:
                                tau_fit = tau_arr
                                g_data = g_arr

                        if g_fit_arr is not None and np.size(g_fit_arr):
                            try:
                                g_fit = np.asarray(g_fit_arr, dtype=float)
                            except Exception:
                                g_fit = np.asarray([], dtype=float)

                    rows.append({
                        "First File": tttr_path.as_posix(),
                        "BST File": index_path.as_posix(),
                        "Burst Folder": analysis_root.as_posix(),
                        "Burst Index": local_idx,
                        "Burst Start": int(start),
                        "Burst End": int(stop),
                        "pair_name": cfg["pair_name"],
                        "pair_channel_a": cfg["channel_a_logical"],
                        "pair_channel_b": cfg["channel_b_logical"],
                        "td_mean_ms": float(td_mean),
                        "td_peak_ms": float(td_peak),
                        "n_bins": int(cfg["n_bins"]),
                        "n_casc": int(cfg["n_casc"]),
                        "make_fine": bool(cfg["make_fine"]),
                    })

                    try:
                        self._cached_curves.append({
                            "First File": tttr_path.as_posix(),
                            "Burst Folder": analysis_root.as_posix(),
                            "Index File": index_path.as_posix(),
                            "Burst Index": local_idx,
                            "pair_name": cfg["pair_name"],
                            "pair_channel_a": cfg["channel_a_logical"],
                            "pair_channel_b": cfg["channel_b_logical"],
                            "td_mean_ms": float(td_mean),
                            "td_peak_ms": float(td_peak),
                            "fit_mode": str(fit_mode),
                            "tau": np.asarray(tau_fit, dtype=float),
                            "g": np.asarray(g_data, dtype=float),
                            "g_fit": np.asarray(g_fit, dtype=float),
                            "td_grid": np.asarray(td_grid, dtype=float),
                            "p": np.asarray(p, dtype=float),
                        })
                    except Exception:
                        pass

        progress.close()

        if not rows:
            QtWidgets.QMessageBox.information(self, "Burst-wise FCS", "No valid burst correlations could be computed.")
            return None

        return pd.DataFrame(rows)

    def _save_td4_results(self, result_df: pd.DataFrame) -> None:
        """Write diffusion times to td4-style files in the burst analysis folder.

        Layout (per burst analysis folder and underlying TTTR file):

        - Folder: ``<Burst Folder>/td4`` where ``<Burst Folder>`` is the
          burstwise analysis directory (sibling of ``bi4_bur``).
        - File:   ``<stem>.td4`` where ``stem`` is derived from the
          underlying TTTR file name.
        - Header: ``Burst Index`` followed by one or more diffusion-time
          columns, e.g. ``td_mean_ms``, ``td_peak_ms``, and additional
          columns for other correlation pairs if present. All columns are
          tab-separated, with a trailing tab to mimic the b*4 writer.
        - Body:   zero-interleaved rows (zero row, data row, zero row, ...)
          for compatibility with existing burst result formats.
        """

        if result_df is None or result_df.empty:
            QtWidgets.QMessageBox.information(self, "Burst-wise FCS", "No results to save.")
            return

        df = result_df.copy()
        if 'Burst Folder' not in df.columns:
            df['Burst Folder'] = df['First File'].map(lambda fn: str(pathlib.Path(fn).parent))
        df['First Stem'] = df['First File'].map(lambda fn: pathlib.Path(fn).stem)

        # Group by burst analysis folder and TTTR stem so that results from
        # different burstwise folders are written to separate td4 files.
        groups = df.groupby(['Burst Folder', 'First Stem'], sort=False)
        total_groups = len(groups)

        progress = QtWidgets.QProgressDialog("Saving td4 results...", "Cancel", 0, total_groups, self)
        progress.setWindowTitle("Burst-wise FCS")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        wrote_settings_for: set[pathlib.Path] = set()
        current = 0

        for (burst_folder, stem), df_g in groups:
            current += 1
            progress.setValue(current)
            if progress.wasCanceled():
                progress.close()
                QtWidgets.QMessageBox.information(self, "Burst-wise FCS", "Save operation was canceled.")
                return

            # Determine output directory inside the burst analysis folder
            try:
                burst_root = pathlib.Path(burst_folder)
            except Exception:
                try:
                    first_file = pathlib.Path(str(df_g['First File'].iloc[0]))
                    burst_root = first_file.parent
                except Exception:
                    continue

            out_dir = burst_root / "td4"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Build wide (multi-column) table: one row per burst index and
            # separate td_* columns per FCS pair.
            if 'pair_name' in df_g.columns:
                df_g_sorted = df_g.sort_values(['Burst Index', 'pair_name'])
                burst_ids = np.sort(df_g_sorted['Burst Index'].unique())
                wide_df = pd.DataFrame({'Burst Index': burst_ids})

                pair_names = [str(p) for p in df_g_sorted['pair_name'].dropna().unique()]
                for pname in pair_names:
                    sub = df_g_sorted[df_g_sorted['pair_name'] == pname].set_index('Burst Index')
                    if 'td_mean_ms' in sub.columns:
                        col_mean = f"td_mean__{pname}"
                        try:
                            wide_df[col_mean] = sub['td_mean_ms'].reindex(burst_ids).to_numpy(dtype=float)
                        except Exception:
                            wide_df[col_mean] = np.nan
                    if 'td_peak_ms' in sub.columns:
                        col_peak = f"td_peak__{pname}"
                        try:
                            wide_df[col_peak] = sub['td_peak_ms'].reindex(burst_ids).to_numpy(dtype=float)
                        except Exception:
                            wide_df[col_peak] = np.nan

                value_cols = [c for c in wide_df.columns if c.startswith('td_')]
                cols = ["Burst Index"] + value_cols

                try:
                    arr = wide_df[cols].to_numpy(dtype=float, copy=False)
                except Exception:
                    continue
            else:
                df_g_sorted = df_g.sort_values('Burst Index')
                df_for_save = df_g_sorted.copy()
                if 'td_mean_ms' in df_for_save.columns:
                    df_for_save = df_for_save.rename(columns={'td_mean_ms': 'td_mean'})
                if 'td_peak_ms' in df_for_save.columns:
                    df_for_save = df_for_save.rename(columns={'td_peak_ms': 'td_peak'})
                value_cols = [c for c in df_for_save.columns if c.startswith('td_')]
                cols = ["Burst Index"] + value_cols

                try:
                    arr = df_for_save[cols].to_numpy(dtype=float, copy=False)
                except Exception:
                    continue

            out = np.zeros((arr.shape[0] * 2 + 1, arr.shape[1]), dtype=float)
            out[1::2] = arr

            out_file = out_dir / f"{stem}.td4"
            try:
                with out_file.open('w', newline='') as f:
                    f.write('\t'.join(cols) + '\t\n')
                    np.savetxt(f, out, delimiter='\t', fmt='%.6f')
            except Exception:
                continue

            # Write a small JSON sidecar once per td4 folder with meta info
            if out_dir not in wrote_settings_for:
                try:
                    df_g_sorted_meta = df_g.sort_values(['Burst Index', 'pair_name']) if 'pair_name' in df_g.columns else df_g
                    meta: Dict[str, Any] = {}
                    if 'pair_name' in df_g_sorted_meta.columns:
                        pairs_meta: List[Dict[str, Any]] = []
                        for pname in df_g_sorted_meta['pair_name'].dropna().unique():
                            sub = df_g_sorted_meta[df_g_sorted_meta['pair_name'] == pname]
                            if sub.empty:
                                continue
                            row0 = sub.iloc[0]
                            pairs_meta.append({
                                "name": str(pname),
                                "channel_a": str(row0.get("pair_channel_a", "")),
                                "channel_b": str(row0.get("pair_channel_b", "")),
                                "n_bins": int(row0.get("n_bins", 0)),
                                "n_casc": int(row0.get("n_casc", 0)),
                                "make_fine": bool(row0.get("make_fine", False)),
                            })
                        meta["pairs"] = pairs_meta
                    else:
                        meta = {
                            "channel_a": str(df_g.get("channel_a").iloc[0] if "channel_a" in df_g else ""),
                            "channel_b": str(df_g.get("channel_b").iloc[0] if "channel_b" in df_g else ""),
                            "n_bins": int(df_g.get("n_bins").iloc[0] if "n_bins" in df_g else 0),
                            "n_casc": int(df_g.get("n_casc").iloc[0] if "n_casc" in df_g else 0),
                            "make_fine": bool(df_g.get("make_fine").iloc[0] if "make_fine" in df_g else False),
                        }

                    settings_file = out_dir / 'td4_settings.json'
                    with settings_file.open('w', encoding='utf-8') as sf:
                        json.dump(meta, sf, indent=4, sort_keys=False)
                    wrote_settings_for.add(out_dir)
                except Exception:
                    pass

        progress.close()

    # ------------------------------------------------------------------
    # Finish hook
    # ------------------------------------------------------------------

    def onFinish(self):  # type: ignore[override]
        """Override the base wizard's Finish behavior for burst-wise FCS."""

        try:
            result_df = self._run_burstwise_fcs()
        except Exception as e:  # pragma: no cover - defensive UI layer
            QtWidgets.QMessageBox.critical(
                self,
                "Burst-wise FCS Error",
                f"An unexpected error occurred during burst-wise correlation:\n{e}",
            )
            return

        if result_df is None or result_df.empty:
            # _run_burstwise_fcs already informed the user
            return

        self._save_td4_results(result_df)
        # Refresh integrated browser view with newly cached curves
        try:
            self._update_browser_view()
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self,
            "Burst-wise FCS",
            "Finished burst-wise FCS analysis and saved td4 files.",
        )


if __name__ == "plugin":  # pragma: no cover
    # Invoked by the ChiSurf plugin manager
    wizard = BurstWiseFCSWizard()
    wizard.show()


if __name__ == "__main__":  # pragma: no cover
    # Allow running this module directly as a macro/script
    app = QtWidgets.QApplication.instance()
    if app is None:
        from qtpy import QtWidgets as _QtWidgets  # fallback if run outside ChiSurf
        app = _QtWidgets.QApplication([])
    dlg = BurstWiseFCSWizard()
    dlg.show()
    # Only start a local event loop if we created our own QApplication
    try:
        app.exec()
    except Exception:
        pass

