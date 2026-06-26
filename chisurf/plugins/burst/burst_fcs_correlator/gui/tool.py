"""Burst-wise FCS correlator — declarative AutoForm tool.

A QMainWindow whose settings are declared in ``burst_fcs.view.json`` and whose
correlation / distribution plots live in a declarative ``dock_area`` section
(``burst_fcs_plots.view.json``). Compute goes through :class:`BurstFcsClient`
(the backend RPC). The legacy ``wizard.py`` remains available; this is its
modern replacement.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups
from chisurf.gui.widgets.wizard.tttr_correlator import WizardTTTRCorrelator

from ..core.algorithms import BurstFcsSettings, PairConfig, parse_channel_list
from ..file_list import BurstFileListWidget
from .client import BurstFcsClient

_GUI_DIR = pathlib.Path(__file__).parent


class _BurstFcsModel:
    """Settings model + plot data sources for the declarative editor."""

    def __init__(self) -> None:
        self.n_bins = 3
        self.n_casc = 20
        self.make_fine = False
        self.padding_ms = 100.0
        self.fit_mode = "simple"
        self.maxent_log10_reg = 0.0
        self.maxent_td_min = 0.0
        self.maxent_td_max = 0.0
        self.tmin_fit = 0.0
        self.tmax_fit = 0.0
        self._selected: Optional[Dict[str, Any]] = None

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_GUI_DIR / "burst_fcs.view.json")

    # -- settings <-> core ---------------------------------------------
    def to_settings(self) -> BurstFcsSettings:
        def _opt(v):
            return float(v) if v and float(v) > 0.0 else None
        return BurstFcsSettings(
            n_bins=int(self.n_bins),
            n_casc=int(self.n_casc),
            make_fine=bool(self.make_fine),
            padding_ms=float(self.padding_ms),
            fit_mode=str(self.fit_mode),
            maxent_reg=float(10.0 ** float(self.maxent_log10_reg)),
            maxent_td_min=_opt(self.maxent_td_min),
            maxent_td_max=_opt(self.maxent_td_max),
            tmin_fit=_opt(self.tmin_fit),
            tmax_fit=_opt(self.tmax_fit),
        )

    # -- declarative plot sources --------------------------------------
    def corr_plot_series(self) -> List[Dict[str, Any]]:
        c = self._selected
        if not c:
            return []
        series = [{"x": c.get("tau_raw", c.get("tau", [])), "y": c.get("g_raw", c.get("g", [])),
                   "name": "data", "color": "w"}]
        g_fit = c.get("g_fit") or []
        tau = c.get("tau") or []
        if len(g_fit) and len(tau) == len(g_fit):
            series.append({"x": tau, "y": g_fit, "name": "fit", "color": "r", "width": 2})
        return series

    def dist_plot_series(self) -> List[Dict[str, Any]]:
        c = self._selected
        if not c:
            return []
        td = c.get("td_grid") or []
        p = c.get("p") or []
        if len(td) and len(td) == len(p):
            return [{"x": td, "y": p, "name": "P(τ_D)", "color": "y", "width": 2}]
        return []


class _PlotsProxy:
    """Tiny proxy so a second AutoForm renders the plots dock_area spec."""

    def __init__(self, model: _BurstFcsModel):
        self._model = model

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_GUI_DIR / "burst_fcs_plots.view.json")

    def corr_plot_series(self):
        return self._model.corr_plot_series()

    def dist_plot_series(self):
        return self._model.dist_plot_series()


class BurstFcsTool(QtWidgets.QMainWindow):
    """Modern burst-wise FCS correlator (declarative AutoForm + dockable plots)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔬 Burst-wise FCS Correlator")
        self.resize(960, 620)

        self._client = BurstFcsClient()
        self._model = _BurstFcsModel()
        self._detector_setups: Dict[str, Any] = {}
        self._pair_presets: List[Dict[str, Any]] = []
        self._curves: List[Dict[str, Any]] = []
        # Hidden correlator reused only to resolve FCS presets → channel lists.
        self._corr = WizardTTTRCorrelator()

        self._build_toolbar()
        self._build_central()
        self._populate_detector_setups()

    # ------------------------------------------------------------------
    # Toolbar: Run action + an emoji "≡ Settings" pop-up menu (no loose
    # tool buttons — the JSON actions live in the menu).
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        self.act_run = QtWidgets.QAction("▶ Run", self)
        self.act_run.setToolTip("Run burst-wise FCS on the checked files / pairs")
        self.act_run.triggered.connect(self._on_run)
        tb.addAction(self.act_run)

        tb.addSeparator()

        menu_btn = QtWidgets.QToolButton(self)
        menu_btn.setText("≡ Settings")
        menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(menu_btn)
        menu.addAction("📂 Load settings…", self._on_load_settings)
        menu.addAction("💾 Save settings…", self._on_save_settings)
        menu.addSeparator()
        menu.addAction("🔬 Show FCS pairs JSON", self._on_show_pairs_json)
        menu_btn.setMenu(menu)
        tb.addWidget(menu_btn)

    # ------------------------------------------------------------------
    def _build_central(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Left: detector setup + declarative settings + pairs + files
        left = QtWidgets.QWidget()
        lcol = QtWidgets.QVBoxLayout(left)
        lcol.setContentsMargins(6, 6, 6, 6)
        lcol.setSpacing(6)

        setup_row = QtWidgets.QHBoxLayout()
        setup_row.addWidget(QtWidgets.QLabel("Detector setup:"))
        self.combo_setup = QtWidgets.QComboBox()
        setup_row.addWidget(self.combo_setup, 1)
        lcol.addLayout(setup_row)

        from chisurf.gui.autoform import AutoForm
        self._settings_form = AutoForm(self._model, parent=self)
        lcol.addWidget(self._settings_form)

        lcol.addWidget(QtWidgets.QLabel("FCS channel pairs:"))
        self.list_pairs = QtWidgets.QListWidget()
        self.list_pairs.setMaximumHeight(120)
        lcol.addWidget(self.list_pairs)

        lcol.addWidget(QtWidgets.QLabel("Burst folders or BUR/BST files:"))
        self.file_list = BurstFileListWidget(left)
        lcol.addWidget(self.file_list, 1)

        # Right: browser filter + curve list + declarative dock_area plots
        right = QtWidgets.QWidget()
        rcol = QtWidgets.QVBoxLayout(right)
        rcol.setContentsMargins(6, 6, 6, 6)
        rcol.setSpacing(6)

        self.line_filter = QtWidgets.QLineEdit()
        self.line_filter.setPlaceholderText("🔎 Filter by file or pair name (e.g. 'GG')")
        self.line_filter.textChanged.connect(self._refresh_browser_list)
        rcol.addWidget(self.line_filter)

        self.list_browser = QtWidgets.QListWidget()
        self.list_browser.setMaximumHeight(160)
        self.list_browser.currentRowChanged.connect(self._on_browser_selected)
        rcol.addWidget(self.list_browser)

        self._plots_proxy = _PlotsProxy(self._model)
        self._plots_form = AutoForm(self._plots_proxy, parent=self)
        rcol.addWidget(self._plots_form, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Detector setups → FCS channel pairs
    # ------------------------------------------------------------------
    def _populate_detector_setups(self) -> None:
        try:
            cfg = load_detector_setups()
            setups = cfg.get("setups", {}) if isinstance(cfg, dict) else {}
        except Exception:
            setups = {}
        self._detector_setups = setups if isinstance(setups, dict) else {}
        self.combo_setup.blockSignals(True)
        self.combo_setup.clear()
        self.combo_setup.addItem("")
        for name in sorted(self._detector_setups.keys()):
            self.combo_setup.addItem(str(name))
        self.combo_setup.blockSignals(False)
        self.combo_setup.currentTextChanged.connect(self._on_setup_changed)

    def _on_setup_changed(self, setup_name: str) -> None:
        self.list_pairs.clear()
        self._pair_presets = []
        if not setup_name:
            return
        data = (self._detector_setups or {}).get(setup_name)
        if not isinstance(data, dict):
            return
        detectors = data.get("detectors", {}) or {}
        try:
            self._corr.load_fcs_presets(setup_name, detectors)
        except Exception:
            pass
        presets = getattr(self._corr, "_fcs_presets", []) or []
        for idx, pair in enumerate(presets):
            cha = str(pair.get("channel_a", ""))
            chb = str(pair.get("channel_b", ""))
            name = str(pair.get("name", "")) or (
                (f"{cha}×{chb}" if cha != chb else f"{cha}_ACF") if cha and chb else f"Pair {idx + 1}"
            )
            item = QtWidgets.QListWidgetItem(name, self.list_pairs)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            item.setData(QtCore.Qt.UserRole, int(idx))
            self._pair_presets.append(pair)

    def _selected_pairs(self) -> List[PairConfig]:
        """Resolve the checked presets to PairConfig (channel int-lists)."""
        pairs: List[PairConfig] = []
        cb_preset = getattr(self._corr, "comboBox_fcs_preset", None)
        for row in range(self.list_pairs.count()):
            item = self.list_pairs.item(row)
            if item is None or item.checkState() != QtCore.Qt.Checked:
                continue
            p_idx = int(item.data(QtCore.Qt.UserRole))
            if not (0 <= p_idx < len(self._pair_presets)):
                continue
            try:
                if cb_preset is not None:
                    cb_preset.setCurrentIndex(p_idx + 1)
            except Exception:
                pass
            ch_a = parse_channel_list(getattr(self._corr, "lineEdit", None).text()
                                      if getattr(self._corr, "lineEdit", None) else "")
            ch_b = parse_channel_list(getattr(self._corr, "lineEdit_2", None).text()
                                      if getattr(self._corr, "lineEdit_2", None) else "")
            if not ch_a or not ch_b:
                continue
            try:
                micro_a = self._corr.microtime_range_a
                micro_b = self._corr.microtime_range_b
            except Exception:
                micro_a, micro_b = [], []
            pairs.append(PairConfig(
                pair_name=item.text(), chs_a=ch_a, chs_b=ch_b,
                micro_a=list(micro_a or []), micro_b=list(micro_b or []),
            ))
        return pairs

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _resolve_files(self) -> List[Dict[str, Any]]:
        """Return [{tttr_path, ranges}] from the checked file-list entries."""
        try:
            entries = list(self.file_list.checked_files())
        except Exception:
            entries = []
        out: List[Dict[str, Any]] = []
        for entry in entries:
            p = pathlib.Path(entry)
            if p.is_dir():
                for sub in ("bi4_bur", "bur"):
                    d = p / sub
                    if d.is_dir():
                        for bur in sorted(d.glob("*.bur")):
                            r = self._client.parse_bur(str(bur), str(p))
                            res = r.get("result", {})
                            if res.get("tttr_path") and res.get("ranges"):
                                out.append(res)
                bid = p / "BID"
                if bid.is_dir():
                    for bst in sorted(bid.glob("*.bst")):
                        r = self._client.parse_bst(str(bst))
                        res = r.get("result", {})
                        if res.get("tttr_path") and res.get("ranges"):
                            out.append(res)
            elif p.is_file():
                if p.suffix.lower() == ".bur":
                    root = p.parent.parent if p.parent.name in ("bi4_bur", "bur") else p.parent
                    res = self._client.parse_bur(str(p), str(root)).get("result", {})
                    if res.get("tttr_path") and res.get("ranges"):
                        out.append(res)
                elif p.suffix.lower() == ".bst":
                    res = self._client.parse_bst(str(p)).get("result", {})
                    if res.get("tttr_path") and res.get("ranges"):
                        out.append(res)
        return out

    def _on_run(self) -> None:
        pairs = self._selected_pairs()
        if not pairs:
            QtWidgets.QMessageBox.warning(self, "Burst-wise FCS",
                                          "Select a detector setup and at least one FCS pair.")
            return
        files = self._resolve_files()
        if not files:
            QtWidgets.QMessageBox.information(self, "Burst-wise FCS",
                                             "No burst folders or BUR/BST files selected.")
            return

        settings = self.to_settings_dict()
        pair_dicts = [{"pair_name": p.pair_name, "chs_a": p.chs_a, "chs_b": p.chs_b,
                       "micro_a": p.micro_a, "micro_b": p.micro_b} for p in pairs]

        progress = QtWidgets.QProgressDialog("Computing burst-wise FCS…", "Cancel",
                                             0, len(files), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Burst-wise FCS")
        progress.show()

        self._curves = []
        for i, f in enumerate(files):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()
            try:
                r = self._client.correlate_file(
                    f["tttr_path"], f["ranges"], pair_dicts, settings,
                )
                self._curves.extend(r.get("result", {}).get("curves", []))
            except Exception:
                continue
        progress.setValue(len(files))

        self._refresh_browser_list()
        if not self._curves:
            QtWidgets.QMessageBox.information(self, "Burst-wise FCS",
                                             "No correlation curves were produced.")

    def to_settings_dict(self) -> Dict[str, Any]:
        return self._model.to_settings().to_dict()

    # ------------------------------------------------------------------
    # Browser
    # ------------------------------------------------------------------
    def _refresh_browser_list(self) -> None:
        flt = self.line_filter.text().strip().lower()
        self.list_browser.clear()
        self._browser_index: List[int] = []
        for idx, c in enumerate(self._curves):
            label = f"{c.get('file', '')} · b{c.get('burst_index', 0)} · {c.get('pair_name', '')}"
            if flt and flt not in label.lower():
                continue
            self.list_browser.addItem(label)
            self._browser_index.append(idx)

    def _on_browser_selected(self, row: int) -> None:
        if row < 0 or row >= len(getattr(self, "_browser_index", [])):
            self._model._selected = None
        else:
            self._model._selected = self._curves[self._browser_index[row]]
        try:
            self._plots_form.refresh_plots()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settings JSON (load / save / show pairs)
    # ------------------------------------------------------------------
    def _on_save_settings(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save burst-FCS settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            pathlib.Path(path).write_text(json.dumps(self.to_settings_dict(), indent=2))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save settings", str(e))

    def _on_load_settings(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load burst-FCS settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            cfg = json.loads(pathlib.Path(path).read_text())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load settings", str(e))
            return
        for k in ("n_bins", "n_casc", "make_fine", "padding_ms", "fit_mode",
                  "maxent_td_min", "maxent_td_max", "tmin_fit", "tmax_fit"):
            if k in cfg:
                setattr(self._model, k, cfg[k])
        if "maxent_reg" in cfg and float(cfg["maxent_reg"]) > 0:
            self._model.maxent_log10_reg = float(np.log10(float(cfg["maxent_reg"])))
        try:
            self._settings_form.rebuild()
        except Exception:
            pass

    def _on_show_pairs_json(self) -> None:
        pairs = [{"pair_name": p.pair_name, "chs_a": p.chs_a, "chs_b": p.chs_b}
                 for p in self._selected_pairs()]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("🔬 FCS pairs JSON")
        v = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QPlainTextEdit(json.dumps(pairs, indent=2))
        edit.setReadOnly(True)
        v.addWidget(edit)
        dlg.resize(420, 360)
        dlg.exec_()
