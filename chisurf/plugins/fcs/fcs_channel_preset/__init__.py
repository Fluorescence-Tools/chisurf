"""FCS channel definition plugin.

This GUI plugin defines logical FCS correlation channel pairs per
*detector setup*. Detector setups (windows/detectors/TTTR reading) are
managed by :mod:`chisurf.gui.widgets.wizard.tttr_channel_definition` and
stored in ``detector_setups.json``.  This plugin adds a small JSON file
in the user settings directory::

    ~/.chisurf/fcs_channel_setups.json

The JSON structure is::

    {
        "version": 1,
        "setups": {
            "<setup_name>": {
                "correlator": {"n_bins": int, "n_casc": int, "make_fine": bool},
                "pairs": [
                    {"name": str, "channel_a": str, "channel_b": str, "kind": str | null},
                    ...
                ],
            },
            ...
        },
        "last_used_setup": "<setup_name>" | null
    }

The *channel_a* and *channel_b* names correspond to the logical channel
keys constructed from a detector setup, e.g. ``"prompt_green"`` or
``"delayed_red"``. They are derived from the PIE windows and detectors.

The burst-wise diffusion plugin reads this JSON file to know which
channel pairs to correlate and which basic correlator settings to use.
"""

from __future__ import annotations

from typing import Any, Dict, List

from qtpy import QtWidgets, QtCore

from chisurf.settings import cs_settings
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups, JsonEditorDialog
from chisurf.fluorescence.fcs.channel_setups import (
    FCS_CHANNEL_SETUPS_FILE,
    load_fcs_channel_setups,
    save_fcs_channel_setups,
    build_channels_from_setup,
)


# Plugin category/name for the ChiSurf menu
name = "Setup:FCS Channel Definitions"


class FCSChannelDialog(QtWidgets.QDialog):
    """Minimal editor for FCS channel-pair definitions per detector setup."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FCS Channel Definitions")
        self.resize(720, 460)

        self._detector_setups: Dict[str, Any] = {}
        self._fcs_cfg: Dict[str, Any] = {}
        self._channels_for_setup: Dict[str, List[Dict[str, Any]]] = {}
        self._current_setup: str | None = None
        self._channel_names: List[str] = []

        self._build_ui()
        self._load_state()

    # ---- UI ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Detector setup:", self))
        self.setup_combo = QtWidgets.QComboBox(self)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        top.addWidget(self.setup_combo, 1)
        self.btn_reload = QtWidgets.QPushButton("Reload", self)
        self.btn_reload.clicked.connect(self._reload_detector_setups)
        top.addWidget(self.btn_reload)
        layout.addLayout(top)

        info = QtWidgets.QLabel(
            f"FCS channel definitions are stored in\n  {FCS_CHANNEL_SETUPS_FILE}",
            self,
        )
        info.setWordWrap(True)
        info_row = QtWidgets.QHBoxLayout()
        info_row.addWidget(info, 1)
        self.btn_edit_json = QtWidgets.QToolButton(self)
        self.btn_edit_json.setText("Edit JSON")
        self.btn_edit_json.setToolTip("Edit the raw fcs_channel_setups.json file")
        self.btn_edit_json.clicked.connect(self._on_edit_json)
        info_row.addWidget(self.btn_edit_json)
        layout.addLayout(info_row)

        # Correlator settings
        corr_box = QtWidgets.QGroupBox("Correlator settings (tttrlib.Correlator)", self)
        g = QtWidgets.QGridLayout(corr_box)
        self.spin_bins = QtWidgets.QSpinBox(corr_box)
        self.spin_bins.setRange(1, 512)
        self.spin_bins.setValue(int(cs_settings["correlator"]["B"]))
        self.spin_casc = QtWidgets.QSpinBox(corr_box)
        self.spin_casc.setRange(1, 64)
        self.spin_casc.setValue(int(cs_settings["correlator"]["number_of_cascades"]))
        self.check_fine = QtWidgets.QCheckBox("Fine correlation (use microtimes)", corr_box)
        self.check_fine.setChecked(bool(cs_settings["correlator"]["fine"]))

        r = 0
        g.addWidget(QtWidgets.QLabel("Bins per cascade (B)", corr_box), r, 0)
        g.addWidget(self.spin_bins, r, 1)
        r += 1
        g.addWidget(QtWidgets.QLabel("Number of cascades", corr_box), r, 0)
        g.addWidget(self.spin_casc, r, 1)
        r += 1
        g.addWidget(self.check_fine, r, 0, 1, 2)
        layout.addWidget(corr_box)

        # Channel pairs table
        mid = QtWidgets.QHBoxLayout()

        right_box = QtWidgets.QGroupBox("Channel pairs", self)
        right_layout = QtWidgets.QVBoxLayout(right_box)
        self.table_pairs = QtWidgets.QTableWidget(right_box)
        self.table_pairs.setColumnCount(6)
        self.table_pairs.setHorizontalHeaderLabels([
            "Name",
            "Channel A",
            "Channel B",
            "Bins",
            "Cascades",
            "Fine",
        ])
        self.table_pairs.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        header = self.table_pairs.horizontalHeader()
        try:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            for col in range(1, 6):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        except Exception:
            pass
        self.table_pairs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_pairs.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Allow direct in-place editing of all cells (channels, name, and correlator params)
        self.table_pairs.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.table_pairs.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table_pairs.customContextMenuRequested.connect(self._on_pairs_context_menu)

        # Controls for adding pairs (positioned above the table)
        controls = QtWidgets.QHBoxLayout()
        self.combo_a = QtWidgets.QComboBox(right_box)
        self.combo_b = QtWidgets.QComboBox(right_box)
        self.edit_name = QtWidgets.QLineEdit(right_box)
        self.edit_name.setPlaceholderText("Pair label (optional)")
        self.btn_add = QtWidgets.QPushButton("Add", right_box)

        controls.addWidget(QtWidgets.QLabel("A:", right_box))
        controls.addWidget(self.combo_a)
        controls.addWidget(QtWidgets.QLabel("B:", right_box))
        controls.addWidget(self.combo_b)
        controls.addWidget(self.edit_name, 1)
        controls.addWidget(self.btn_add)
        right_layout.addLayout(controls)
        right_layout.addWidget(self.table_pairs, 1)

        self.btn_add.clicked.connect(self._on_add_pair)

        mid.addWidget(right_box, 1)
        layout.addLayout(mid, 1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.btn_save = QtWidgets.QPushButton("Save", self)
        self.btn_close = QtWidgets.QPushButton("Close", self)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_close.clicked.connect(self.close)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

    # ---- Loading / saving -------------------------------------------
    def _load_state(self) -> None:
        self._reload_detector_setups()
        self._fcs_cfg = load_fcs_channel_setups()
        last = self._fcs_cfg.get("last_used_setup")
        if isinstance(last, str) and last:
            idx = self.setup_combo.findText(last)
            if idx >= 0:
                self.setup_combo.setCurrentIndex(idx)

    def _reload_detector_setups(self) -> None:
        data = load_detector_setups()
        setups = data.get("setups", {}) if isinstance(data, dict) else {}
        self._detector_setups = setups

        self.setup_combo.blockSignals(True)
        self.setup_combo.clear()
        for name in sorted(setups.keys()):
            self.setup_combo.addItem(name)
        self.setup_combo.blockSignals(False)

        if self.setup_combo.count() > 0:
            self._on_setup_changed(0)

    # ---- Helpers ----------------------------------------------------
    def _current_setup_dict(self) -> Dict[str, Any] | None:
        if not self._current_setup:
            return None
        return self._detector_setups.get(self._current_setup)

    def _populate_channels(self) -> None:
        self.combo_a.clear()
        self.combo_b.clear()
        self._channels_for_setup = {}

        sd = self._current_setup_dict()
        if not isinstance(sd, dict):
            return
        wins = sd.get("windows", {}) or {}
        dets = sd.get("detectors", {}) or {}
        self._channels_for_setup = build_channels_from_setup(wins, dets)
        names = sorted(self._channels_for_setup.keys())
        self._channel_names = names
        self.combo_a.addItems(names)
        self.combo_b.addItems(names)

    def _populate_pairs(self) -> None:
        self.table_pairs.setRowCount(0)
        if not self._current_setup:
            return
        cfg_all = self._fcs_cfg.get("setups", {}) if isinstance(self._fcs_cfg, dict) else {}
        cfg = cfg_all.get(self._current_setup, {}) if isinstance(cfg_all, dict) else {}

        corr = cfg.get("correlator", {}) if isinstance(cfg, dict) else {}
        try:
            self.spin_bins.setValue(int(corr.get("n_bins", cs_settings["correlator"]["B"])))
        except Exception:
            pass
        try:
            self.spin_casc.setValue(int(corr.get("n_casc", cs_settings["correlator"]["number_of_cascades"])))
        except Exception:
            pass
        try:
            self.check_fine.setChecked(bool(corr.get("make_fine", cs_settings["correlator"]["fine"])))
        except Exception:
            pass

        pairs = cfg.get("pairs", []) if isinstance(cfg, dict) else []
        if not isinstance(pairs, list):
            pairs = []
        for p in pairs:
            try:
                name = str(p.get("name", ""))
                cha = str(p.get("channel_a", ""))
                chb = str(p.get("channel_b", ""))
                pcorr = p.get("correlator", {}) if isinstance(p, dict) else {}
            except Exception:
                continue
            row = self.table_pairs.rowCount()
            self.table_pairs.insertRow(row)
            self.table_pairs.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            # Channel A/B as combo boxes populated with available channels
            cb_names = list(self._channel_names)
            if cha and cha not in cb_names:
                cb_names.append(cha)
            if chb and chb not in cb_names:
                cb_names.append(chb)
            combo_a = QtWidgets.QComboBox(self.table_pairs)
            combo_b = QtWidgets.QComboBox(self.table_pairs)
            combo_a.addItems(cb_names)
            combo_b.addItems(cb_names)
            idx_a = combo_a.findText(cha)
            if idx_a >= 0:
                combo_a.setCurrentIndex(idx_a)
            idx_b = combo_b.findText(chb)
            if idx_b >= 0:
                combo_b.setCurrentIndex(idx_b)
            self.table_pairs.setCellWidget(row, 1, combo_a)
            self.table_pairs.setCellWidget(row, 2, combo_b)
            b_txt = ""
            nc_txt = ""
            fine_txt = ""
            try:
                if isinstance(pcorr, dict) and "n_bins" in pcorr:
                    b_txt = str(int(pcorr.get("n_bins")))
            except Exception:
                b_txt = ""
            try:
                if isinstance(pcorr, dict) and "n_casc" in pcorr:
                    nc_txt = str(int(pcorr.get("n_casc")))
            except Exception:
                nc_txt = ""
            try:
                if isinstance(pcorr, dict) and "make_fine" in pcorr:
                    fine_txt = "1" if bool(pcorr.get("make_fine")) else "0"
            except Exception:
                fine_txt = ""
            self.table_pairs.setItem(row, 3, QtWidgets.QTableWidgetItem(b_txt))
            self.table_pairs.setItem(row, 4, QtWidgets.QTableWidgetItem(nc_txt))
            # Fine as checkbox
            chk = QtWidgets.QCheckBox(self.table_pairs)
            try:
                chk.setChecked(fine_txt in ("1", "true", "t", "yes", "y"))
            except Exception:
                chk.setChecked(False)
            chk.setTristate(False)
            self.table_pairs.setCellWidget(row, 5, chk)

    def _collect_pairs_cfg(self) -> Dict[str, Any]:
        corr = {
            "n_bins": int(self.spin_bins.value()),
            "n_casc": int(self.spin_casc.value()),
            "make_fine": bool(self.check_fine.isChecked()),
        }
        pairs: List[Dict[str, Any]] = []
        n_rows = self.table_pairs.rowCount()
        for r in range(n_rows):
            # Channel A/B may be combo boxes
            w_a = self.table_pairs.cellWidget(r, 1)
            w_b = self.table_pairs.cellWidget(r, 2)
            if isinstance(w_a, QtWidgets.QComboBox):
                cha = w_a.currentText().strip()
            else:
                a_item = self.table_pairs.item(r, 1)
                if a_item is None:
                    continue
                cha = a_item.text().strip()
            if isinstance(w_b, QtWidgets.QComboBox):
                chb = w_b.currentText().strip()
            else:
                b_item = self.table_pairs.item(r, 2)
                if b_item is None:
                    continue
                chb = b_item.text().strip()
            if not cha or not chb:
                continue
            nm_item = self.table_pairs.item(r, 0)
            name = nm_item.text().strip() if nm_item is not None else ""
            if not name:
                name = f"{cha}×{chb}" if cha != chb else f"{cha}_ACF"
            # Derive kind implicitly from channels (no explicit column in the UI)
            kind = "ACF" if cha == chb else "CCF"
            b_item = self.table_pairs.item(r, 3)
            nc_item = self.table_pairs.item(r, 4)
            fine_widget = self.table_pairs.cellWidget(r, 5)
            pair: Dict[str, Any] = {
                "name": name,
                "channel_a": cha,
                "channel_b": chb,
                "kind": kind,
            }
            n_bins_val = None
            n_casc_val = None
            fine_val = None
            try:
                if b_item is not None:
                    txt = b_item.text().strip()
                    if txt:
                        n_bins_val = int(txt)
            except Exception:
                n_bins_val = None
            try:
                if nc_item is not None:
                    txt = nc_item.text().strip()
                    if txt:
                        n_casc_val = int(txt)
            except Exception:
                n_casc_val = None
            try:
                if isinstance(fine_widget, QtWidgets.QCheckBox):
                    fine_val = bool(fine_widget.isChecked())
                else:
                    fine_item = self.table_pairs.item(r, 5)
                    if fine_item is not None:
                        txt = fine_item.text().strip().lower()
                        if txt:
                            fine_val = txt in ("1", "true", "t", "yes", "y")
            except Exception:
                fine_val = None
            if n_bins_val is not None or n_casc_val is not None or fine_val is not None:
                pcorr: Dict[str, Any] = {}
                if n_bins_val is not None:
                    pcorr["n_bins"] = n_bins_val
                if n_casc_val is not None:
                    pcorr["n_casc"] = n_casc_val
                if fine_val is not None:
                    pcorr["make_fine"] = fine_val
                pair["correlator"] = pcorr
            pairs.append(pair)
        return {"correlator": corr, "pairs": pairs}

    # ---- Slots ------------------------------------------------------
    def _on_setup_changed(self, _idx: int) -> None:
        self._current_setup = self.setup_combo.currentText().strip() or None
        self._populate_channels()
        self._populate_pairs()

    def _on_add_pair(self) -> None:
        cha = self.combo_a.currentText().strip()
        chb = self.combo_b.currentText().strip()
        if not cha or not chb:
            return
        name = self.edit_name.text().strip()
        if not name:
            name = f"{cha}×{chb}" if cha != chb else f"{cha}_ACF"
        row = self.table_pairs.rowCount()
        self.table_pairs.insertRow(row)
        self.table_pairs.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        # Channel A/B as combo boxes mirroring the current channel list
        cb_names = list(self._channel_names)
        if cha and cha not in cb_names:
            cb_names.append(cha)
        if chb and chb not in cb_names:
            cb_names.append(chb)
        combo_a = QtWidgets.QComboBox(self.table_pairs)
        combo_b = QtWidgets.QComboBox(self.table_pairs)
        combo_a.addItems(cb_names)
        combo_b.addItems(cb_names)
        idx_a = combo_a.findText(cha)
        if idx_a >= 0:
            combo_a.setCurrentIndex(idx_a)
        idx_b = combo_b.findText(chb)
        if idx_b >= 0:
            combo_b.setCurrentIndex(idx_b)
        self.table_pairs.setCellWidget(row, 1, combo_a)
        self.table_pairs.setCellWidget(row, 2, combo_b)
        # Initialize per-pair correlator settings from the current global settings
        self.table_pairs.setItem(row, 3, QtWidgets.QTableWidgetItem(str(int(self.spin_bins.value()))))
        self.table_pairs.setItem(row, 4, QtWidgets.QTableWidgetItem(str(int(self.spin_casc.value()))))
        chk = QtWidgets.QCheckBox(self.table_pairs)
        chk.setChecked(self.check_fine.isChecked())
        chk.setTristate(False)
        self.table_pairs.setCellWidget(row, 5, chk)

    def _on_pairs_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self.table_pairs)
        act_remove = menu.addAction("Remove selected")
        action = menu.exec_(self.table_pairs.viewport().mapToGlobal(pos))
        if action == act_remove:
            self._on_remove_pairs()

    def _on_remove_pairs(self) -> None:
        sel = self.table_pairs.selectionModel()
        if sel is None:
            return
        rows = sorted({i.row() for i in sel.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table_pairs.removeRow(r)

    def _on_edit_json(self) -> None:
        """Open a simple JSON editor for the FCS channel setups file."""
        cfg = load_fcs_channel_setups()
        dlg = JsonEditorDialog(cfg, self)
        if dlg.exec_():
            edited = dlg.get_edited_data()
            if isinstance(edited, dict):
                ok = save_fcs_channel_setups(edited)
                if ok:
                    self._fcs_cfg = edited
                    # Refresh current setup view (pairs/correlator settings)
                    self._populate_pairs()
                else:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error",
                        f"Could not save to:\n{FCS_CHANNEL_SETUPS_FILE}",
                    )

    def _on_save(self) -> None:
        if not self._current_setup:
            QtWidgets.QMessageBox.warning(self, "No setup selected", "Select a detector setup first.")
            return
        cfg_all = load_fcs_channel_setups()
        setups = cfg_all.get("setups")
        if not isinstance(setups, dict):
            setups = {}
            cfg_all["setups"] = setups
        setups[self._current_setup] = self._collect_pairs_cfg()
        cfg_all["last_used_setup"] = self._current_setup
        ok = save_fcs_channel_setups(cfg_all)
        if ok:
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Saved FCS channel pairs for setup '{self._current_setup}' to:\n{FCS_CHANNEL_SETUPS_FILE}",
            )
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Could not save to:\n{FCS_CHANNEL_SETUPS_FILE}",
            )


# ---- Plugin entry ---------------------------------------------------
if __name__ == "plugin":  # pragma: no cover
    app = QtWidgets.QApplication.instance()
    parent = None if app is None else app.activeWindow()
    dlg = FCSChannelDialog(parent)
    dlg.setWindowModality(QtCore.Qt.NonModal)
    dlg.show()
