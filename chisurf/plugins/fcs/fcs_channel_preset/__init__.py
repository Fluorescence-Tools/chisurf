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

from chisurf.core.settings import cs_settings
from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
    resolve_active_user_id,
)
from chisurf.core.fluorescence.fcs.channel_setups import (
    load_fcs_channel_setups,
    save_fcs_channel_setups,
    build_channels_from_setup,
)

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


# Plugin category/name for the ChiSurf menu
name = "Setup:FCS Definitions"
icon = "📡"


@persist_plugin_state("fcs_channel_preset")
class FCSChannelDialog(QtWidgets.QDialog):
    """Minimal editor for FCS channel-pair definitions per detector setup."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, db_path: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FCS Channel Definitions")
        self.resize(720, 460)
        self._db_path = db_path

        self._detector_setups: Dict[str, Any] = {}
        self._fcs_cfg: Dict[str, Any] = {}
        self._channels_for_setup: Dict[str, List[Dict[str, Any]]] = {}
        self._current_setup: str | None = None
        self._channel_names: List[str] = []
        self._public_checkbox = QtWidgets.QCheckBox("Public")
        self._public_checkbox.setChecked(False)
        self._public_checkbox.setToolTip(
            "When checked, this setup is visible to all users in "
            "the MFDB. Only the owner can change this setting."
        )
        self._public_checkbox.setEnabled(False)

        self._build_ui()
        self._load_state()

    # ---- UI ---------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Detector setup:", self))
        self.setup_combo = QtWidgets.QComboBox(self)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        top.addWidget(self.setup_combo, 1)
        self.btn_reload = QtWidgets.QToolButton(self)
        self.btn_reload.setText("🔄 Reload")
        self.btn_reload.clicked.connect(self._reload_detector_setups)
        top.addWidget(self.btn_reload)
        top.addWidget(self._public_checkbox)
        layout.addLayout(top)

        # Channel pairs table
        mid = QtWidgets.QHBoxLayout()

        right_box = QtWidgets.QGroupBox("Channel pairs", self)
        right_layout = QtWidgets.QVBoxLayout(right_box)
        self.table_pairs = QtWidgets.QTableWidget(right_box)
        self.table_pairs.setColumnCount(7)
        self.table_pairs.setHorizontalHeaderLabels([
            "Name",
            "Channel A",
            "Channel B",
            "Bins",
            "Cascades",
            "Fine",
            "",
        ])
        self.table_pairs.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        header = self.table_pairs.horizontalHeader()
        try:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            for col in range(1, 6):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(6, QtWidgets.QHeaderView.Fixed)
            header.resizeSection(6, 30)
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
        self.btn_add = QtWidgets.QToolButton(right_box)
        self.btn_add.setText("➕ Add")

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
        self.btn_save = QtWidgets.QToolButton(self)
        self.btn_save.setText("💾 Save")
        self.btn_close = QtWidgets.QToolButton(self)
        self.btn_close.setText("❌ Close")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_close.clicked.connect(self.close)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

    # ---- Loading / saving -------------------------------------------
    def _load_state(self) -> None:
        self._reload_detector_setups()
        self._fcs_cfg = load_fcs_channel_setups(db_path=self._db_path, skip_migration=True)
        last = self._fcs_cfg.get("last_used_setup")
        if isinstance(last, str) and last:
            idx = self.setup_combo.findText(last)
            if idx >= 0:
                self.setup_combo.setCurrentIndex(idx)

    def _reload_detector_setups(self) -> None:
        data = load_detector_setups(db_path=self._db_path, skip_migration=True)
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

        pairs = cfg.get("pairs", []) if isinstance(cfg, dict) else []
        if not isinstance(pairs, list):
            pairs = []
        for p in pairs:
            try:
                name = str(p.get("name", ""))
                cha = str(p.get("channel_a", ""))
                chb = str(p.get("channel_b", ""))
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
            # Per-pair correlator values from structured columns
            b_val = p.get("n_bins")
            nc_val = p.get("n_casc")
            fine_val = p.get("make_fine")
            self.table_pairs.setItem(
                row, 3,
                QtWidgets.QTableWidgetItem(str(b_val) if b_val is not None else ""),
            )
            self.table_pairs.setItem(
                row, 4,
                QtWidgets.QTableWidgetItem(str(nc_val) if nc_val is not None else ""),
            )
            chk = QtWidgets.QCheckBox(self.table_pairs)
            if fine_val is not None:
                chk.setChecked(bool(fine_val))
            chk.setTristate(False)
            self.table_pairs.setCellWidget(row, 5, chk)
            # Delete button per row
            btn_del = QtWidgets.QToolButton(self.table_pairs)
            btn_del.setText("✕")
            btn_del.setToolTip("Remove this pair")
            btn_del.clicked.connect(lambda checked, r=row: self._on_remove_row(r))
            self.table_pairs.setCellWidget(row, 6, btn_del)

    def _collect_pairs_cfg(self) -> Dict[str, Any]:
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
            if n_bins_val is not None:
                pair["n_bins"] = n_bins_val
            if n_casc_val is not None:
                pair["n_casc"] = n_casc_val
            if fine_val is not None:
                pair["make_fine"] = fine_val
            pairs.append(pair)
        return {"pairs": pairs}

    # ---- Slots ------------------------------------------------------
    def _on_setup_changed(self, _idx: int) -> None:
        self._current_setup = self.setup_combo.currentText().strip() or None
        self._populate_channels()
        self._populate_pairs()
        # Update public checkbox from loaded setup metadata
        cfg_all = self._fcs_cfg.get("setups", {}) if isinstance(self._fcs_cfg, dict) else {}
        sd = cfg_all.get(self._current_setup, {}) if self._current_setup else {}
        if isinstance(sd, dict):
            is_pub = bool(sd.get("_is_public", False))
            owner = sd.get("_owner")
            active = resolve_active_user_id()
            can_edit = (owner is None) or (owner == active)
            self._public_checkbox.setChecked(is_pub)
            self._public_checkbox.setEnabled(can_edit)

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
        # Seed per-pair correlator from cs_settings defaults
        try:
            _def_bins = int(cs_settings["correlator"]["B"])
        except Exception:
            _def_bins = 2
        try:
            _def_casc = int(cs_settings["correlator"]["number_of_cascades"])
        except Exception:
            _def_casc = 25
        try:
            _def_fine = bool(cs_settings["correlator"]["fine"])
        except Exception:
            _def_fine = True
        self.table_pairs.setItem(row, 3, QtWidgets.QTableWidgetItem(str(_def_bins)))
        self.table_pairs.setItem(row, 4, QtWidgets.QTableWidgetItem(str(_def_casc)))
        chk = QtWidgets.QCheckBox(self.table_pairs)
        chk.setChecked(_def_fine)
        chk.setTristate(False)
        self.table_pairs.setCellWidget(row, 5, chk)
        # Delete button
        btn_del = QtWidgets.QToolButton(self.table_pairs)
        btn_del.setText("✕")
        btn_del.setToolTip("Remove this pair")
        btn_del.clicked.connect(lambda checked, r=row: self._on_remove_row(r))
        self.table_pairs.setCellWidget(row, 6, btn_del)

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

    def _on_remove_row(self, row: int) -> None:
        self.table_pairs.removeRow(row)

    def _on_save(self) -> None:
        if not self._current_setup:
            QtWidgets.QMessageBox.warning(self, "No setup selected", "Select a detector setup first.")
            return
        cfg_all = load_fcs_channel_setups()
        setups = cfg_all.get("setups")
        if not isinstance(setups, dict):
            setups = {}
            cfg_all["setups"] = setups
        setup_data = self._collect_pairs_cfg()
        setup_data["_is_public"] = self._public_checkbox.isChecked()
        setups[self._current_setup] = setup_data
        cfg_all["last_used_setup"] = self._current_setup
        ok = save_fcs_channel_setups(cfg_all, is_public=self._public_checkbox.isChecked())
        if ok:
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Saved FCS channel pairs for setup '{self._current_setup}'.",
            )


# ---- Plugin entry ---------------------------------------------------
if __name__ == "plugin":  # pragma: no cover
    app = QtWidgets.QApplication.instance()
    parent = None if app is None else app.activeWindow()
    dlg = FCSChannelDialog(parent)
    dlg.setWindowModality(QtCore.Qt.NonModal)
    dlg.show()
