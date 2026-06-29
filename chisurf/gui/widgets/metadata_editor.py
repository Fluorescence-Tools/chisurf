from __future__ import annotations

from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from chisurf.core.fio.mmcif.db.pdbx_metadata import get_pdbx_metadata_keys, get_pdbx_metadata_descriptions


# ---------------------------------------------------------------------------
# Shared metadata key definitions (previously duplicated in fitinfo.py and
# burst_selection/gui/tool.py)
# ---------------------------------------------------------------------------

COMMON_METADATA_KEYS = [
    "pH", "temperature", "ionic_strength", "buffer_composition",
    "solvent_phase", "labeling_efficiency", "donor_only_fraction",
    "acceptor_only_fraction", "dye_ratio", "quencher_concentration",
    "time_resolution", "excitation_wavelength", "emission_wavelength",
    "power", "temperature_control", "data_notes",
    "_exptl_crystal_grow.ph",
    "_exptl_crystal_grow.temp",
    "_exptl_crystal_grow.method",
    "_exptl_crystal_grow.comp_details",
    "_diffrn_radiation_wavelength.wavelength",
    "_diffrn_radiation.monochromator",
    "_diffrn_detector.detector",
    "_diffrn_detector.type",
    "_diffrn_standards.number",
    "_diffrn_standards.interval_count",
    "pdbx.sample_type",
    "pdbihm.entry_id",
    "flrcif.sample_class",
    "flrcif.experiment_type",
    "flrcif.data_type",
]

try:
    _PDBX_KEYS = get_pdbx_metadata_keys()
except Exception:
    _PDBX_KEYS = []
ALL_METADATA_KEYS = COMMON_METADATA_KEYS + [k for k in _PDBX_KEYS if k not in COMMON_METADATA_KEYS]

try:
    _PDBX_DESCRIPTIONS = get_pdbx_metadata_descriptions()
except Exception:
    _PDBX_DESCRIPTIONS = {}

_COMMON_DESCRIPTIONS: Dict[str, str] = {
    "pH": "Solution pH",
    "temperature": "Temperature in Kelvin",
    "ionic_strength": "Ionic strength (mM or M)",
    "buffer_composition": "Buffer composition and concentration",
    "solvent_phase": "Solvent phase (liquid, solid, gas)",
    "labeling_efficiency": "Fraction of labeled molecules",
    "donor_only_fraction": "Fraction of donor-only molecules",
    "acceptor_only_fraction": "Fraction of acceptor-only molecules",
    "dye_ratio": "Dye stoichiometry ratio",
    "quencher_concentration": "Quencher concentration",
    "time_resolution": "Time resolution of the measurement",
    "excitation_wavelength": "Excitation wavelength in nm",
    "emission_wavelength": "Emission wavelength in nm",
    "power": "Excitation power",
    "temperature_control": "Temperature control method",
    "data_notes": "Free-form data notes",
    "pdbx.sample_type": "PDBx sample type",
    "pdbihm.entry_id": "PDB-IHM entry identifier",
    "flrcif.sample_class": "FLR-CIF sample class",
    "flrcif.experiment_type": "FLR-CIF experiment type",
    "flrcif.data_type": "FLR-CIF data type",
}


def key_description(key: str) -> str:
    desc = _COMMON_DESCRIPTIONS.get(key)
    if desc:
        return desc
    return _PDBX_DESCRIPTIONS.get(key, "")


# ---------------------------------------------------------------------------
# Tooltip delegate for combobox dropdown items
# ---------------------------------------------------------------------------

class TooltipDelegate(QtWidgets.QStyledItemDelegate):
    """Displays tooltips from Qt.UserRole + 1 data on hover."""

    def helpEvent(self, event, view, option, index):
        tip = index.data(Qt.UserRole + 1)
        if tip:
            QtWidgets.QToolTip.showText(event.globalPos(), tip, view)
            return True
        return super().helpEvent(event, view, option, index)


class MetadataKeyComboBox(QtWidgets.QComboBox):
    """Editable combobox with dropdown item tooltips."""

    def showEvent(self, event):
        super().showEvent(event)
        view = self.view()
        if view is not None:
            view.setMouseTracking(True)
            view.setItemDelegate(TooltipDelegate(view))

    def event(self, event):
        if event.type() == QtCore.QEvent.ToolTip:
            view = self.view()
            if view is not None and view.isVisible():
                index = view.indexAt(view.mapFromGlobal(event.globalPos()))
                if index.isValid():
                    tip = index.data(Qt.UserRole + 1)
                    if tip:
                        QtWidgets.QToolTip.showText(event.globalPos(), tip, view)
                        return True
        return super().event(event)


# ---------------------------------------------------------------------------
# Reusable metadata editor widget
# ---------------------------------------------------------------------------

class MetadataEditor(QtWidgets.QWidget):
    """A reusable widget for editing key-value (optionally +details)
    metadata rows.

    Parameters
    ----------
    columns : int
        Number of columns: 2 (key, value) or 3 (key, value, details).
    parent : QWidget or None

    Signals
    -------
    changed()
        Emitted whenever the user edits a cell or adds/deletes a row.
    """

    changed = QtCore.Signal()

    def __init__(self, columns: int = 2, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        if columns not in (2, 3):
            raise ValueError("columns must be 2 or 3")
        self._columns = columns
        self._suppress_change = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        headers = ["key", "value", "details"][:columns]
        self.table = QtWidgets.QTableWidget(0, columns)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        if columns > 1:
            self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 360)
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.table.setMinimumSize(0, 0)
        self.table.setMaximumSize(16777215, 16777215)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(2)
        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedWidth(24)
        add_btn.setToolTip("Add metadata row")
        add_btn.clicked.connect(self._on_add_empty_row)
        delete_btn = QtWidgets.QPushButton("−")
        delete_btn.setFixedWidth(24)
        delete_btn.setToolTip("Delete selected row")
        delete_btn.clicked.connect(self._on_delete_row)
        buttons.addWidget(add_btn)
        buttons.addWidget(delete_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

    # ── Public API ──────────────────────────────────────────────────

    def get_data(self) -> List[Dict[str, str]]:
        """Return metadata rows as a list of dicts.

        Each dict has keys ``"key"`` and ``"value"`` (and optionally
        ``"details"`` for 3-column mode).  Rows with empty keys are
        skipped.
        """
        rows = []
        for r in range(self.table.rowCount()):
            key = self._row_key(r)
            if not key:
                continue
            item = {
                "key": key,
                "value": self._row_value(r),
            }
            if self._columns > 2:
                item["details"] = self._row_details(r)
            rows.append(item)
        return rows

    def set_data(self, data: List[Dict[str, str]]) -> None:
        """Replace all rows with metadata from *data*."""
        # Collect keys from loaded records that aren't in ALL_METADATA_KEYS so
        # they still appear as autocomplete options in the per-row comboboxes.
        extra: List[str] = []
        for item in data:
            k = item.get("key", "")
            if k and k not in ALL_METADATA_KEYS and k not in extra:
                extra.append(k)
        self._extra_keys: List[str] = extra

        self._suppress_change = True
        self.table.setRowCount(0)
        for item in data:
            key = item.get("key", "")
            value = item.get("value", "")
            details = item.get("details", "") if self._columns > 2 else ""
            self._add_row(key=key, value=value, details=details)
        self._suppress_change = False

    def as_dict(self) -> Dict[str, str]:
        """Return metadata as a flat key->value dict (2-column mode)."""
        return {d["key"]: d["value"] for d in self.get_data()}

    def clear(self) -> None:
        """Remove all rows."""
        self._suppress_change = True
        self.table.setRowCount(0)
        self._suppress_change = False

    # ── Internal helpers ────────────────────────────────────────────

    def _row_key(self, row: int) -> str:
        widget = self.table.cellWidget(row, 0)
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText().strip()
        item = self.table.item(row, 0)
        return item.text().strip() if item is not None else ""

    def _row_value(self, row: int) -> str:
        item = self.table.item(row, 1)
        return item.text().strip() if item is not None else ""

    def _row_details(self, row: int) -> str:
        if self._columns < 3:
            return ""
        item = self.table.item(row, 2)
        return item.text().strip() if item is not None else ""

    def _make_key_combo(self, row: int) -> MetadataKeyComboBox:
        """Create a populated MetadataKeyComboBox for a table row."""
        combo = MetadataKeyComboBox()
        combo.setEditable(True)
        all_keys = ALL_METADATA_KEYS + getattr(self, "_extra_keys", [])
        combo.addItems(all_keys)
        for idx, key in enumerate(ALL_METADATA_KEYS):
            combo.setItemData(idx, key_description(key), Qt.UserRole + 1)
        comp = combo.completer()
        if comp is not None:
            comp.setFilterMode(Qt.MatchContains)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
        combo.currentTextChanged.connect(lambda text, r=row: self._update_row_tooltip(r, text))
        combo.currentTextChanged.connect(self._emit_changed)
        return combo

    def _update_row_tooltip(self, row: int, key: str):
        desc = key_description(key)
        combo = self.table.cellWidget(row, 0)
        if isinstance(combo, QtWidgets.QComboBox):
            combo.setToolTip(desc)
        for col in range(1, self._columns):
            item = self.table.item(row, col)
            if item is not None:
                item.setToolTip(desc)

    def _add_row(self, key: str = "", value: str = "", details: str = ""):
        row = self.table.rowCount()
        self.table.insertRow(row)

        combo = self._make_key_combo(row)
        self.table.setCellWidget(row, 0, combo)
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
        if self._columns > 2:
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(details))

        if key:
            combo.blockSignals(True)
            combo.setCurrentText(key)
            combo.blockSignals(False)
            self._update_row_tooltip(row, key)
        else:
            combo.setCurrentIndex(-1)

    def _on_add_empty_row(self):
        self._suppress_change = True
        self._add_row()
        self._suppress_change = False

    def _on_delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _on_item_changed(self):
        self._emit_changed()

    def _emit_changed(self):
        if not self._suppress_change:
            self.changed.emit()
