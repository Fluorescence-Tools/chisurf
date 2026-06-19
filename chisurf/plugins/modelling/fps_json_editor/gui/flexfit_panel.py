"""Panel for editing FlexFit residue and bond constraints."""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

_FLEXFIT_KEYS = {"Flexible residues", "Bonds"}


def _int_or(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


class FlexFitPanel(QtWidgets.QWidget):
    """A widget for defining flexible residues and bond constraints for FlexFit simulations.

    Attributes
    ----------
    flexfit_changed : QtCore.Signal
        Emitted when any FlexFit setting (sets, residues, bonds) changes.
    """

    flexfit_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the FlexFitPanel with layout and table widgets."""
        super().__init__(parent)
        self._extra_sections: dict[str, Any] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Set selection row
        set_row = QtWidgets.QHBoxLayout()
        set_row.setSpacing(3)
        set_row.addWidget(QtWidgets.QLabel("Set:"))
        self.flexfit_set_combo = QtWidgets.QComboBox()
        self.flexfit_set_combo.currentIndexChanged[int].connect(self._on_flexfit_set_changed)
        set_row.addWidget(self.flexfit_set_combo, stretch=1)

        self.flexfit_add_set_btn = QtWidgets.QPushButton("+")
        self.flexfit_add_set_btn.setToolTip("Add a new FlexFit set")
        self.flexfit_add_set_btn.setMaximumWidth(30)
        self.flexfit_add_set_btn.clicked.connect(self._on_flexfit_add_set)
        set_row.addWidget(self.flexfit_add_set_btn)

        self.flexfit_remove_set_btn = QtWidgets.QPushButton("-")
        self.flexfit_remove_set_btn.setToolTip("Remove the current FlexFit set")
        self.flexfit_remove_set_btn.setMaximumWidth(30)
        self.flexfit_remove_set_btn.clicked.connect(self._on_flexfit_remove_set)
        set_row.addWidget(self.flexfit_remove_set_btn)
        layout.addLayout(set_row)

        # Flexible residues list table
        layout.addWidget(QtWidgets.QLabel("Flexible residues:"))
        self.flexfit_res_table = QtWidgets.QTableWidget(0, 2)
        self.flexfit_res_table.setHorizontalHeaderLabels(["Chain", "Residue"])
        self.flexfit_res_table.horizontalHeader().setStretchLastSection(True)
        self.flexfit_res_table.setMinimumHeight(80)
        self.flexfit_res_table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.flexfit_res_table)

        res_buttons = QtWidgets.QHBoxLayout()
        res_buttons.setSpacing(3)
        self.flexfit_add_res_btn = QtWidgets.QPushButton("Add residue")
        self.flexfit_add_res_btn.clicked.connect(self._on_flexfit_add_residue)
        self.flexfit_remove_res_btn = QtWidgets.QPushButton("Remove selected")
        self.flexfit_remove_res_btn.clicked.connect(self._on_flexfit_remove_residue)
        res_buttons.addWidget(self.flexfit_add_res_btn)
        res_buttons.addWidget(self.flexfit_remove_res_btn)
        res_buttons.addStretch()
        layout.addLayout(res_buttons)

        # Bonds list table
        layout.addWidget(QtWidgets.QLabel("Bonds:"))
        self.flexfit_bond_table = QtWidgets.QTableWidget(0, 6)
        self.flexfit_bond_table.setHorizontalHeaderLabels([
            "Chain 1", "Residue 1", "Atom 1",
            "Chain 2", "Residue 2", "Atom 2",
        ])
        self.flexfit_bond_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.flexfit_bond_table.setMinimumHeight(80)
        self.flexfit_bond_table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.flexfit_bond_table)

        bond_buttons = QtWidgets.QHBoxLayout()
        bond_buttons.setSpacing(3)
        self.flexfit_add_bond_btn = QtWidgets.QPushButton("Add bond")
        self.flexfit_add_bond_btn.clicked.connect(self._on_flexfit_add_bond)
        self.flexfit_remove_bond_btn = QtWidgets.QPushButton("Remove selected")
        self.flexfit_remove_bond_btn.clicked.connect(self._on_flexfit_remove_bond)
        bond_buttons.addWidget(self.flexfit_add_bond_btn)
        bond_buttons.addWidget(self.flexfit_remove_bond_btn)
        bond_buttons.addStretch()
        layout.addLayout(bond_buttons)

        layout.addStretch()

    def update_flexfit(self, extra_sections: dict[str, Any]) -> None:
        """Repopulate the UI controls from extra_sections.

        Parameters
        ----------
        extra_sections : dict
            Model's extra sections.
        """
        self._extra_sections = extra_sections
        flexfit = self._extra_sections.get("FlexFit", {}) or {}
        if not isinstance(flexfit, dict):
            flexfit = {}

        self.flexfit_set_combo.blockSignals(True)
        old_set = self.flexfit_set_combo.currentText()
        self.flexfit_set_combo.clear()
        for name in flexfit:
            self.flexfit_set_combo.addItem(name)

        idx = self.flexfit_set_combo.findText(old_set)
        if idx < 0 and self.flexfit_set_combo.count() > 0:
            idx = 0
        if idx >= 0:
            self.flexfit_set_combo.setCurrentIndex(idx)
        self.flexfit_set_combo.blockSignals(False)

        enabled = self.flexfit_set_combo.count() > 0
        self.flexfit_remove_set_btn.setEnabled(enabled)
        self._show_flexfit_set()

    def flush_flexfit(self, extra_sections: dict[str, Any]) -> None:
        """Flush current tables back to extra_sections.

        Parameters
        ----------
        extra_sections : dict
            Model's extra sections to write to.
        """
        self._extra_sections = extra_sections
        flexfit = self._extra_sections.get("FlexFit", {}) or {}
        if not isinstance(flexfit, dict):
            flexfit = {}

        active_name = self.flexfit_set_combo.currentText()
        if not active_name:
            return

        residues = []
        for row in range(self.flexfit_res_table.rowCount()):
            c = self.flexfit_res_table.item(row, 0)
            r = self.flexfit_res_table.item(row, 1)
            residues.append({
                "chain_identifier": c.text().strip() if c and c.text() else "",
                "residue_seq_number": _int_or(r.text().strip() if r and r.text() else "0"),
            })

        bonds = []
        for row in range(self.flexfit_bond_table.rowCount()):
            def _cell(col):
                it = self.flexfit_bond_table.item(row, col)
                return it.text().strip() if it and it.text() else ""

            def _end(col_offset):
                return {
                    "chain_identifier": _cell(col_offset),
                    "residue_seq_number": _int_or(_cell(col_offset + 1)),
                    "atom_name": _cell(col_offset + 2),
                }

            bonds.append([_end(0), _end(3)])

        old_entry = flexfit.get(active_name, {})
        if isinstance(old_entry, dict):
            entry = {k: v for k, v in old_entry.items() if k not in _FLEXFIT_KEYS}
        else:
            entry = {}
        entry["Flexible residues"] = residues
        entry["Bonds"] = bonds
        flexfit[active_name] = entry

        self._extra_sections["FlexFit"] = flexfit

    def _show_flexfit_set(self) -> None:
        name = self.flexfit_set_combo.currentText()
        flexfit = self._extra_sections.get("FlexFit", {}) or {}
        entry = flexfit.get(name, {}) if isinstance(flexfit, dict) else {}
        if not isinstance(entry, dict):
            entry = {}

        self.flexfit_res_table.blockSignals(True)
        residues = entry.get("Flexible residues", []) or []
        self.flexfit_res_table.setRowCount(0)
        for res in residues:
            row = self.flexfit_res_table.rowCount()
            self.flexfit_res_table.insertRow(row)
            self.flexfit_res_table.setItem(
                row, 0,
                QtWidgets.QTableWidgetItem(str(res.get("chain_identifier", "")))
            )
            self.flexfit_res_table.setItem(
                row, 1,
                QtWidgets.QTableWidgetItem(str(res.get("residue_seq_number", "")))
            )
        self.flexfit_res_table.blockSignals(False)

        self.flexfit_bond_table.blockSignals(True)
        bonds = entry.get("Bonds", []) or []
        self.flexfit_bond_table.setRowCount(0)
        for bond in bonds:
            if not isinstance(bond, list) or len(bond) < 2:
                continue
            e1, e2 = bond[0], bond[1]
            row = self.flexfit_bond_table.rowCount()
            self.flexfit_bond_table.insertRow(row)
            for col, key in [
                (0, "chain_identifier"), (1, "residue_seq_number"), (2, "atom_name")
            ]:
                val = e1.get(key, "") if isinstance(e1, dict) else ""
                self.flexfit_bond_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))
            for col, key in [
                (3, "chain_identifier"), (4, "residue_seq_number"), (5, "atom_name")
            ]:
                val = e2.get(key, "") if isinstance(e2, dict) else ""
                self.flexfit_bond_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))
        self.flexfit_bond_table.blockSignals(False)

    def _on_flexfit_set_changed(self, index: int) -> None:
        self.flush_flexfit(self._extra_sections)
        self._show_flexfit_set()
        self.flexfit_changed.emit()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        self.flush_flexfit(self._extra_sections)
        self.flexfit_changed.emit()

    def _on_flexfit_add_set(self) -> None:
        self.flush_flexfit(self._extra_sections)
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New FlexFit set", "Set name:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        flexfit = self._extra_sections.setdefault("FlexFit", {})
        if name in flexfit:
            return
        flexfit[name] = {"Flexible residues": [], "Bonds": []}
        self.update_flexfit(self._extra_sections)
        idx = self.flexfit_set_combo.findText(name)
        if idx >= 0:
            self.flexfit_set_combo.setCurrentIndex(idx)
        self.flexfit_changed.emit()

    def _on_flexfit_remove_set(self) -> None:
        name = self.flexfit_set_combo.currentText()
        if not name:
            return
        flexfit = self._extra_sections.get("FlexFit", {})
        if isinstance(flexfit, dict) and name in flexfit:
            del flexfit[name]
        self.update_flexfit(self._extra_sections)
        self.flexfit_changed.emit()

    def _on_flexfit_add_residue(self) -> None:
        row = self.flexfit_res_table.rowCount()
        self.flexfit_res_table.insertRow(row)
        self.flexfit_res_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.flexfit_res_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self.flush_flexfit(self._extra_sections)
        self.flexfit_changed.emit()

    def _on_flexfit_remove_residue(self) -> None:
        row = self.flexfit_res_table.currentRow()
        if row >= 0:
            self.flexfit_res_table.removeRow(row)
            self.flush_flexfit(self._extra_sections)
            self.flexfit_changed.emit()

    def _on_flexfit_add_bond(self) -> None:
        row = self.flexfit_bond_table.rowCount()
        self.flexfit_bond_table.insertRow(row)
        for col in range(6):
            self.flexfit_bond_table.setItem(row, col, QtWidgets.QTableWidgetItem(""))
        self.flush_flexfit(self._extra_sections)
        self.flexfit_changed.emit()

    def _on_flexfit_remove_bond(self) -> None:
        row = self.flexfit_bond_table.currentRow()
        if row >= 0:
            self.flexfit_bond_table.removeRow(row)
            self.flush_flexfit(self._extra_sections)
            self.flexfit_changed.emit()
