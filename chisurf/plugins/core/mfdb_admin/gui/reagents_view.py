"""Self-contained reagent / consumable inventory view (PRD-15 LIMS P4).

A thin Qt widget over the ``mfdb.reagents.*`` RPC handlers (PRD-23: logic in the
backend, view only renders): list reagent lots (kind filter + show-expired toggle),
show the selected lot's fields, and create a lot. Like ``StudiesView`` it is
**standalone** (constructed with an ``MFDBClient``) so it slots into the mfdb-admin
dock and smoke-tests in isolation.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

from chisurf.core.mfdb.reagents import REAGENT_KINDS

_KINDS = tuple(sorted(REAGENT_KINDS))
#: Lot columns shown in the table; the last is the (stable) lot id.
_LOT_COLUMNS = ("Kind", "Name", "Lot #", "Vendor", "Expiry", "Lot ID")
_LOT_KEYS = ("kind", "name", "lot_number", "vendor", "expiry", "lot_id")


class ReagentLotsView(QtWidgets.QWidget):
    """List + filter + per-lot detail + create for reagent lots."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client

        layout = QtWidgets.QVBoxLayout(self)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Kind:"))
        self.kind_filter_combo = QtWidgets.QComboBox()
        self.kind_filter_combo.addItem("all")
        self.kind_filter_combo.addItems(_KINDS)
        self.kind_filter_combo.currentTextChanged.connect(self.refresh)
        filter_row.addWidget(self.kind_filter_combo)
        self.show_expired_check = QtWidgets.QCheckBox("Show expired")
        self.show_expired_check.toggled.connect(self.refresh)
        filter_row.addWidget(self.show_expired_check)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self.lot_table = QtWidgets.QTableWidget(0, len(_LOT_COLUMNS))
        self.lot_table.setHorizontalHeaderLabels(list(_LOT_COLUMNS))
        self.lot_table.horizontalHeader().setStretchLastSection(True)
        self.lot_table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.lot_table)

        layout.addWidget(QtWidgets.QLabel("Selected lot:"))
        self.detail_table = QtWidgets.QTableWidget(0, 2)
        self.detail_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.detail_table)

        create_box = QtWidgets.QGroupBox("Add lot")
        create_form = QtWidgets.QFormLayout(create_box)
        self.new_kind_combo = QtWidgets.QComboBox()
        self.new_kind_combo.addItems(_KINDS)
        create_form.addRow("Kind", self.new_kind_combo)
        self.new_name_edit = QtWidgets.QLineEdit()
        self.new_name_edit.setPlaceholderText("e.g. Alexa 488")
        create_form.addRow("Name", self.new_name_edit)
        self.new_lot_number_edit = QtWidgets.QLineEdit()
        self.new_lot_number_edit.setPlaceholderText("e.g. A488-42")
        create_form.addRow("Lot #", self.new_lot_number_edit)
        self.new_vendor_edit = QtWidgets.QLineEdit()
        create_form.addRow("Vendor", self.new_vendor_edit)
        self.new_expiry_edit = QtWidgets.QLineEdit()
        self.new_expiry_edit.setPlaceholderText("YYYY-MM-DD")
        create_form.addRow("Expiry", self.new_expiry_edit)
        self.create_btn = QtWidgets.QPushButton("Create lot")
        self.create_btn.clicked.connect(self.create_lot)
        create_form.addRow(self.create_btn)
        layout.addWidget(create_box)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self.refresh()

    # -- data plumbing (via the client) --------------------------------------

    def refresh(self) -> None:
        kind = self.kind_filter_combo.currentText()
        lots = self._client.list_reagent_lots(
            kind=None if kind == "all" else kind,
            include_expired=self.show_expired_check.isChecked(),
        ) or []
        self.lot_table.setRowCount(len(lots))
        for row, lot in enumerate(lots):
            for col, key in enumerate(_LOT_KEYS):
                value = lot.get(key)
                self.lot_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem("" if value is None else str(value))
                )
        # Fit the fixed-width columns to content; the last (Lot ID) stretches to fill.
        self.lot_table.resizeColumnsToContents()
        self.detail_table.setRowCount(0)

    def _selected_lot(self) -> dict[str, Any] | None:
        items = self.lot_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        return {key: self.lot_table.item(row, col).text() for col, key in enumerate(_LOT_KEYS)}

    def _on_select(self) -> None:
        lot = self._selected_lot()
        if not lot:
            return
        fields = [(k, v) for k, v in lot.items() if v]
        self.detail_table.setRowCount(len(fields))
        for row, (k, v) in enumerate(fields):
            self.detail_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.detail_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(v)))

    def create_lot(self) -> None:
        name = self.new_name_edit.text().strip()
        if not name:
            self.message_label.setText("Enter a lot name.")
            return
        result = self._client.create_reagent_lot(
            kind=self.new_kind_combo.currentText(),
            name=name,
            lot_number=self.new_lot_number_edit.text().strip(),
            vendor=self.new_vendor_edit.text().strip(),
            expiry=self.new_expiry_edit.text().strip() or None,
        )
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        else:
            self.message_label.setText(f"Created lot {name}.")
            self.new_name_edit.clear()
            self.new_lot_number_edit.clear()
            self.new_vendor_edit.clear()
            self.new_expiry_edit.clear()
        self.refresh()
