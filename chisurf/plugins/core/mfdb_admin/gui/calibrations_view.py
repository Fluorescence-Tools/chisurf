"""Self-contained calibration provenance view (PRD-05).

A thin Qt widget over the ``mfdb.calibrations.*`` RPC handlers (PRD-23: logic in the
backend, view only renders): list calibration records (type/method/value/notes),
register a literature ("god-given") value, and surface **stale uses** — fits whose
calibration is superseded by a newer one of the same type (the read side of "when a
calibration changes, find the downstream fits"). Standalone, like ``StudiesView`` /
``ReagentLotsView``.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

from chisurf.core.mfdb.staleness import CALIBRATION_TYPES

_CAL_COLUMNS = ("Type", "Method", "Value", "Notes", "Artifact ID")
_CAL_KEYS = ("calibration_type", "method", "value", "notes", "artifact_id")
_STALE_COLUMNS = ("Used by", "Type", "Used calibration", "Latest calibration")
_STALE_KEYS = ("used_by_id", "calibration_type", "used_artifact_id", "latest_artifact_id")


class CalibrationsView(QtWidgets.QWidget):
    """List calibrations + register literature values + show stale uses."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Calibrations:"))
        self.cal_table = QtWidgets.QTableWidget(0, len(_CAL_COLUMNS))
        self.cal_table.setHorizontalHeaderLabels(list(_CAL_COLUMNS))
        self.cal_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.cal_table)

        layout.addWidget(QtWidgets.QLabel("Stale uses (calibration superseded):"))
        self.stale_table = QtWidgets.QTableWidget(0, len(_STALE_COLUMNS))
        self.stale_table.setHorizontalHeaderLabels(list(_STALE_COLUMNS))
        self.stale_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stale_table)

        add_box = QtWidgets.QGroupBox("Register calibration value")
        form = QtWidgets.QFormLayout(add_box)
        self.new_type_combo = QtWidgets.QComboBox()
        self.new_type_combo.addItems(CALIBRATION_TYPES)
        form.addRow("Type", self.new_type_combo)
        self.new_value_edit = QtWidgets.QLineEdit()
        self.new_value_edit.setPlaceholderText("e.g. 54.0")
        form.addRow("Value", self.new_value_edit)
        self.new_notes_edit = QtWidgets.QLineEdit()
        self.new_notes_edit.setPlaceholderText("citation, e.g. Hellenkamp et al. 2018")
        form.addRow("Notes", self.new_notes_edit)
        self.create_btn = QtWidgets.QPushButton("Register (user-provided)")
        self.create_btn.clicked.connect(self.create_calibration)
        form.addRow(self.create_btn)
        layout.addWidget(add_box)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self.refresh()

    # -- data plumbing (via the client) --------------------------------------

    def refresh(self) -> None:
        cals = self._client.list_calibrations() or []
        self.cal_table.setRowCount(len(cals))
        for row, c in enumerate(cals):
            for col, key in enumerate(_CAL_KEYS):
                value = c.get(key)
                self.cal_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem("" if value is None else str(value))
                )
        self.cal_table.resizeColumnsToContents()

        stale = self._client.stale_calibrations() or []
        self.stale_table.setRowCount(len(stale))
        for row, s in enumerate(stale):
            for col, key in enumerate(_STALE_KEYS):
                value = s.get(key)
                self.stale_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem("" if value is None else str(value))
                )
        self.stale_table.resizeColumnsToContents()

    def create_calibration(self) -> None:
        text = self.new_value_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            self.message_label.setText("Enter a numeric value.")
            return
        result = self._client.create_calibration(
            self.new_type_combo.currentText(), value,
            notes=self.new_notes_edit.text().strip(),
        )
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        else:
            self.message_label.setText(
                f"Registered {self.new_type_combo.currentText()} = {value}."
            )
            self.new_value_edit.clear()
            self.new_notes_edit.clear()
        self.refresh()
