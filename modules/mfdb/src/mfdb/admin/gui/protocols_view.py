"""Self-contained protocols view (PRD-14 Increment 3).

A thin Qt widget over the ``mfdb.protocols.*`` RPC handlers (PRD-23: logic in the
backend, view only renders): list protocols (scoped), show a protocol's version history
and its declared parameter schema (the operation_type's PRD-11 schema), and create a new
protocol/version. Like ``LifecycleView`` it is **standalone** (constructed with an
``MFDBClient``) so the in-flight mfdb-admin dock rewrite (``OVERHAUL_PLAN.md``) slots it
in without entangling the legacy tab framework, and so it smoke-tests in isolation.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

_CATEGORIES = ("measurement", "processing", "analysis")


class ProtocolsView(QtWidgets.QWidget):
    """List + version history + parameter schema + create for protocols."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client

        layout = QtWidgets.QVBoxLayout(self)

        scope_row = QtWidgets.QHBoxLayout()
        scope_row.addWidget(QtWidgets.QLabel("Scope:"))
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItems(["all", "own", "public"])
        self.scope_combo.currentTextChanged.connect(self.refresh)
        scope_row.addWidget(self.scope_combo)
        scope_row.addStretch()
        layout.addLayout(scope_row)

        self.protocol_table = QtWidgets.QTableWidget(0, 4)
        self.protocol_table.setHorizontalHeaderLabels(
            ["Name", "Version", "Category", "Operation type"]
        )
        self.protocol_table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.protocol_table)

        layout.addWidget(QtWidgets.QLabel("Version history:"))
        self.version_table = QtWidgets.QTableWidget(0, 3)
        self.version_table.setHorizontalHeaderLabels(["Version", "Description", "Created"])
        layout.addWidget(self.version_table)

        layout.addWidget(QtWidgets.QLabel("Parameter schema (from operation type):"))
        self.schema_table = QtWidgets.QTableWidget(0, 4)
        self.schema_table.setHorizontalHeaderLabels(["Name", "Type", "Required", "Units"])
        layout.addWidget(self.schema_table)

        create_row = QtWidgets.QHBoxLayout()
        self.new_name_edit = QtWidgets.QLineEdit()
        self.new_name_edit.setPlaceholderText("New protocol name")
        create_row.addWidget(self.new_name_edit)
        self.new_category_combo = QtWidgets.QComboBox()
        self.new_category_combo.addItems(_CATEGORIES)
        create_row.addWidget(self.new_category_combo)
        self.new_op_type_edit = QtWidgets.QLineEdit()
        self.new_op_type_edit.setPlaceholderText("operation_type (optional)")
        create_row.addWidget(self.new_op_type_edit)
        self.create_btn = QtWidgets.QPushButton("Create / new version")
        self.create_btn.clicked.connect(self.create_protocol)
        create_row.addWidget(self.create_btn)
        layout.addLayout(create_row)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self.refresh()

    # -- data plumbing (via the client) --------------------------------------

    def refresh(self) -> None:
        protocols = self._client.list_protocols(self.scope_combo.currentText()) or []
        self.protocol_table.setRowCount(len(protocols))
        for row, p in enumerate(protocols):
            for col, key in enumerate(("name", "version", "category", "operation_type")):
                self.protocol_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(str(p.get(key) or ""))
                )

    def _selected_name(self) -> str | None:
        items = self.protocol_table.selectedItems()
        if not items:
            return None
        return self.protocol_table.item(items[0].row(), 0).text()

    def _on_select(self) -> None:
        name = self._selected_name()
        if not name:
            return
        versions = self._client.list_protocol_versions(name) or []
        self.version_table.setRowCount(len(versions))
        for row, v in enumerate(versions):
            for col, key in enumerate(("version", "description", "created_at")):
                self.version_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(str(v.get(key) or ""))
                )

        schema = (self._client.get_protocol(name) or {}).get("parameter_schema", [])
        self.schema_table.setRowCount(len(schema))
        for row, s in enumerate(schema):
            for col, key in enumerate(("name", "value_type", "required", "units")):
                self.schema_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(str(s.get(key) if s.get(key) is not None else ""))
                )

    def create_protocol(self) -> None:
        name = self.new_name_edit.text().strip()
        if not name:
            self.message_label.setText("Enter a protocol name.")
            return
        result = self._client.create_protocol(
            name,
            self.new_category_combo.currentText(),
            operation_type=self.new_op_type_edit.text().strip() or None,
        )
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        else:
            self.message_label.setText(
                f"Created {name} v{result.get('version')}."
            )
            self.new_name_edit.clear()
        self.refresh()
