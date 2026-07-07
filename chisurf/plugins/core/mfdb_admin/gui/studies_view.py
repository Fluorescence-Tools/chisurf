"""Self-contained studies/projects view (PRD-13 Increment 3).

A thin Qt widget over the ``mfdb.studies.*`` RPC handlers (PRD-23: logic in the backend,
view only renders): list studies (scoped), show a study's members and configurable
fields, and create a study / add a member / set a field. Like ``LifecycleView`` /
``ProtocolsView`` it is **standalone** (constructed with an ``MFDBClient``) so the
in-flight mfdb-admin dock rewrite (``OVERHAUL_PLAN.md``) slots it in, and it smoke-tests
in isolation.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

_MEMBER_TYPES = ("sample", "artifact")


class StudiesView(QtWidgets.QWidget):
    """List + members + configurable fields + create for studies."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client

        layout = QtWidgets.QVBoxLayout(self)

        scope_row = QtWidgets.QHBoxLayout()
        scope_row.addWidget(QtWidgets.QLabel("Scope:"))
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItems(["all", "mine", "public"])
        self.scope_combo.currentTextChanged.connect(self.refresh)
        scope_row.addWidget(self.scope_combo)
        scope_row.addStretch()
        layout.addLayout(scope_row)

        self.study_table = QtWidgets.QTableWidget(0, 3)
        self.study_table.setHorizontalHeaderLabels(["Name", "Public", "Study ID"])
        self.study_table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.study_table)

        layout.addWidget(QtWidgets.QLabel("Members:"))
        self.member_table = QtWidgets.QTableWidget(0, 3)
        self.member_table.setHorizontalHeaderLabels(["Type", "ID", "Role"])
        layout.addWidget(self.member_table)

        layout.addWidget(QtWidgets.QLabel("Configurable fields:"))
        self.field_table = QtWidgets.QTableWidget(0, 2)
        self.field_table.setHorizontalHeaderLabels(["Key", "Value"])
        layout.addWidget(self.field_table)

        create_row = QtWidgets.QHBoxLayout()
        self.new_name_edit = QtWidgets.QLineEdit()
        self.new_name_edit.setPlaceholderText("New study name")
        create_row.addWidget(self.new_name_edit)
        self.create_btn = QtWidgets.QPushButton("Create study")
        self.create_btn.clicked.connect(self.create_study)
        create_row.addWidget(self.create_btn)
        layout.addLayout(create_row)

        member_row = QtWidgets.QHBoxLayout()
        self.member_type_combo = QtWidgets.QComboBox()
        self.member_type_combo.addItems(_MEMBER_TYPES)
        member_row.addWidget(self.member_type_combo)
        self.member_id_edit = QtWidgets.QLineEdit()
        self.member_id_edit.setPlaceholderText("member id (sample/artifact)")
        member_row.addWidget(self.member_id_edit)
        self.add_member_btn = QtWidgets.QPushButton("Add member")
        self.add_member_btn.clicked.connect(self.add_member)
        member_row.addWidget(self.add_member_btn)
        layout.addLayout(member_row)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self.refresh()

    # -- data plumbing (via the client) --------------------------------------

    def refresh(self) -> None:
        studies = self._client.list_studies(self.scope_combo.currentText()) or []
        self.study_table.setRowCount(len(studies))
        for row, s in enumerate(studies):
            self.study_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(s.get("name") or "")))
            self.study_table.setItem(row, 1, QtWidgets.QTableWidgetItem("yes" if s.get("is_public") else "no"))
            self.study_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(s.get("study_id") or "")))

    def _selected_study_id(self) -> str | None:
        items = self.study_table.selectedItems()
        if not items:
            return None
        return self.study_table.item(items[0].row(), 2).text()

    def _on_select(self) -> None:
        sid = self._selected_study_id()
        if not sid:
            return
        detail = self._client.get_study(sid) or {}
        members = detail.get("members", [])
        self.member_table.setRowCount(len(members))
        for row, m in enumerate(members):
            for col, key in enumerate(("member_type", "member_id", "role")):
                self.member_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(m.get(key) or "")))
        fields = detail.get("fields", {}) or {}
        self.field_table.setRowCount(len(fields))
        for row, (k, v) in enumerate(sorted(fields.items())):
            self.field_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.field_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(v)))

    def create_study(self) -> None:
        name = self.new_name_edit.text().strip()
        if not name:
            self.message_label.setText("Enter a study name.")
            return
        result = self._client.create_study(name)
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        else:
            self.message_label.setText(f"Created study {name}.")
            self.new_name_edit.clear()
        self.refresh()

    def add_member(self) -> None:
        sid = self._selected_study_id()
        member_id = self.member_id_edit.text().strip()
        if not sid or not member_id:
            self.message_label.setText("Select a study and enter a member id.")
            return
        result = self._client.add_study_member(
            sid, self.member_type_combo.currentText(), member_id
        )
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        else:
            self.message_label.setText("Member added.")
            self.member_id_edit.clear()
        self._on_select()
