"""Self-contained lifecycle state + history view (PRD-12 Increment 4).

A thin Qt widget over the ``mfdb.lifecycle.*`` RPC handlers (PRD-23: all logic is in
the backend; this view only calls the client and renders). It shows an entity's current
state and transition history and offers the legal next transitions. It is deliberately
**standalone** — constructed with an ``MFDBClient`` — so it can be slotted into the
mfdb-admin dock layout (currently being rewritten, see ``OVERHAUL_PLAN.md``) without
entangling the in-flux tab framework, and so it can be smoke-tested in isolation.

>>> view = LifecycleView(client)            # client: MFDBClient
>>> view.set_entity("sample", "s1"); view.refresh()
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets


class LifecycleView(QtWidgets.QWidget):
    """Current state + history + admin transition for one entity."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._defs: dict[str, Any] = {}

        layout = QtWidgets.QVBoxLayout(self)

        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("Entity type:"))
        self.entity_type_combo = QtWidgets.QComboBox()
        picker.addWidget(self.entity_type_combo)
        picker.addWidget(QtWidgets.QLabel("Entity ID:"))
        self.entity_id_edit = QtWidgets.QLineEdit()
        picker.addWidget(self.entity_id_edit)
        self.refresh_btn = QtWidgets.QPushButton("Load")
        self.refresh_btn.clicked.connect(self.refresh)
        picker.addWidget(self.refresh_btn)
        layout.addLayout(picker)

        self.state_label = QtWidgets.QLabel("Current state: —")
        layout.addWidget(self.state_label)

        transition = QtWidgets.QHBoxLayout()
        transition.addWidget(QtWidgets.QLabel("Transition to:"))
        self.to_state_combo = QtWidgets.QComboBox()
        transition.addWidget(self.to_state_combo)
        transition.addWidget(QtWidgets.QLabel("Reason:"))
        self.reason_edit = QtWidgets.QLineEdit()
        transition.addWidget(self.reason_edit)
        self.transition_btn = QtWidgets.QPushButton("Apply transition")
        self.transition_btn.clicked.connect(self.apply_transition)
        transition.addWidget(self.transition_btn)
        layout.addLayout(transition)

        self.history_table = QtWidgets.QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["From", "To", "When", "Reason"])
        layout.addWidget(self.history_table)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self._load_definitions()

    # -- data plumbing (all via the client / backend) ------------------------

    def _load_definitions(self) -> None:
        try:
            self._defs = self._client.lifecycle_definitions() or {}
        except Exception as exc:  # pragma: no cover - defensive
            self._defs = {}
            self.message_label.setText(f"Could not load lifecycle definitions: {exc}")
        self.entity_type_combo.clear()
        self.entity_type_combo.addItems(sorted(self._defs))

    def set_entity(self, entity_type: str, entity_id: str) -> None:
        """Pre-select an entity (used when opening the view from another record)."""
        idx = self.entity_type_combo.findText(entity_type)
        if idx >= 0:
            self.entity_type_combo.setCurrentIndex(idx)
        self.entity_id_edit.setText(entity_id)

    def _current_entity(self) -> tuple[str, str]:
        return self.entity_type_combo.currentText(), self.entity_id_edit.text().strip()

    def _allowed_next_states(self, entity_type: str, current: str | None) -> list[str]:
        spec = self._defs.get(entity_type, {})
        return [
            to for frm, to in spec.get("transitions", [])
            if (frm if frm else None) == current
        ]

    def refresh(self) -> None:
        """Reload the current state, the legal next states, and the history."""
        entity_type, entity_id = self._current_entity()
        if not entity_id:
            self.message_label.setText("Enter an entity ID.")
            return
        current = self._client.lifecycle_state(entity_type, entity_id)
        self.state_label.setText(f"Current state: {current or '—'}")

        self.to_state_combo.clear()
        self.to_state_combo.addItems(self._allowed_next_states(entity_type, current))

        history = self._client.lifecycle_history(entity_type, entity_id) or []
        self.history_table.setRowCount(len(history))
        for row, h in enumerate(history):
            for col, key in enumerate(("from_state", "to_state", "created_at", "reason")):
                item = QtWidgets.QTableWidgetItem(str(h.get(key) or ""))
                self.history_table.setItem(row, col, item)
        self.message_label.setText("")

    def apply_transition(self) -> None:
        """Apply the selected transition, then refresh; surface an illegal jump."""
        entity_type, entity_id = self._current_entity()
        to_state = self.to_state_combo.currentText()
        if not entity_id or not to_state:
            self.message_label.setText("Pick an entity and a target state.")
            return
        result = self._client.lifecycle_transition(
            entity_type, entity_id, to_state, reason=self.reason_edit.text().strip()
        )
        # Refresh first (it updates state/history and clears the message), then show
        # the outcome so it is not wiped by the refresh.
        self.refresh()
        if result.get("error"):
            self.message_label.setText(f"Rejected: {result['error']}")
        elif result.get("changed"):
            self.message_label.setText(f"Transitioned to {result.get('state')}.")
            self.reason_edit.clear()
        else:
            self.message_label.setText("No change (already in that state).")
