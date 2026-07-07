"""Per-sample metadata key-value editor dock.

Each sample has its own set of metadata key-value pairs stored in the
``flr_sample_key_value`` table. This dock provides a sample selector,
a key-value table with autocomplete, row-click detail binding, and
persistence.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

import chisurf.logging

from chisurf.gui.widgets.metadata_editor import (
    ALL_METADATA_KEYS,
    MetadataEditor,
    key_description,
)


class MetadataDock(QtWidgets.QWidget):
    """Per-sample metadata editor.

    Layout:
        Sample selector -> key-value table (MetadataEditor) ->
        detail form (key autocomplete, value, details) ->
        Save / Add / Delete buttons.

    Signals
    -------
    dataChanged(str)
        Emitted after metadata is saved for a sample.  Argument: sample_id.
    """

    dataChanged = QtCore.Signal(str)

    def __init__(
        self,
        client: Any,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._current_sample_id: str | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ---- Sample selector ----
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.setContentsMargins(4, 4, 4, 0)
        selector_layout.addWidget(QtWidgets.QLabel("Sample:"))
        self._sample_combo = QtWidgets.QComboBox()
        self._sample_combo.setEditable(True)
        self._sample_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._sample_combo.setPlaceholderText("Select a sample...")
        self._sample_combo.setMinimumWidth(300)
        self._sample_combo.currentIndexChanged.connect(self._on_sample_changed)
        self._sample_combo.editTextChanged.connect(self._on_sample_text_changed)
        selector_layout.addWidget(self._sample_combo, stretch=1)

        refresh_btn = QtWidgets.QToolButton()
        refresh_btn.setText("🔄")
        refresh_btn.setToolTip("Reload sample list and metadata")
        refresh_btn.clicked.connect(self.refresh_samples)
        selector_layout.addWidget(refresh_btn)
        layout.addLayout(selector_layout)

        # ---- Key-value table (MetadataEditor) ----
        self._editor = MetadataEditor(columns=3)
        self._editor.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self._editor, stretch=1)

        # ---- Detail form (auto-populated from selected row) ----
        detail_group = QtWidgets.QGroupBox("Detail")
        detail_layout = QtWidgets.QFormLayout(detail_group)
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(4)

        self._detail_key = QtWidgets.QComboBox()
        self._detail_key.setEditable(True)
        self._detail_key.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._detail_key.addItems(ALL_METADATA_KEYS)
        for i, k in enumerate(ALL_METADATA_KEYS):
            self._detail_key.setItemData(i, key_description(k), QtCore.Qt.UserRole + 1)
        comp = self._detail_key.completer()
        if comp is not None:
            comp.setFilterMode(QtCore.Qt.MatchContains)
            comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._detail_key.setCurrentIndex(-1)
        self._detail_key.currentTextChanged.connect(lambda text: self._update_value_completer(text))
        detail_layout.addRow("Key:", self._detail_key)

        self._detail_value = QtWidgets.QLineEdit()
        self._detail_value.setPlaceholderText("Value (autocomplete from DB)")
        self._detail_value.textChanged.connect(self._on_value_text_changed)
        detail_layout.addRow("Value:", self._detail_value)

        self._detail_details = QtWidgets.QLineEdit()
        self._detail_details.setPlaceholderText("Details / notes")
        detail_layout.addRow("Details:", self._detail_details)

        detail_layout.addRow(self._build_detail_buttons())
        layout.addWidget(detail_group)

        # ---- Status bar ----
        self._status_label = QtWidgets.QLabel("Select a sample to edit its metadata.")
        self._status_label.setStyleSheet("color: #555555; padding: 2px 4px;")
        layout.addWidget(self._status_label)

    def _build_detail_buttons(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar_layout = QtWidgets.QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(4)

        apply_btn = QtWidgets.QToolButton()
        apply_btn.setText("⬇ Apply to row")
        apply_btn.setToolTip("Update the currently selected metadata row with detail form values")
        apply_btn.clicked.connect(self._apply_detail_to_row)
        bar_layout.addWidget(apply_btn)

        bar_layout.addStretch()

        save_btn = QtWidgets.QToolButton()
        save_btn.setText("💾 Save all")
        save_btn.setToolTip("Persist all metadata rows for the current sample")
        save_btn.clicked.connect(self._save_metadata)
        bar_layout.addWidget(save_btn)

        add_btn = QtWidgets.QToolButton()
        add_btn.setText("➕ Add row")
        add_btn.setToolTip("Add an empty metadata row")
        add_btn.clicked.connect(self._editor._on_add_empty_row)
        bar_layout.addWidget(add_btn)

        delete_btn = QtWidgets.QToolButton()
        delete_btn.setText("🗑 Delete row")
        delete_btn.setToolTip("Delete the selected metadata row")
        delete_btn.clicked.connect(self._editor._on_delete_row)
        bar_layout.addWidget(delete_btn)

        return bar

    # ---- Public API ----

    def refresh_samples(self) -> None:
        """Reload the sample list from the backend."""
        block = self._sample_combo.blockSignals(True)
        current = self._sample_combo.currentText()
        self._sample_combo.clear()
        try:
            samples = self._client.list_samples()
        except Exception:
            samples = []

        for s in samples:
            sid = s.get("sample_id", "")
            label = s.get("description") or sid
            self._sample_combo.addItem(f"{sid} — {label}", sid)

        idx = self._sample_combo.findText(current, QtCore.Qt.MatchFlag(0))
        if idx >= 0:
            self._sample_combo.setCurrentIndex(idx)
        self._sample_combo.blockSignals(block)

        if self._current_sample_id:
            self._load_metadata(self._current_sample_id)

    def load_sample(self, sample_id: str) -> None:
        """Load metadata for a specific sample (called from external jump)."""
        self._current_sample_id = sample_id
        # Find or set the combo
        for i in range(self._sample_combo.count()):
            if self._sample_combo.itemData(i) == sample_id:
                self._sample_combo.setCurrentIndex(i)
                break
        else:
            self._sample_combo.setEditText(sample_id)
        self._load_metadata(sample_id)

    # ---- Internal: sample selection ----

    def _on_sample_changed(self, idx: int) -> None:
        sample_id = self._sample_combo.itemData(idx)
        if sample_id:
            self._current_sample_id = sample_id
            self._load_metadata(sample_id)

    def _on_sample_text_changed(self, text: str) -> None:
        # Extract sample_id from "sid — description" format
        if " — " in text:
            sid = text.split(" — ", 1)[0].strip()
            if sid:
                self._current_sample_id = sid
                self._load_metadata(sid)

    # ---- Internal: metadata CRUD ----

    def _load_metadata(self, sample_id: str) -> None:
        try:
            sample = self._client.get_sample(sample_id)
        except Exception:
            self._status_label.setText(f"Failed to load sample '{sample_id}'")
            return

        key_values = sample.get("key_values", []) if sample else []
        self._editor.set_data(key_values)
        # Add any DB-loaded keys not in the static list to the detail key combo
        for k in getattr(self._editor, "_extra_keys", []):
            if self._detail_key.findText(k, QtCore.Qt.MatchFixedString | QtCore.Qt.MatchCaseSensitive) == -1:
                self._detail_key.addItem(k)
        self._status_label.setText(
            f"Sample <b>{sample_id}</b>: {len(key_values)} metadata keys"
        )

    def _save_metadata(self) -> None:
        if not self._current_sample_id:
            self._status_label.setText("No sample selected.")
            return

        rows = self._editor.get_data()
        sample_id = self._current_sample_id
        try:
            self._client.save_sample_key_values(sample_id, rows)
            self._status_label.setText(
                f"Saved {len(rows)} metadata keys for sample <b>{sample_id}</b>."
            )
            self.dataChanged.emit(sample_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    # ---- Internal: row selection -> detail form ----

    def _on_row_selected(self) -> None:
        row = self._editor.table.currentRow()
        if row < 0:
            return

        key = self._editor._row_key(row)
        value = self._editor._row_value(row)
        details = self._editor._row_details(row)

        block = self._detail_key.blockSignals(True)
        self._detail_key.setEditText(key)
        self._detail_key.blockSignals(block)
        self._detail_key.setToolTip(key_description(key))

        self._detail_value.setText(value)
        self._detail_details.setText(details)

        # Update value completer for this key
        self._update_value_completer(key)

    def _apply_detail_to_row(self) -> None:
        """Push detail form values back to the currently selected table row."""
        row = self._editor.table.currentRow()
        if row < 0:
            self._status_label.setText("Select a metadata row first.")
            return

        key = self._detail_key.currentText().strip()
        value = self._detail_value.text().strip()
        details = self._detail_details.text().strip()

        # Update key combo
        combo = self._editor.table.cellWidget(row, 0)
        if isinstance(combo, QtWidgets.QComboBox):
            combo.setEditText(key)
            combo.setToolTip(key_description(key))

        # Update value
        val_item = self._editor.table.item(row, 1)
        if val_item is not None:
            val_item.setText(value)

        # Update details
        if self._editor._columns > 2:
            det_item = self._editor.table.item(row, 2)
            if det_item is not None:
                det_item.setText(details)

        self._status_label.setText("Detail applied to row.")

    # ---- Value autocomplete via backend ----

    # ---- Value autocomplete ----

    def _update_value_completer(self, key: str) -> None:
        """Set up the value field completer from known values for *key*."""
        suggestions = self._suggest_values_for_key(key)
        model = QtCore.QStringListModel(suggestions)
        completer = QtWidgets.QCompleter(model, self)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._detail_value.setCompleter(completer)

    def _on_value_text_changed(self, text: str) -> None:
        key = self._detail_key.currentText().strip()
        if not key or len(text) < 2:
            return
        self._update_value_completer(key)

    def _suggest_values_for_key(self, key: str) -> list[str]:
        """Query existing metadata values for a given key across all samples."""
        if not key:
            return []
        try:
            samples = self._client.list_samples()
        except Exception:
            return []

        seen: set[str] = set()
        values: list[str] = []
        for s in samples:
            sid = s.get("sample_id", "")
            if not sid:
                continue
            try:
                sample = self._client.get_sample(sid)
            except Exception:
                continue
            for kv in (sample.get("key_values") or []):
                if kv.get("key") == key:
                    v = kv.get("value", "")
                    if v and v not in seen:
                        seen.add(v)
                        values.append(v)
        return values
