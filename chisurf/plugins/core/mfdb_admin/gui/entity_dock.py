"""Generic dictionary-driven entity dock for mfdb-admin.

EntityDock composes all mixins into a single reusable dock widget.
Every editable table gets a uniform layout:
  filter bar → table → detail form → New / Save / Delete buttons

Clicking a row auto-fills the form; double-clicking an FK cell jumps
to the referenced entity dock via the ``jumpRequested`` signal.
"""

from __future__ import annotations

from typing import Any

import chisurf.logging

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.core.mfdb.dictionary_schema_map import DictionarySchemaMap
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary

from .entity_registry import EntitySpec, build_registry_dict
from .mixins import (
    CrossLinkMixin,
    CrudMixin,
    DockToolbarMixin,
    FormMixin,
    SchemaMixin,
    TableMixin,
)


class EntityDock(
    SchemaMixin,
    CrudMixin,
    TableMixin,
    FormMixin,
    CrossLinkMixin,
    DockToolbarMixin,
    QtWidgets.QWidget,
):
    """A single generic dock for browsing and editing one MFDB entity type.

    Signals
    -------
    jumpRequested(str, str)
        Emitted when a foreign-key cell is double-clicked.
        Arguments: (entity_key, record_id).
    statusMessage(str)
        Emitted to report status to the host window.
    """

    jumpRequested = QtCore.Signal(str, str)
    statusMessage = QtCore.Signal(str)

    def __init__(
        self,
        spec: EntitySpec,
        client: Any,
        dictionary: MmcifDictionary,
        schema_map: DictionarySchemaMap | None = None,
        registry_dict: dict[str, Any] | None = None,
        extra_buttons: list[QtWidgets.QWidget] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        self._client = client
        self._extra_buttons = extra_buttons or []
        if registry_dict is None:
            registry_dict = build_registry_dict()
        self._registry_dict = registry_dict

        super().__init__(
            spec=spec,
            dictionary=dictionary,
            schema_map=schema_map,
            registry_dict=registry_dict,
            parent=parent,
        )

        self._setup_ui()
        # Defer first load so the widget is fully visible first
        QtCore.QTimer.singleShot(0, self.refresh)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        toolbar = self.build_toolbar(extra_buttons=self._extra_buttons)
        layout.addWidget(toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._splitter = splitter  # exposed for state persistence
        layout.addWidget(splitter, stretch=1)

        # Table (top pane)
        table = self.build_table()
        table.itemSelectionChanged.connect(self._on_selection_changed)
        table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        splitter.addWidget(table)

        # Form (bottom pane)
        form = self.build_form(parent=self)
        # Auto-save whenever the user commits a field (Enter / focus-out /
        # dropdown selection).  Ctrl+Z undo still works inside each widget
        # because commitRequested fires only after the user explicitly commits.
        if hasattr(form, "commitRequested"):
            form.commitRequested.connect(self._on_auto_save)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._loading_form: bool = False

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            rows = self.list_rows()
            self.populate_table(rows)
            self._style_fk_cells()
            self._set_status(f"{len(rows)} {self._spec.title.lower()}")
        except Exception as exc:
            msg = f"Load failed: {exc}"
            self._set_status(msg)
            chisurf.logging.warning("EntityDock.refresh(%s): %s", self._spec.key, exc)

    def _set_status(self, msg: str) -> None:
        if hasattr(self, "_status_label"):
            self._status_label.setText(msg)
        self.statusMessage.emit(msg)

    # ------------------------------------------------------------------
    # Row selection → form auto-fill
    # ------------------------------------------------------------------

    def load_into_form(self, data: dict[str, Any]) -> None:
        """Populate form, suppressing auto-save during the load."""
        self._loading_form = True
        try:
            super().load_into_form(data)
        finally:
            self._loading_form = False

    def _on_selection_changed(self) -> None:
        """Populate the form when a table row is selected."""
        if self._table is None or self._form is None:
            return
        record_id = self.selected_row_id()
        if not record_id:
            return
        data = self._row_cache.get(record_id, {})
        if not data:
            try:
                data = self.get_row(record_id)
                self._row_cache[record_id] = data
            except Exception as exc:
                chisurf.logging.warning(
                    "EntityDock.get_row(%s, %s): %s", self._spec.key, record_id, exc
                )
        if data:
            self.load_into_form(data)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """FK cells emit jumpRequested; any cell triggers form autofill."""
        self._on_fk_double_clicked(row, col)

    # ------------------------------------------------------------------
    # CRUD buttons
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        """Create a new record immediately and open it in the form."""
        if not self._spec.writable:
            return
        # Decide what ID to pre-fill for user-defined string IDs
        id_spec = next(
            (fs for fs in self.field_specs if fs.name == self._spec.id_field), None
        )
        if id_spec and id_spec.readonly:
            # Auto-generated ID (e.g. integer PK, UUID): let the server assign
            new_data: dict[str, Any] = {}
        else:
            # User-defined string ID: pick the next "untitled_N"
            existing = set(self._row_cache.keys())
            n = 1
            while f"untitled_{n}" in existing:
                n += 1
            new_data = {self._spec.id_field: f"untitled_{n}"}

        try:
            result = self.save_row(new_data)
        except Exception as exc:
            self._set_status(f"Create failed: {exc}")
            chisurf.logging.warning("EntityDock._on_new(%s): %s", self._spec.key, exc)
            return

        # Refresh table so the new row appears, then select and load it
        self.refresh()
        new_id = str(result.get(self._spec.id_field, new_data.get(self._spec.id_field, "")))
        if new_id:
            self.select_row_by_id(new_id)
        self.load_into_form(result)
        self._set_status(f"Created {self._spec.key}: {new_id or '(new)'}")

    def _on_auto_save(self) -> None:
        """Auto-save the current form when the user commits a field edit."""
        if self._loading_form:
            return
        record_id = self.selected_row_id()
        if not record_id:
            return
        data = self.collect_form()
        try:
            result = self.save_row(data)
            saved = result or data
            self._row_cache[record_id] = saved
            self._update_selected_row(saved)
            self._set_status(f"Saved {self._spec.key} ✓")
        except Exception as exc:
            self._set_status(f"Auto-save failed: {exc}")
            chisurf.logging.warning("EntityDock._on_auto_save(%s): %s", self._spec.key, exc)

    def _update_selected_row(self, data: dict[str, Any]) -> None:
        """Update the currently selected table row in-place without a full refresh."""
        if self._table is None:
            return
        row = self._table.currentRow()
        if row < 0:
            return
        col_keys = [key for key, _ in self.columns()]
        for col_idx, key in enumerate(col_keys):
            val = data.get(key, "")
            item = self._table.item(row, col_idx + 1)
            if item is not None:
                item.setText(str(val) if val is not None else "")

    def _on_delete(self) -> None:
        ids = self.checked_row_ids()
        if not ids:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing checked",
                f"Tick the checkbox in the first column to select "
                f"{self._spec.title.lower()} to delete.",
            )
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            f"Delete {self._spec.title}",
            f"Delete {len(ids)} {self._spec.title.lower()}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        failures: list[str] = []
        for item_id in ids:
            try:
                self.delete_row(item_id)
            except Exception as exc:
                failures.append(f"{item_id}: {exc}")

        if failures:
            QtWidgets.QMessageBox.warning(
                self,
                "Some deletes failed",
                "\n".join(failures[:10]),
            )
        else:
            self._set_status(f"Deleted {len(ids)} {self._spec.title.lower()}")

        self.refresh()

    # ------------------------------------------------------------------
    # External navigation
    # ------------------------------------------------------------------

    def jump_to(self, record_id: str) -> None:
        """Switch focus to a specific record by ID and populate the form."""
        rid = str(record_id)
        self.select_row_by_id(rid)
        data = self._row_cache.get(rid, {})
        if not data:
            try:
                data = self.get_row(rid)
                self._row_cache[rid] = data
            except Exception:
                pass
        if data:
            self.load_into_form(data)
