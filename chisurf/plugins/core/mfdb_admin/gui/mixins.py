"""Composable mixins for the dictionary-driven EntityDock.

Each mixin provides a single responsibility and operates on self.spec +
self.client. They are combined in EntityDock via multiple inheritance.
"""

from __future__ import annotations

from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets

import chisurf.logging

from chisurf.core.mfdb.dictionary_schema_map import DictionarySchemaMap
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
from chisurf.gui.widgets.general import apply_compact_table_style

from .entity_schema import FieldSpec, field_specs_for_category
from .entity_registry import EntitySpec
from .generic_form import SCHEMAS


class SchemaMixin:
    """Provides lazy field_specs, columns(), and id_field from the dictionary."""

    def __init__(
        self,
        spec: EntitySpec,
        dictionary: MmcifDictionary,
        schema_map: DictionarySchemaMap | None = None,
        registry_dict: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._spec = spec
        self._dictionary = dictionary
        self._schema_map = schema_map
        self._registry_dict = registry_dict or {}
        self._field_specs: list[FieldSpec] | None = None
        super().__init__(*args, **kwargs)

    @property
    def spec(self) -> EntitySpec:
        return self._spec

    @property
    def field_specs(self) -> list[FieldSpec]:
        if self._field_specs is None:
            # Try dictionary-driven first
            if self._spec.schema_type and self._spec.schema_type in SCHEMAS:
                self._field_specs = self._legacy_specs()
            else:
                self._field_specs = field_specs_for_category(
                    self._dictionary,
                    self._spec.category,
                    schema_map=self._schema_map,
                    registry=self._registry_dict,
                    id_field=self._spec.id_field,
                )
                if not self._field_specs:
                    self._field_specs = self._legacy_specs()
        return self._field_specs

    def _legacy_specs(self) -> list[FieldSpec]:
        legacy = SCHEMAS.get(self._spec.schema_type, [])
        result: list[FieldSpec] = []
        for f in legacy:
            result.append(FieldSpec(
                name=f.get("name", ""),
                label=f.get("label", ""),
                widget=f.get("type", "str"),
                choices=f.get("choices", []),
                required=f.get("required", False),
                readonly=f.get("readonly", False),
                placeholder=f.get("placeholder", ""),
                fk_target=self._legacy_fk(f.get("name", "")),
            ))
        return result

    def _legacy_fk(self, name: str) -> str | None:
        # Skip the entity's own primary key
        if name == self._spec.id_field:
            return None
        if name.endswith("_id") and name not in ("id",):
            base = name[:-3]
            if base in self._registry_dict:
                return base
        return None

    def columns(self) -> list[tuple[str, str]]:
        """Return (key, label) pairs for visible table columns.

        The ID field is always first so that id_col=1 in the table is reliable.
        Timestamp-only audit fields (created_at, updated_at) are hidden.
        """
        id_col: tuple[str, str] | None = None
        rest: list[tuple[str, str]] = []
        for fs in self.field_specs:
            if fs.name in ("created_at", "updated_at"):
                continue
            pair = (fs.name, fs.label or fs.name)
            if fs.name == self._spec.id_field:
                id_col = pair
            else:
                rest.append(pair)
        if id_col:
            return [id_col] + rest
        return rest

    @property
    def id_field(self) -> str:
        return self._spec.id_field


class CrudMixin:
    """Generic CRUD operations over MFDB RPC calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def list_rows(self, **filters: Any) -> list[dict[str, Any]]:
        method = f"mfdb.{self._spec.rpc}.list"
        params = filters if filters else None
        result = self._client._call(method, params)
        return list(result.get(self._spec.list_key, []))

    def get_row(self, row_id: str) -> dict[str, Any]:
        method = f"mfdb.{self._spec.rpc}.get"
        params = {self._spec.id_field: row_id}
        result = self._client._call(method, params)
        return result.get(self._spec.item_key) or {}

    def save_row(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self._spec.writable:
            raise RuntimeError(f"Entity '{self._spec.key}' is read-only")
        method = f"mfdb.{self._spec.rpc}.save"
        result = self._client._call(method, {self._spec.key: data})
        return result.get(self._spec.item_key) or {}

    def delete_row(self, row_id: str) -> dict[str, Any]:
        if not self._spec.writable:
            raise RuntimeError(f"Entity '{self._spec.key}' is read-only")
        method = f"mfdb.{self._spec.rpc}.delete"
        params = {self._spec.id_field: row_id}
        return self._client._call(method, params)


class TableMixin:
    """Manages a QTableWidget driven by field_spec columns.

    The first data column (index 1, after the checkbox at 0) is always the
    entity's ID field so that ``checked_row_ids(id_col=1)`` and
    ``select_row_by_id(id_col=1, ...)`` are always correct.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._table: QtWidgets.QTableWidget | None = None
        self._row_cache: dict[str, dict[str, Any]] = {}

    def build_table(self) -> QtWidgets.QTableWidget:
        cols = self.columns()
        table = QtWidgets.QTableWidget(0, len(cols) + 1)
        headers = ["✓"] + [label for _, label in cols]
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setSortingEnabled(True)
        apply_compact_table_style(table)
        # Re-enable sorting (apply_compact_table_style disables it)
        table.setSortingEnabled(True)
        # Checkbox column narrow; last column stretches via apply_compact_table_style
        table.setColumnWidth(0, 28)
        self._table = table
        return table

    def populate_table(self, rows: list[dict[str, Any]]) -> None:
        if self._table is None:
            return
        table = self._table
        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setRowCount(0)

        col_keys = [key for key, _ in self.columns()]
        id_field = self._spec.id_field

        self._row_cache = {}

        for record in rows:
            row_idx = table.rowCount()
            table.insertRow(row_idx)

            # Cache full record keyed by id
            record_id = str(record.get(id_field, ""))
            if record_id:
                self._row_cache[record_id] = record

            # Checkbox
            cb = QtWidgets.QTableWidgetItem("")
            cb.setFlags(
                QtCore.Qt.ItemIsUserCheckable
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsSelectable
            )
            cb.setCheckState(QtCore.Qt.Unchecked)
            cb.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row_idx, 0, cb)

            # Data columns (id is always first per SchemaMixin.columns())
            for col_idx, key in enumerate(col_keys):
                val = record.get(key, "")
                if val is None:
                    val = ""
                cell = QtWidgets.QTableWidgetItem(str(val))
                cell.setFlags(cell.flags() & ~QtCore.Qt.ItemIsEditable)
                table.setItem(row_idx, col_idx + 1, cell)

            # UUID tooltip on the ID cell (col 1)
            uuid_val = next(
                (str(v) for k, v in record.items() if "uuid" in k.lower() and v),
                "",
            )
            if uuid_val:
                id_cell = table.item(row_idx, 1)
                if id_cell is not None:
                    id_cell.setToolTip(f"UUID: {uuid_val}")

        table.setSortingEnabled(was_sorting)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 28)

    @property
    def table(self) -> QtWidgets.QTableWidget | None:
        return self._table

    def checked_row_ids(self) -> list[str]:
        """Return IDs (from column 1, the id field) of all checked rows."""
        if self._table is None:
            return []
        ids: list[str] = []
        for row in range(self._table.rowCount()):
            cb = self._table.item(row, 0)
            if cb is not None and cb.checkState() == QtCore.Qt.Checked:
                id_item = self._table.item(row, 1)
                if id_item is not None:
                    ids.append(id_item.text().strip())
        return ids

    def select_row_by_id(self, value: str) -> None:
        """Select the row whose id column (col 1) matches value."""
        if self._table is None or not value:
            return
        for row in range(self._table.rowCount()):
            cell = self._table.item(row, 1)
            if cell and cell.text() == str(value):
                self._table.selectRow(row)
                self._table.scrollToItem(
                    cell, QtWidgets.QAbstractItemView.PositionAtCenter
                )
                return

    def selected_row_id(self) -> str | None:
        """Return the id of the currently selected row, or None."""
        if self._table is None:
            return None
        rows = self._table.selectedItems()
        if not rows:
            return None
        row = self._table.currentRow()
        if row < 0:
            return None
        id_item = self._table.item(row, 1)
        return id_item.text().strip() if id_item else None


class FormMixin:
    """Manages a MFDBDetailWidget driven by field_specs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._form: Any | None = None  # MFDBDetailWidget

    def build_form(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
        from .autoform_entity_form import EntityForm

        # Use self._registry_dict directly — no MRO class search
        rd: dict[str, Any] = getattr(self, "_registry_dict", {})
        client = getattr(self, "_client", None)

        dropdown_providers: dict[str, Callable[[], list[tuple[str, str]]]] = {}
        for fs in self.field_specs:
            if fs.fk_target and client is not None:
                target_spec = rd.get(fs.fk_target)
                if target_spec:
                    provider = self._make_dropdown_provider(target_spec)
                    dropdown_providers[fs.name] = provider

        # Render the entity detail form through the declarative AutoForm
        # machinery (PRD-40). EntityForm matches MFDBDetailWidget's public API
        # (set_data/get_data/commitRequested), so the rest of FormMixin is
        # transport-agnostic — load_into_form/collect_form duck-type the form.
        form = EntityForm(
            field_specs=self.field_specs,
            dropdown_providers=dropdown_providers,
            parent=parent,
        )
        self._form = form
        return form

    def _make_dropdown_provider(
        self, target_spec: dict[str, Any]
    ) -> Callable[[], list[tuple[str, str]]]:
        rpc = target_spec["rpc"]
        list_key = target_spec["list_key"]
        id_f = target_spec["id_field"]
        client = self._client

        def provider() -> list[tuple[str, str]]:
            try:
                result = client._call(f"mfdb.{rpc}.list", None)
                items = result.get(list_key, [])
                out = []
                for r in items:
                    val = str(r.get(id_f, ""))
                    lbl = (
                        r.get("name")
                        or r.get("display_name")
                        or r.get("description")
                        or r.get("chromophore_name")
                        or val
                    )
                    out.append((val, f"{val} — {lbl}" if lbl != val else val))
                return out
            except Exception:
                return []

        return provider

    def load_into_form(self, data: dict[str, Any]) -> None:
        """Populate the form from a record dict."""
        if self._form is not None and hasattr(self._form, "set_data"):
            self._form.set_data(data)

    def collect_form(self) -> dict[str, Any]:
        """Collect current form values."""
        if self._form is not None and hasattr(self._form, "get_data"):
            return self._form.get_data()
        return {}

    @property
    def form(self) -> Any | None:
        return self._form


class CrossLinkMixin:
    """Renders FK cells as blue-underlined links; double-click emits jump request."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _style_fk_cells(self) -> None:
        """Apply link styling to cells that correspond to FK fields."""
        if self._table is None:
            return
        fk_col_indices: list[int] = []
        col_keys = [key for key, _ in self.columns()]
        for col_idx, key in enumerate(col_keys):
            for fs in self.field_specs:
                if fs.name == key and fs.fk_target:
                    fk_col_indices.append(col_idx + 1)  # +1 for checkbox
                    break

        if not fk_col_indices:
            return

        link_color = QtGui.QColor("#1565C0")
        for row in range(self._table.rowCount()):
            for col in fk_col_indices:
                item = self._table.item(row, col)
                if item and item.text():
                    item.setForeground(QtGui.QBrush(link_color))
                    font = item.font()
                    font.setUnderline(True)
                    item.setFont(font)

    def _on_fk_double_clicked(self, row: int, col: int) -> None:
        """Emit jumpRequested if the double-clicked cell is an FK column.

        Only emits when the resolved fk_target is a known registry key so that
        dictionary parent links that point to non-navigable categories (e.g.
        ``entity_poly_seq``) don't produce silent navigation failures.
        """
        if self._table is None or col == 0:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        fk_value = item.text().strip()
        if not fk_value:
            return
        col_keys = [key for key, _ in self.columns()]  # always fresh
        data_col = col - 1  # -1 for checkbox
        if data_col < 0 or data_col >= len(col_keys):
            return
        key = col_keys[data_col]
        rd = getattr(self, "_registry_dict", {})
        for fs in self.field_specs:
            if fs.name == key and fs.fk_target:
                if fs.fk_target not in rd:
                    return  # target not in registry — can't navigate
                self.jumpRequested.emit(fs.fk_target, fk_value)
                return


class DockToolbarMixin:
    """Provides the uniform button row: New / Save / Delete + filter bar."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._filter_edit: QtWidgets.QLineEdit | None = None

    def build_toolbar(
        self, extra_buttons: list[QtWidgets.QWidget] | None = None
    ) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText("Filter...")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_edit, stretch=1)

        if self._spec.writable:
            new_btn = QtWidgets.QToolButton()
            new_btn.setText("➕ New")
            new_btn.setToolTip(f"Create a new {self._spec.key} — edit fields, changes save automatically on Enter")
            new_btn.clicked.connect(self._on_new)
            layout.addWidget(new_btn)

            del_btn = QtWidgets.QToolButton()
            del_btn.setText("🗑 Delete")
            del_btn.setToolTip("Delete checked rows")
            del_btn.clicked.connect(self._on_delete)
            layout.addWidget(del_btn)

        if extra_buttons:
            for btn in extra_buttons:
                layout.addWidget(btn)

        return bar

    def _on_filter_changed(self, text: str) -> None:
        if self._table is None:
            return
        text_lower = text.strip().lower()
        for row in range(self._table.rowCount()):
            visible = not text_lower
            if not visible:
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    if item and text_lower in item.text().lower():
                        visible = True
                        break
            self._table.setRowHidden(row, not visible)
