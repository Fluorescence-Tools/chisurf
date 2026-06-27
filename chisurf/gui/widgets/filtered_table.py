"""Reusable searchable/filterable table widget."""

from qtpy import QtCore, QtWidgets


class FilteredTableWidget(QtWidgets.QWidget):
    """A QTableWidget with integrated search/filter functionality.

    Features:
    - Search line edit above the table
    - Case-insensitive substring filtering by default
    - Custom filter function support
    - Selection preservation across filter changes
    - Custom item factory support for specialized items (e.g., with tooltips)

    Example:
        table = FilteredTableWidget()
        table.set_data([
            {"id": 1, "name": "Probe A"},
            {"id": 2, "name": "Probe B"},
        ], key_fn=lambda x: x["name"])
        table.set_filter_fn(lambda item, text: text.lower() in item["name"].lower())
    """

    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None, show_header=True, placeholder="Search…"):
        super().__init__(parent)
        self._data = []
        self._key_fn = lambda x: str(x)
        self._filter_fn = None
        self._item_factory = None
        self._selected_key = None
        self._setup_ui(show_header, placeholder)

    def _setup_ui(self, show_header, placeholder):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(placeholder)
        self.search_edit.setStyleSheet("padding: 1px 4px; font-size: 10px;")
        self.search_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self.search_edit)

        self.table = QtWidgets.QTableWidget()
        from chisurf.gui.widgets.general import apply_compact_table_style
        apply_compact_table_style(self.table)
        self.table.setColumnCount(1)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.table)

        if show_header:
            self.table.setHorizontalHeaderLabels(["Filter"])
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        else:
            self.table.horizontalHeader().setVisible(False)

        self.table.itemSelectionChanged.connect(self._emit_selection_changed)

    def set_data(self, data: list, key_fn=None):
        """Set the data to display.

        Parameters
        ----------
        data : list
            List of items to display.
        key_fn : callable, optional
            Function to extract the unique key from an item. Default: identity.
        """
        self._data = list(data)
        self._key_fn = key_fn if key_fn else lambda x: str(x)
        self._update_view()

    def set_filter_fn(self, filter_fn):
        """Set a custom filter function.

        Parameters
        ----------
        filter_fn : callable
            Function(item, text) -> bool. Return True to include item.
        """
        self._filter_fn = filter_fn
        self._update_view()

    def set_item_factory(self, factory_fn):
        """Set a custom factory for creating table items.

        Parameters
        ----------
        factory_fn : callable
            Function(item, key) -> QTableWidgetItem.
        """
        self._item_factory = factory_fn
        self._update_view()

    def _default_item_factory(self, item, key):
        """Default factory creates a plain label item.

        Specialized items (e.g. hover tooltips) are opt-in via
        :meth:`set_item_factory`, keeping this generic table free of any
        domain-specific dependency.
        """
        name = item.get("name") if isinstance(item, dict) else str(item)
        return QtWidgets.QTableWidgetItem(name)

    def _on_filter_changed(self):
        self._update_view()

    def _update_view(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)

        text = self.search_edit.text().strip().lower()
        visible = []
        for item in self._data:
            key = self._key_fn(item)
            if self._filter_fn:
                if self._filter_fn(item, text):
                    visible.append((key, item))
            else:
                # Use 'name' field from dict for filtering, not the key_fn result
                display_name = item.get("name") if isinstance(item, dict) else str(item)
                if not text or text in display_name.lower():
                    visible.append((key, item))

        self.table.setRowCount(len(visible))
        for row, (key, item) in enumerate(visible):
            if self._item_factory:
                widget_item = self._item_factory(item, key)
            else:
                widget_item = self._default_item_factory(item, key)
            widget_item.setData(QtCore.Qt.UserRole, key)
            self.table.setItem(row, 0, widget_item)

        self.table.blockSignals(False)

        if self._selected_key is not None:
            self._select_key(self._selected_key)

    def get_selected(self):
        """Get the key of the selected item."""
        rows = self.table.selectionModel().selectedRows()
        if rows:
            item = self.table.item(rows[0].row(), 0)
            if item is not None:
                return item.data(QtCore.Qt.UserRole)
        return None

    def set_selected(self, key):
        """Select an item by its key."""
        self._selected_key = key
        self._select_key(key)

    def _select_key(self, key):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(QtCore.Qt.UserRole) == key:
                self.table.selectRow(row)
                return

    def _emit_selection_changed(self):
        self.selectionChanged.emit()

    def clear(self):
        """Clear all data and selection."""
        self._data = []
        self._selected_key = None
        self.table.setRowCount(0)
