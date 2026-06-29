"""Reusable MFDB dataset-picker widget and dialog.

Provides :class:`MfdbDatasetBrowser` (standalone widget) and
:class:`MfdbDatasetPickerDialog` (modal dialog with
:meth:`MfdbDatasetPickerDialog.pick_dataset`) for selecting a dataset
registered in the MFDB object store.

All MFDB communication goes through an RPC client (duck-typed with a
``call(method, params)`` method). No direct repository or database calls
in the Qt layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.general import apply_compact_table_style


@dataclass
class DatasetSelection:
    """Result of a successful dataset pick.

    Parameters
    ----------
    artifact_id : str
        Selected artifact identifier.
    artifact_kind : str
        Artifact kind (e.g. ``raw_measurement``).
    data_format : str or None
        Data format vocabulary value.
    label : str
        Human-readable label for display.
    local_path : str or None
        Local file path if the caller also called ``datasets.open``.
    metadata : dict
        Arbitrary metadata from the artifact.
    """
    artifact_id: str
    artifact_kind: str
    data_format: str | None
    label: str
    local_path: str | None = None
    metadata: dict = field(default_factory=dict)


def _get_display_name(ds: dict[str, Any]) -> str:
    import os

    art_id = ds.get("artifact_id", "")
    orig_filename = ds.get("original_filename")
    # Prefer the stored object name, then an on-disk file/folder path. Directory
    # references (e.g. a burst run's output folder) carry no object and store
    # their path in ``folder_path`` or ``metadata_json["path"]`` — use that
    # basename so the run shows a real name instead of its UUID.
    for candidate in (ds.get("file_path"), ds.get("folder_path")):
        if not orig_filename and candidate:
            orig_filename = os.path.basename(str(candidate).rstrip("/\\"))
    if not orig_filename:
        metadata = ds.get("metadata_json") or ds.get("metadata")
        if isinstance(metadata, str):
            try:
                import json

                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = None
        if isinstance(metadata, dict):
            meta_path = metadata.get("path") or metadata.get("folder_path")
            if meta_path:
                orig_filename = os.path.basename(str(meta_path).rstrip("/\\"))
    return orig_filename or art_id


class MfdbDatasetBrowser(QtWidgets.QWidget):
    """Reusable widget for browsing MFDB datasets.

    Supports scope toggling (Mine / Public / All), debounced search,
    kind/format filters (optionally locked by the caller), a two-pane
    sample → dataset view, and a flat "All datasets" mode.

    Parameters
    ----------
    client : object, optional
        RPC client with a ``call(method, params)`` interface. When omitted
        the widget operates in a disabled ``'not connected'`` state.
    kinds : list of str, optional
        Pre-set artifact kind filter.
    formats : list of str, optional
        Pre-set data format filter.
    parent : QWidget, optional
        Parent widget.
    """

    datasetSelected = QtCore.Signal(object)  # DatasetSelection

    def __init__(
        self,
        client: Any = None,
        kinds: list[str] | None = None,
        formats: list[str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._kinds = kinds
        self._formats = formats
        self._datasets: list[dict[str, Any]] = []
        self._total: int = 0
        self._sample_counts: dict[str, int] = {}
        self._debounce_timer: QtCore.QTimer | None = None
        self._current_page: int = 0
        self._page_size: int = 50
        self._search_timer: QtCore.QTimer | None = None

        self._build_ui()
        self._refresh_connected_state()
        if self._is_connected():
            self._refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # scope toggle
        scope_layout = QtWidgets.QHBoxLayout()
        scope_layout.setSpacing(2)
        self.scope_combo = QtWidgets.QComboBox(self)
        self.scope_combo.addItem("Mine", "own")
        self.scope_combo.addItem("Public", "public")
        self.scope_combo.addItem("All", "all")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_layout.addWidget(QtWidgets.QLabel("Scope:"))
        scope_layout.addWidget(self.scope_combo)
        scope_layout.addStretch()
        layout.addLayout(scope_layout)

        # search bar
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Search datasets...")
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        layout.addWidget(self.search_edit)

        # filter row
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.setSpacing(2)
        if self._kinds:
            kind_label = ", ".join(self._kinds)
            kind_widget = QtWidgets.QLabel(f"Kind: {kind_label}")
            kind_widget.setStyleSheet("font-weight: bold; color: #666;")
            filter_layout.addWidget(kind_widget)
        if self._formats:
            fmt_label = ", ".join(self._formats)
            fmt_widget = QtWidgets.QLabel(f"Format: {fmt_label}")
            fmt_widget.setStyleSheet("font-weight: bold; color: #666;")
            filter_layout.addWidget(fmt_widget)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # dataset list
        self.dataset_list = QtWidgets.QTableWidget(self)
        self.dataset_list.setColumnCount(6)
        self.dataset_list.setHorizontalHeaderLabels([
            "Filename", "Kind", "Format", "Sample Name", "Deposition Date", "Ref Count"
        ])
        apply_compact_table_style(self.dataset_list)
        self.dataset_list.setSortingEnabled(True)
        self.dataset_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.dataset_list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dataset_list.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.dataset_list.itemDoubleClicked.connect(self._on_dataset_activated)
        self.dataset_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.dataset_list, 1)

        # pagination bar
        pagination_layout = QtWidgets.QHBoxLayout()
        pagination_layout.setSpacing(2)
        self.prev_btn = QtWidgets.QPushButton("< Prev", self)
        self.prev_btn.clicked.connect(self._on_prev_page)
        self.prev_btn.setEnabled(False)
        pagination_layout.addWidget(self.prev_btn)
        self.page_label = QtWidgets.QLabel("", self)
        pagination_layout.addWidget(self.page_label)
        self.next_btn = QtWidgets.QPushButton("Next >", self)
        self.next_btn.clicked.connect(self._on_next_page)
        self.next_btn.setEnabled(False)
        pagination_layout.addWidget(self.next_btn)
        pagination_layout.addStretch()
        self.total_label = QtWidgets.QLabel("", self)
        pagination_layout.addWidget(self.total_label)
        layout.addLayout(pagination_layout)

        # status bar
        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

    # ── connection state ───────────────────────────────────────────

    def set_client(self, client: Any) -> None:
        self._client = client
        self._refresh_connected_state()
        if self._is_connected():
            self._refresh()

    def _is_connected(self) -> bool:
        return self._client is not None

    def _refresh_connected_state(self) -> None:
        connected = self._is_connected()
        self.scope_combo.setEnabled(connected)
        self.search_edit.setEnabled(connected)
        self.dataset_list.setEnabled(connected)
        self.prev_btn.setEnabled(connected and self._current_page > 0)
        self.next_btn.setEnabled(connected and (self._current_page + 1) * self._page_size < self._total)
        if not connected:
            self.status_label.setText("MFDB not connected")
            self.dataset_list.setRowCount(0)
            self.total_label.setText("")

    def _call_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._client:
            return {"datasets": [], "total": 0, "sample_counts": {}}
        try:
            result = self._client.call(method, params or {})
            return result or {}
        except Exception as exc:
            self.status_label.setText(f"RPC error: {exc}")
            return {"datasets": [], "total": 0, "sample_counts": {}}

    # ── data loading ───────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self._is_connected():
            return
        scope = self.scope_combo.currentData() or "all"
        query = self.search_edit.text().strip() or None
        result = self._call_rpc("mfdb.datasets.browse", {
            "scope": scope,
            "query": query,
            "kinds": self._kinds,
            "formats": self._formats,
            "limit": self._page_size,
            "offset": self._current_page * self._page_size,
        })
        self._datasets = result.get("datasets", [])
        self._total = result.get("total", 0)
        self._sample_counts = result.get("sample_counts", {})
        self._populate_list()
        self._update_pagination()

    def _populate_list(self) -> None:
        self.dataset_list.blockSignals(True)
        was_sorting = self.dataset_list.isSortingEnabled()
        self.dataset_list.setSortingEnabled(False)
        self.dataset_list.setRowCount(0)
        self.dataset_list.setRowCount(len(self._datasets))
        for row_idx, ds in enumerate(self._datasets):
            art_id = ds.get("artifact_id", "")
            kind = ds.get("artifact_kind", "")
            fmt = ds.get("data_format") or ""
            sample_name = ds.get("sample_name") or "NA"
            created_at = ds.get("created_at") or ""
            ref_count = ds.get("object_refcount")
            orig_filename = _get_display_name(ds)

            item_id = QtWidgets.QTableWidgetItem(orig_filename)
            item_id.setData(QtCore.Qt.ItemDataRole.UserRole, art_id)
            item_id.setToolTip(f"ID: {art_id}")
            item_kind = QtWidgets.QTableWidgetItem(kind)
            item_fmt = QtWidgets.QTableWidgetItem(fmt)
            item_sample = QtWidgets.QTableWidgetItem(sample_name)
            item_created = QtWidgets.QTableWidgetItem(created_at)
            item_ref = QtWidgets.QTableWidgetItem()
            item_ref.setData(QtCore.Qt.ItemDataRole.DisplayRole, ref_count if ref_count is not None else 0)

            self.dataset_list.setItem(row_idx, 0, item_id)
            self.dataset_list.setItem(row_idx, 1, item_kind)
            self.dataset_list.setItem(row_idx, 2, item_fmt)
            self.dataset_list.setItem(row_idx, 3, item_sample)
            self.dataset_list.setItem(row_idx, 4, item_created)
            self.dataset_list.setItem(row_idx, 5, item_ref)
        self.dataset_list.setSortingEnabled(was_sorting)
        self.dataset_list.blockSignals(False)
        self.status_label.setText(
            f"Showing {len(self._datasets)} of {self._total} datasets"
        )

    def _update_pagination(self) -> None:
        total_pages = max(1, (self._total + self._page_size - 1) // self._page_size)
        self.page_label.setText(f"Page {self._current_page + 1} of {total_pages}")
        self.prev_btn.setEnabled(self._current_page > 0)
        self.next_btn.setEnabled(
            (self._current_page + 1) * self._page_size < self._total
        )
        self.total_label.setText(f"Total: {self._total}")

    # ── slots ──────────────────────────────────────────────────────

    def _on_scope_changed(self, index: int) -> None:
        del index
        self._current_page = 0
        self._refresh()

    def _on_search_text_changed(self, text: str) -> None:
        del text
        if self._search_timer is None:
            self._search_timer = QtCore.QTimer(self)
            self._search_timer.setSingleShot(True)
            self._search_timer.timeout.connect(self._on_search_debounced)
        self._search_timer.start(300)

    def _on_search_debounced(self) -> None:
        self._current_page = 0
        self._refresh()

    def _on_prev_page(self) -> None:
        if self._current_page > 0:
            self._current_page -= 1
            self._refresh()

    def _on_next_page(self) -> None:
        if (self._current_page + 1) * self._page_size < self._total:
            self._current_page += 1
            self._refresh()

    def _on_selection_changed(self) -> None:
        row = self.dataset_list.currentRow()
        if row < 0:
            return
        first_item = self.dataset_list.item(row, 0)
        if not first_item:
            return
        art_id = first_item.data(QtCore.Qt.ItemDataRole.UserRole)
        for ds in self._datasets:
            if ds.get("artifact_id") == art_id:
                display_name = _get_display_name(ds)
                label = f"{display_name}  [{ds.get('artifact_kind', '')}]"
                fmt = ds.get("data_format")
                if fmt:
                    label += f"  ({fmt})"
                selection = DatasetSelection(
                    artifact_id=art_id,
                    artifact_kind=ds.get("artifact_kind", ""),
                    data_format=ds.get("data_format"),
                    label=label,
                    metadata=ds,
                )
                self.datasetSelected.emit(selection)
                return

    def _on_dataset_activated(self, item: QtWidgets.QTableWidgetItem) -> None:
        row = item.row()
        first_item = self.dataset_list.item(row, 0)
        if not first_item:
            return
        art_id = first_item.data(QtCore.Qt.ItemDataRole.UserRole)
        for ds in self._datasets:
            if ds.get("artifact_id") == art_id:
                display_name = _get_display_name(ds)
                label = f"{display_name}  [{ds.get('artifact_kind', '')}]"
                fmt = ds.get("data_format")
                if fmt:
                    label += f"  ({fmt})"
                selection = DatasetSelection(
                    artifact_id=art_id,
                    artifact_kind=ds.get("artifact_kind", ""),
                    data_format=ds.get("data_format"),
                    label=label,
                    metadata=ds,
                )
                self.datasetSelected.emit(selection)
                return

    # ── public API ─────────────────────────────────────────────────

    def selected_dataset(self) -> DatasetSelection | None:
        row = self.dataset_list.currentRow()
        if row < 0:
            return None
        first_item = self.dataset_list.item(row, 0)
        if not first_item:
            return None
        art_id = first_item.data(QtCore.Qt.ItemDataRole.UserRole)
        for ds in self._datasets:
            if ds.get("artifact_id") == art_id:
                display_name = _get_display_name(ds)
                label = f"{display_name}  [{ds.get('artifact_kind', '')}]"
                fmt = ds.get("data_format")
                if fmt:
                    label += f"  ({fmt})"
                return DatasetSelection(
                    artifact_id=art_id,
                    artifact_kind=ds.get("artifact_kind", ""),
                    data_format=ds.get("data_format"),
                    label=label,
                    metadata=ds,
                )
        return None

    def set_kinds(self, kinds: list[str]) -> None:
        self._kinds = kinds
        if self._is_connected():
            self._refresh()

    def set_formats(self, formats: list[str]) -> None:
        self._formats = formats
        if self._is_connected():
            self._refresh()


class MfdbDatasetPickerDialog(QtWidgets.QDialog):
    """Modal dialog wrapping :class:`MfdbDatasetBrowser`.

    Call :meth:`pick_dataset` as a static convenience method.
    """

    def __init__(
        self,
        client: Any = None,
        kinds: list[str] | None = None,
        formats: list[str] | None = None,
        scope: str = "mine",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Dataset from MFDB")
        self.resize(800, 450)
        self._selection: DatasetSelection | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self.browser = MfdbDatasetBrowser(
            client=client,
            kinds=kinds,
            formats=formats,
            parent=self,
        )
        self.browser.datasetSelected.connect(self._on_dataset_selected)
        layout.addWidget(self.browser, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if scope == "mine":
            self.browser.scope_combo.setCurrentIndex(0)
        elif scope == "public":
            self.browser.scope_combo.setCurrentIndex(1)
        elif scope == "all":
            self.browser.scope_combo.setCurrentIndex(2)

    def _on_dataset_selected(self, sel: DatasetSelection) -> None:
        self._selection = sel

    def _on_accept(self) -> None:
        if self._selection is None:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a dataset."
            )
            return
        self.accept()

    def selected_dataset(self) -> DatasetSelection | None:
        return self._selection

    @staticmethod
    def pick_dataset(
        parent: QtWidgets.QWidget | None = None,
        kinds: list[str] | None = None,
        formats: list[str] | None = None,
        scope: str = "mine",
        client: Any = None,
    ) -> DatasetSelection | None:
        """Show a modal dataset picker and return the selection.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        kinds : list of str, optional
            Pre-set artifact kind filter.
        formats : list of str, optional
            Pre-set data format filter.
        scope : str, default='mine'
            Initial scope (``'mine'``, ``'public'``, or ``'all'``).
        client : object, optional
            RPC client. When omitted or ``None``, the dialog returns
            ``None`` immediately (MFDB not connected).

        Returns
        -------
        DatasetSelection or None
            The selected dataset, or ``None`` if cancelled or not connected.
        """
        if client is None:
            return None
        dialog = MfdbDatasetPickerDialog(
            client=client,
            kinds=kinds,
            formats=formats,
            scope=scope,
            parent=parent,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return dialog.selected_dataset()
