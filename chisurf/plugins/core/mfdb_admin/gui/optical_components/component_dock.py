"""Optical component manager dock widget with auto-generated forms.

Supports fluorophores, filters, dichroics, detectors, and light sources,
swappable via radio tabs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.general import apply_compact_table_style

from .component_detail_form import ComponentDetailForm
from chisurf.gui.widgets.spectrum_view import SpectrumView

_PROPERTY_MAP = {
    "Cut-On Wavelength (nm)": "cut_on",
    "Cut-Off Wavelength (nm)": "cut_off",
    "Center Wavelength (nm)": "center_wavelength",
    "Bandwidth (nm)": "bandwidth",
    "Optical Density": "optical_density",
}


class OpticalComponentDock(QtWidgets.QWidget):
    """Autoform-based dock for browsing, editing, and approving optical components.

    Combines a component-type radio button selector, a filterable component table,
    a dynamic detail form, and a spectrum viewer in a vertical splitter layout.
    """

    statusMessage = QtCore.Signal(str)

    def __init__(
        self,
        client: Any,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._client = client
        self._probe_cache: dict[str, dict[str, Any]] = {}
        self._detail_cache: dict[str, dict[str, Any]] = {}
        self._active_component: dict[str, Any] = {}

        self._load_registry()
        self._setup_ui()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _load_registry(self) -> None:
        """Load the component type registry from components.json."""
        registry_path = Path(__file__).resolve().parent / "components.json"
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                self._registry = json.load(f)
        except Exception as e:
            # Fallback registry if file fails to load
            print(f"Error loading components.json: {e}")
            self._registry = [
                {
                    "key": "fluorophore",
                    "label": "Fluorophores",
                    "icon": "🌈",
                    "categories": ["fluorophore", "organic_dye", "protein", "other"],
                    "columns": [
                        ["ID", "probe_id"],
                        ["Name", "chromophore_name"],
                        ["Type", "type_name"],
                        ["Abs max", "abs_max"],
                        ["Em max", "em_max"],
                        ["QY", "qy"],
                        ["Status", "verification_status"],
                        ["Quality", "quality"],
                        ["Source", "source"]
                    ],
                    "view": "fluorophore.view.json",
                }
            ]
        self._active_component = self._registry[0]

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

        # Radio button row (switches components)
        type_selector = self._build_type_selector()
        layout.addWidget(type_selector)

        # Splitter: table top, spectra middle, form bottom
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        # --- Filter + Table ---
        table_container = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(2)

        filter_bar = self._build_filter_bar()
        table_layout.addWidget(filter_bar)

        self._table = QtWidgets.QTableWidget()
        apply_compact_table_style(self._table)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_spectrum_double_click)
        self._table.itemChanged.connect(self._on_item_changed)
        table_layout.addWidget(self._table)
        self._install_context_menu()
        splitter.addWidget(table_container)

        # --- Spectrum view ---
        self._spectrum_view = SpectrumView()
        splitter.addWidget(self._spectrum_view)

        # --- Form (AutoForm + JSON view schemes, PRD-40) ---
        self._form_stack = QtWidgets.QStackedWidget()
        self._forms = {}
        for item in self._registry:
            key = item["key"]
            view_file = item["view"]
            view_path = Path(__file__).resolve().parent / view_file
            form = ComponentDetailForm(view_path, parent=self)
            self._form_stack.addWidget(form)
            self._forms[key] = form

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form_stack)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        self._status_label = QtWidgets.QLabel()
        layout.addWidget(self._status_label)

        self._loading = False

    def _build_toolbar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._import_btn = QtWidgets.QToolButton()
        self._import_btn.setText("📥 Import ref. set")
        self._import_btn.setToolTip("Import reference data from the bundled reference database")
        self._import_btn.clicked.connect(self._on_import)
        layout.addWidget(self._import_btn)

        self._approve_btn = QtWidgets.QToolButton()
        self._approve_btn.setText("✅ Approve")
        self._approve_btn.setToolTip("Mark the selected item as approved")
        self._approve_btn.clicked.connect(self._on_approve)
        layout.addWidget(self._approve_btn)

        self._reject_btn = QtWidgets.QToolButton()
        self._reject_btn.setText("❌ Reject")
        self._reject_btn.setToolTip("Mark the selected item as rejected")
        self._reject_btn.clicked.connect(self._on_reject)
        layout.addWidget(self._reject_btn)

        self._review_btn = QtWidgets.QToolButton()
        self._review_btn.setText("🔍 Review queue")
        self._review_btn.setToolTip("Filter to unverified items needing review")
        self._review_btn.setCheckable(True)
        self._review_btn.clicked.connect(self._on_review_queue)
        layout.addWidget(self._review_btn)

        self._ai_btn = QtWidgets.QToolButton()
        self._ai_btn.setText("🤖 AI triage")
        self._ai_btn.setToolTip("Run deterministic checks on the selected item")
        self._ai_btn.clicked.connect(self._on_ai_triage)
        layout.addWidget(self._ai_btn)

        layout.addStretch()
        return widget

    def _build_type_selector(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Component Type:"))

        self._button_group = QtWidgets.QButtonGroup(self)
        self._button_group.setExclusive(True)

        for idx, item in enumerate(self._registry):
            rb = QtWidgets.QRadioButton(f"{item['icon']} {item['label']}")
            if idx == 0:
                rb.setChecked(True)
            layout.addWidget(rb)
            self._button_group.addButton(rb, idx)

        self._button_group.buttonClicked.connect(self._on_button_clicked)
        layout.addStretch()
        return widget

    def _build_filter_bar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QtWidgets.QLabel("Search:"))
        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Name...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self.refresh)
        self._search_edit.textChanged.connect(self._on_search_text)
        self._search_edit.returnPressed.connect(self._search_now)
        layout.addWidget(self._search_edit)

        self._auto_check = QtWidgets.QCheckBox("Auto")
        self._auto_check.setChecked(True)
        layout.addWidget(self._auto_check)

        layout.addWidget(QtWidgets.QLabel("Status:"))
        self._status_combo = QtWidgets.QComboBox()
        self._status_combo.addItems(["all", "approved", "unverified", "rejected", "needs_review"])
        self._status_combo.currentTextChanged.connect(lambda _: self.refresh())
        layout.addWidget(self._status_combo)

        refresh_btn = QtWidgets.QToolButton()
        refresh_btn.setText("🔄")
        refresh_btn.setToolTip("Refresh list")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)

        return widget

    def _on_button_clicked(self, button: QtWidgets.QAbstractButton) -> None:
        idx = self._button_group.id(button)
        self._active_component = self._registry[idx]

        # Update Detail Form stack
        active_key = self._active_component["key"]
        self._form_stack.setCurrentWidget(self._forms[active_key])

        # Clear caches, spectra and details form
        self._detail_cache.clear()
        self._spectrum_view.clear()
        self._forms[active_key].clear()

        # Re-query data
        self.refresh()

    def _on_search_text(self, _text: str) -> None:
        if self._auto_check.isChecked():
            self._search_timer.start()

    def _search_now(self) -> None:
        self._search_timer.stop()
        self.refresh()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._detail_cache.clear()
        try:
            categories = self._active_component["categories"]
            params: dict[str, Any] = {
                "limit": 500,
                "offset": 0,
                "category": categories,
            }
            status = self._status_combo.currentText()
            if status != "all":
                params["verification_status"] = status
            search = self._search_edit.text().strip()
            if search:
                params["search"] = search

            result = self._client._call("fluorophores.list", params)
            probes = result.get("probes", [])
            self._probe_cache = {str(p["probe_id"]): p for p in probes}
            self._populate_table(probes)
            self._set_status(f"{len(probes)} items (of {result.get('total', '?')} total)")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    def _populate_table(self, probes: list[dict[str, Any]]) -> None:
        self._loading = True
        self._table.setSortingEnabled(False)
        self._table.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        try:
            columns = self._active_component["columns"]
            headers = ["✓"] + [c[0] for c in columns]
            self._table.setColumnCount(len(headers))
            self._table.setHorizontalHeaderLabels(headers)
            
            # Temporarily disable ResizeToContents mode during populating to avoid layout recalculation overhead
            header = self._table.horizontalHeader()
            for col in range(self._table.columnCount()):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.Interactive)

            self._table.setRowCount(len(probes))

            for row, p in enumerate(probes):
                cb = QtWidgets.QTableWidgetItem("")
                cb.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                cb.setCheckState(QtCore.Qt.Unchecked)
                cb.setTextAlignment(QtCore.Qt.AlignCenter)
                self._table.setItem(row, 0, cb)

                for col_idx, (_, col_key) in enumerate(columns):
                    val = p.get(col_key, "")
                    if val is None:
                        val = ""
                    item = QtWidgets.QTableWidgetItem(str(val))
                    if col_key == "verification_status":
                        colors = {
                            "approved": QtGui.QColor(214, 245, 220),
                            "rejected": QtGui.QColor(255, 220, 220),
                            "unverified": QtGui.QColor(255, 244, 204),
                            "needs_review": QtGui.QColor(255, 235, 200),
                        }
                        bg = colors.get(str(val).lower())
                        if bg:
                            item.setBackground(bg)
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self._table.setItem(row, col_idx + 1, item)
        finally:
            self._table.blockSignals(False)
            self._table.setUpdatesEnabled(True)
            self._table.setSortingEnabled(True)
            self._size_columns()
            self._loading = False

    def _size_columns(self) -> None:
        """Sane column widths: narrow checkbox, stretchy Name, fit the rest."""
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        Modes = QtWidgets.QHeaderView
        for col in range(self._table.columnCount()):
            header.setSectionResizeMode(col, Modes.ResizeToContents)
        # Checkbox column: fixed and narrow.
        header.setSectionResizeMode(0, Modes.Fixed)
        self._table.setColumnWidth(0, 28)

        # Name column takes the slack.
        name_idx = -1
        columns = self._active_component["columns"]
        for col_idx, (_, col_key) in enumerate(columns):
            if col_key == "chromophore_name":
                name_idx = col_idx + 1
                break

        if name_idx != -1:
            header.setSectionResizeMode(name_idx, Modes.Stretch)

    # ------------------------------------------------------------------
    # Selection → form + spectra auto-fill
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._loading:
            return
        probe_id = self._selected_probe_id()
        if not probe_id:
            return

        try:
            detail = self._client._call("fluorophores.get", {"probe_id": int(probe_id)})
            self._detail_cache[probe_id] = detail

            # Merge optical_properties into the probe dict
            probe_data = dict(detail.get("probe", {}))
            for prop in detail.get("optical_properties", []):
                pname = prop.get("property_name", "")
                pval = prop.get("property_value", "")

                clean_name = _PROPERTY_MAP.get(pname, pname)
                probe_data[clean_name] = pval

            active_key = self._active_component["key"]
            self._forms[active_key].set_data(probe_data)

            # Show spectrum: checked overlay if any, otherwise single probe
            checked_ids = self._checked_probe_ids()
            if checked_ids:
                self._update_checked_spectra()
            else:
                self._spectrum_view.display(detail)
        except Exception as exc:
            self._spectrum_view.clear()
            self._set_status(f"Load failed: {exc}")

    def _on_spectrum_double_click(self, row: int, col: int) -> None:
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Import reference set",
            "Import reference set data from the bundled database?\n\n"
            "All imported items will be marked as unverified.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            result = self._client._call("fluorophores.import_reference_set", {"mark_verified": False})
            self._set_status(
                f"Imported: {result.get('probes', 0)} probes, "
                f"{result.get('spectra', 0)} spectra, "
                f"{result.get('optical_properties', 0)} optical properties"
            )
            self.refresh()
        except Exception as exc:
            self._set_status(f"Import failed: {exc}")

    def _on_approve(self) -> None:
        probe_id = self._selected_probe_id()
        if not probe_id:
            return
        try:
            self._client._call("fluorophores.approve", {"probe_id": int(probe_id), "verified_by": "admin"})
            self._set_status(f"Approved item {probe_id}")
            self.refresh()
        except Exception as exc:
            self._set_status(f"Approve failed: {exc}")

    def _on_reject(self) -> None:
        probe_id = self._selected_probe_id()
        if not probe_id:
            return
        try:
            self._client._call("fluorophores.reject", {"probe_id": int(probe_id), "verified_by": "admin"})
            self._set_status(f"Rejected item {probe_id}")
            self.refresh()
        except Exception as exc:
            self._set_status(f"Reject failed: {exc}")

    def _on_review_queue(self, checked: bool) -> None:
        if checked:
            self._status_combo.setCurrentText("unverified")
        else:
            self._status_combo.setCurrentText("all")
        self.refresh()

    def _on_ai_triage(self) -> None:
        probe_id = self._selected_probe_id()
        if not probe_id:
            return
        try:
            result = self._client._call("fluorophores.ai_triage", {"probe_id": int(probe_id)})
            issues = result.get("issues", [])
            quality = result.get("proposed_quality", "unknown")
            if issues:
                msg = (
                    "Issues found (queued for review):\n  • "
                    + "\n  • ".join(issues)
                )
            else:
                msg = (
                    f"No issues found — proposed quality: {quality}.\n\n"
                    "Queued for review; use Approve to confirm."
                )
            QtWidgets.QMessageBox.information(self, "AI Triage Result", msg)
            self.refresh()
        except Exception as exc:
            self._set_status(f"AI triage failed: {exc}")

    # ------------------------------------------------------------------
    # Checkbox-driven multi-spectra overlay
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """React to checkbox toggles in column 0."""
        if self._loading:
            return
        if item.column() != 0:
            return
        # Defer to next event-loop iteration so the check-state is settled
        QtCore.QTimer.singleShot(0, self._update_checked_spectra)

    def _update_checked_spectra(self) -> None:
        """Gather detail data for all checked probes and overlay their spectra."""
        checked_ids = self._checked_probe_ids()
        if not checked_ids:
            # Fall back to single-selection display
            probe_id = self._selected_probe_id()
            if probe_id and probe_id in self._detail_cache:
                self._spectrum_view.display(self._detail_cache[probe_id])
            elif probe_id:
                try:
                    detail = self._client._call("fluorophores.get", {"probe_id": int(probe_id)})
                    self._detail_cache[probe_id] = detail
                    self._spectrum_view.display(detail)
                except Exception:
                    self._spectrum_view.clear()
            else:
                self._spectrum_view.clear()
            return

        details: list[dict[str, Any]] = []
        for pid in checked_ids:
            if pid in self._detail_cache:
                details.append(self._detail_cache[pid])
            else:
                try:
                    detail = self._client._call("fluorophores.get", {"probe_id": int(pid)})
                    self._detail_cache[pid] = detail
                    details.append(detail)
                except Exception:
                    continue

        if details:
            self._spectrum_view.display_multiple(details)
        else:
            self._spectrum_view.clear()

    def _checked_probe_ids(self) -> list[str]:
        """Return probe IDs of all checked rows."""
        ids: list[str] = []
        for row in range(self._table.rowCount()):
            cb = self._table.item(row, 0)
            if cb is not None and cb.checkState() == QtCore.Qt.Checked:
                id_item = self._table.item(row, 1)
                if id_item is not None:
                    ids.append(id_item.text().strip())
        return ids

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _install_context_menu(self) -> None:
        """Attach a right-click context menu to the table."""
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        """Build and show the right-click context menu."""
        menu = QtWidgets.QMenu(self._table)
        current_row = self._table.rowAt(pos.y())
        has_row = current_row >= 0 and current_row < self._table.rowCount()

        if has_row:
            menu.addAction(
                "🔍 Open details",
                lambda: self._open_row_details(current_row),
            )
            menu.addSeparator()

        menu.addAction("📋 Copy checked IDs", self._copy_checked_ids)
        menu.addAction("📋 Copy selected row", self._copy_selected_row)
        menu.addAction("📋 Copy selected cell", self._copy_selected_cell)
        menu.addSeparator()
        menu.addAction("☑ Check selected rows", lambda: self._set_selected_checks(True))
        menu.addAction("☐ Uncheck selected rows", lambda: self._set_selected_checks(False))
        menu.addAction("☑ Check all visible", lambda: self._set_all_checks(True))
        menu.addAction("☐ Uncheck all", lambda: self._set_all_checks(False))
        menu.addAction("🔁 Invert visible checks", self._invert_checks)
        menu.addSeparator()
        menu.addAction("⬛ Select all rows", self._table.selectAll)
        menu.addAction("🔳 Clear selection", self._table.clearSelection)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _open_row_details(self, row: int) -> None:
        """Select a row by index and trigger the detail-form load."""
        self._table.setCurrentCell(row, 0)
        self._on_selection_changed()

    def _copy_checked_ids(self) -> None:
        """Copy comma-separated IDs of checked rows to clipboard."""
        ids = self._checked_probe_ids()
        if ids:
            QtWidgets.QApplication.clipboard().setText(", ".join(ids))
            self._set_status(f"Copied {len(ids)} IDs")
        else:
            self._set_status("No checked items")

    def _copy_selected_row(self) -> None:
        """Copy selected row as tab-separated text."""
        selected = self._table.selectedItems()
        if not selected:
            return
        rows: dict[int, list[str]] = {}
        for item in selected:
            row = item.row()
            if row not in rows:
                rows[row] = []
            rows[row].append(item.text())
        lines = ["\t".join(cols) for cols in rows.values()]
        if lines:
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _copy_selected_cell(self) -> None:
        """Copy the current cell's text."""
        item = self._table.currentItem()
        if item is not None:
            QtWidgets.QApplication.clipboard().setText(item.text())

    def _set_selected_checks(self, checked: bool) -> None:
        """Check or uncheck only the selected rows."""
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        selected = self._table.selectedItems()
        rows = set(item.row() for item in selected)
        for row in rows:
            cb = self._table.item(row, 0)
            if cb is not None and cb.flags() & QtCore.Qt.ItemIsUserCheckable:
                cb.setCheckState(state)

    def _set_all_checks(self, checked: bool) -> None:
        """Check or uncheck every row in the table."""
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(self._table.rowCount()):
            cb = self._table.item(row, 0)
            if cb is not None and cb.flags() & QtCore.Qt.ItemIsUserCheckable:
                cb.setCheckState(state)

    def _invert_checks(self) -> None:
        """Invert the check state of every row."""
        for row in range(self._table.rowCount()):
            cb = self._table.item(row, 0)
            if cb is not None and cb.flags() & QtCore.Qt.ItemIsUserCheckable:
                current = cb.checkState()
                cb.setCheckState(
                    QtCore.Qt.Unchecked if current == QtCore.Qt.Checked else QtCore.Qt.Checked
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_probe_id(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "No selection", "Select an item first.")
            return None
        item = self._table.item(row, 1)
        return item.text().strip() if item else None

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self.statusMessage.emit(msg)

    def selected_row_id(self) -> str | None:
        return self._selected_probe_id()
