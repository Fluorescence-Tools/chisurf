"""Fluorophore database dock widget with auto-generated forms."""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.general import apply_compact_table_style

from .probe_form import ProbeDetailForm
from .spectrum_view import SpectrumView


class FluorophoreDock(QtWidgets.QWidget):
    """Autoform-based dock for browsing, editing, and approving fluorophores.

    Combines a filterable probe table, an MFDBDetailWidget form, and a
    spectrum viewer in a vertical splitter layout.
    """

    statusMessage = QtCore.Signal(str)

    PROBE_COLUMNS = [
        ("ID", "probe_id"),
        ("Name", "chromophore_name"),
        ("Type", "type_name"),
        ("Abs max", "abs_max"),
        ("Em max", "em_max"),
        ("QY", "qy"),
        ("Status", "verification_status"),
        ("Quality", "quality"),
        ("Source", "source"),
    ]

    def __init__(
        self,
        client: Any,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._client = client
        self._probe_cache: dict[str, dict[str, Any]] = {}

        self._setup_ui()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

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
        table_layout.addWidget(self._table)
        splitter.addWidget(table_container)

        # --- Spectrum view ---
        self._spectrum_view = SpectrumView()
        splitter.addWidget(self._spectrum_view)

        # --- Form (AutoForm + JSON view scheme, PRD-40) ---
        self._form = ProbeDetailForm(parent=self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form)
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
        self._import_btn.setToolTip("Import fluorophore data from the bundled reference database")
        self._import_btn.clicked.connect(self._on_import)
        layout.addWidget(self._import_btn)

        self._approve_btn = QtWidgets.QToolButton()
        self._approve_btn.setText("✅ Approve")
        self._approve_btn.setToolTip("Mark the selected probe as approved")
        self._approve_btn.clicked.connect(self._on_approve)
        layout.addWidget(self._approve_btn)

        self._reject_btn = QtWidgets.QToolButton()
        self._reject_btn.setText("❌ Reject")
        self._reject_btn.setToolTip("Mark the selected probe as rejected")
        self._reject_btn.clicked.connect(self._on_reject)
        layout.addWidget(self._reject_btn)

        self._review_btn = QtWidgets.QToolButton()
        self._review_btn.setText("🔍 Review queue")
        self._review_btn.setToolTip("Filter to unverified probes needing review")
        self._review_btn.setCheckable(True)
        self._review_btn.clicked.connect(self._on_review_queue)
        layout.addWidget(self._review_btn)

        self._ai_btn = QtWidgets.QToolButton()
        self._ai_btn.setText("🤖 AI triage")
        self._ai_btn.setToolTip("Run deterministic checks on the selected probe")
        self._ai_btn.clicked.connect(self._on_ai_triage)
        layout.addWidget(self._ai_btn)

        layout.addStretch()
        return widget

    def _build_filter_bar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QtWidgets.QLabel("Search:"))
        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Probe name...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.returnPressed.connect(self.refresh)
        layout.addWidget(self._search_edit)

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

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            params: dict[str, Any] = {
                "limit": 500,
                "offset": 0,
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
            self._set_status(f"{len(probes)} probes (of {result.get('total', '?')} total)")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    def _populate_table(self, probes: list[dict[str, Any]]) -> None:
        headers = ["✓"] + [c[1] for c in self.PROBE_COLUMNS]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(probes))

        for row, p in enumerate(probes):
            cb = QtWidgets.QTableWidgetItem("")
            cb.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            cb.setCheckState(QtCore.Qt.Unchecked)
            cb.setTextAlignment(QtCore.Qt.AlignCenter)
            self._table.setItem(row, 0, cb)

            vals = [
                str(p.get(col, ""))
                for col in ("probe_id", "chromophore_name", "type_name",
                            "abs_max", "em_max", "qy",
                            "verification_status", "quality", "source")
            ]
            for col, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(val)
                if col == 6:  # Status column
                    colors = {
                        "approved": QtGui.QColor(214, 245, 220),
                        "rejected": QtGui.QColor(255, 220, 220),
                        "unverified": QtGui.QColor(255, 244, 204),
                        "needs_review": QtGui.QColor(255, 235, 200),
                    }
                    bg = colors.get(val.lower())
                    if bg:
                        item.setBackground(bg)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self._table.setItem(row, col + 1, item)

    # ------------------------------------------------------------------
    # Selection → form + spectra auto-fill
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._loading:
            return
        probe_id = self._selected_probe_id()
        if not probe_id:
            return

        # Load into form
        data = self._probe_cache.get(probe_id, {})
        if data:
            self._form.set_data(data)

        # Load spectra
        try:
            detail = self._client._call("fluorophores.get", {"probe_id": int(probe_id)})
            self._spectrum_view.display(detail)
        except Exception as exc:
            self._spectrum_view.clear()
            self._set_status(f"Spectra load failed: {exc}")

    def _on_spectrum_double_click(self, row: int, col: int) -> None:
        """Double-clicking anywhere opens the spectrum popup."""
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Import reference set",
            "Import fluorophore data from the bundled reference database?\n\n"
            "All imported probes will be marked as unverified.",
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
            self._set_status(f"Approved probe {probe_id}")
            self.refresh()
        except Exception as exc:
            self._set_status(f"Approve failed: {exc}")

    def _on_reject(self) -> None:
        probe_id = self._selected_probe_id()
        if not probe_id:
            return
        try:
            self._client._call("fluorophores.reject", {"probe_id": int(probe_id), "verified_by": "admin"})
            self._set_status(f"Rejected probe {probe_id}")
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
    # Helpers
    # ------------------------------------------------------------------

    def _selected_probe_id(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a probe first.")
            return None
        item = self._table.item(row, 1)
        return item.text().strip() if item else None

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self.statusMessage.emit(msg)

    def selected_row_id(self) -> str | None:
        return self._selected_probe_id()
