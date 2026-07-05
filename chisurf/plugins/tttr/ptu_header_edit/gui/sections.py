"""Custom AutoForm section for the PTU Header Editor.

The editable tag table (Name / Type / Value / Idx) plus the Open/Add/Remove/Save
tool-buttons are registered here as the ``header_table`` section; the read-only
JSON preview is a plain built-in ``value`` (kind ``text``) section in
``header.view.json``. The widget owns Qt concerns and drives the Qt-free
:class:`~..view_model.HeaderEditorViewModel`. Imported (registered) by ``gui.tool``.
"""

from __future__ import annotations

import logging
import pathlib

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

logger = logging.getLogger(__name__)


def _tool_button(text: str, tooltip: str, slot) -> QtWidgets.QToolButton:
    """Build a configured ``QToolButton`` (emoji label + tooltip) in one call."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.clicked.connect(slot)
    return btn


@register_section("header_table")
def header_table(model, target=None, **options):
    """AutoForm factory for the PTU header tag table and its action buttons."""
    return _HeaderTableSection(model)


class _HeaderTableSection(QtWidgets.QWidget):
    """Editable PTU tag table with Open/Add/Remove/Save tool-buttons + drag-drop."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._syncing = False
        self.setAcceptDrops(True)
        # Fill spare vertical space so the tag table grows with its panel; the
        # AutoForm reads ``_autoform_expanding`` to hand this section the stretch.
        self._autoform_expanding = True
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(_tool_button("📂 Open", "Open a PTU file and read its header.", self._open))
        bar.addWidget(_tool_button("➕ Add", "Add a new header tag.", self._add))
        bar.addWidget(_tool_button("➖ Remove", "Remove the selected tag.", self._remove))
        bar.addStretch(1)
        bar.addWidget(_tool_button("💾 Save", "Save a modified PTU with these tags.", self._save))
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Value", "Idx"])
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(lambda *_: self._sync_to_model())
        layout.addWidget(self.table, 1)

        self._model.add_observer(self._on_model_event)
        self._rebuild_table()

    # ── drag-drop ───────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1 and pathlib.Path(urls[0].toLocalFile()).is_file():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls:
            self._load_ptu(urls[0].toLocalFile())
            event.acceptProposedAction()

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        if event == "loaded":
            self._rebuild_table()

    def _rebuild_table(self) -> None:
        self._syncing = True
        self.table.setRowCount(len(self._model.tags))
        for row, tag in enumerate(self._model.tags):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(tag.get("name", ""))))
            combo = QtWidgets.QComboBox()
            combo.addItems(self._model.TYPE_MAPPING.values())
            combo.setCurrentText(self._model.TYPE_MAPPING.get(tag.get("type"), ""))
            combo.currentTextChanged.connect(lambda *_: self._sync_to_model())
            self.table.setCellWidget(row, 1, combo)
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(tag.get("value", ""))))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(tag.get("idx", -1))))
        self._syncing = False

    def _read_rows(self) -> list[dict]:
        rows = []
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, 0)
            combo = self.table.cellWidget(r, 1)
            value_item = self.table.item(r, 2)
            idx_item = self.table.item(r, 3)
            rows.append(
                {
                    "name": name_item.text() if name_item else "",
                    "type": combo.currentText() if combo else "",
                    "value": value_item.text() if value_item else "",
                    "idx": idx_item.text() if idx_item else "-1",
                }
            )
        return rows

    def _sync_to_model(self) -> None:
        if self._syncing:
            return
        self._model.set_tags(self._read_rows())

    # ── actions ─────────────────────────────────────────────────────────
    def _open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open PTU File", "", "PTU Files (*.ptu);;All Files (*)"
        )
        if path:
            self._load_ptu(path)

    def _load_ptu(self, path: str) -> None:
        try:
            self._model.load_ptu(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open file:\n{exc}")

    def _add(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Tag", "Enter tag name:")
        if not ok or not name:
            return
        value, ok = QtWidgets.QInputDialog.getText(self, "Add Tag", "Enter tag value:")
        if not ok:
            return
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Add Tag", "Enter tag idx:", -1)
        if not ok:
            return
        row = self.table.rowCount()
        self._syncing = True
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        combo = QtWidgets.QComboBox()
        combo.addItems(self._model.TYPE_MAPPING.values())
        combo.currentTextChanged.connect(lambda *_: self._sync_to_model())
        self.table.setCellWidget(row, 1, combo)
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(value))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(idx)))
        self._syncing = False
        self._sync_to_model()

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No tag selected to remove.")
            return
        self.table.removeRow(row)
        self._sync_to_model()

    def _save(self) -> None:
        reason = self._model.can_save()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Cannot save", reason)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Modified PTU File", "", "PTU Files (*.ptu);;All Files (*)"
        )
        if not path:
            return
        try:
            self._model.save(path)
            QtWidgets.QMessageBox.information(self, "Success", "Modified PTU file saved.")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save:\n{exc}")


__all__ = ["header_table"]
