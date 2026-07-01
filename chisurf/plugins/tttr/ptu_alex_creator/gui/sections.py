"""Custom AutoForm sections for the ALEX Creator tool.

The drag-drop file loader plus the Load/Save action buttons (``alex_actions``)
are a bespoke Qt widget registered here; the format and ALEX-period controls are
plain built-in sections in ``alex.view.json`` and the histogram is a declarative
``plot`` section. The widget owns only Qt concerns and drives the Qt-free
:class:`~..view_model.AlexViewModel`. Imported (and registered) by ``gui.tool``.
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


@register_section("alex_actions")
def alex_actions(model, target=None, **options):
    """AutoForm factory for the ALEX file loader and Load/Save buttons."""
    return _ActionsSection(model)


class _ActionsSection(QtWidgets.QWidget):
    """Drag-drop file row with Load and Save tool-buttons."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self._edit = QtWidgets.QLineEdit()
        self._edit.setPlaceholderText("Drop a TTTR file here or browse…")
        self._edit.setAcceptDrops(True)
        self._edit.dragEnterEvent = self._drag_enter
        self._edit.dropEvent = self._drop
        self._edit.editingFinished.connect(lambda: self._load(self._edit.text().strip()))
        row.addWidget(self._edit, 1)
        row.addWidget(_tool_button("📂 Load", "Load a TTTR file.", self._browse))
        self._save_btn = _tool_button("💾 Save", "Save the ALEX-converted file.", self._save)
        self._save_btn.setEnabled(False)
        row.addWidget(self._save_btn)
        layout.addLayout(row)

        self._model.add_observer(self._on_model_event)

    # ── drag-drop ───────────────────────────────────────────────────────
    def _drag_enter(self, event: QtGui.QDragEnterEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1 and pathlib.Path(urls[0].toLocalFile()).is_file():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1:
            path = urls[0].toLocalFile()
            if pathlib.Path(path).is_file():
                event.acceptProposedAction()
                self._load(path)
                return
        event.ignore()

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        if event == "loaded":
            if self._edit.text() != self._model.input_file:
                self._edit.blockSignals(True)
                self._edit.setText(self._model.input_file)
                self._edit.blockSignals(False)
            self._save_btn.setEnabled(self._model.has_data)

    # ── actions ─────────────────────────────────────────────────────────
    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open TTTR file")
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        if not path:
            return
        if not pathlib.Path(path).is_file():
            QtWidgets.QMessageBox.warning(self, "Invalid file", f"'{path}' is not a valid file.")
            return
        try:
            import tttrlib

            from chisurf.gui.widgets.staged_loading import load_with_progress

            tttr_type = self._model.tttr_filetype

            def _read(local_path):
                return tttrlib.TTTR(local_path, tttr_type)

            tttr = load_with_progress(self, _read, str(path), title="Loading TTTR file")
            if tttr is None:
                return
            self._model.set_tttr(tttr, str(path))
            self._model.notify("plot")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot load file:\n{exc}")

    def _save(self) -> None:
        reason = self._model.can_save()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Cannot save", reason)
            return
        start = str(pathlib.Path(self._model.input_file).with_name(self._model.default_save_name()))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ALEX file", start)
        if not path:
            return
        try:
            self._model.save(path)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot save:\n{exc}")


__all__ = ["alex_actions"]
