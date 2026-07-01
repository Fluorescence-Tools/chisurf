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
            from chisurf.gui.widgets.staged_loading import load_with_progress

            from .. import core

            tttr_type = core.resolve_filetype(self._model.input_format, path)

            def _read(local_path):
                return core.load(local_path, tttr_type)

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


# ---------------------------------------------------------------------------
# alex_batch — batch convert / merge of many (.sm) files
# ---------------------------------------------------------------------------


@register_section("alex_batch")
def alex_batch(model, target=None, **options):
    """AutoForm factory for the batch convert/merge file list and controls."""
    return _BatchSection(model)


class _BatchSection(QtWidgets.QWidget):
    """Drag-drop list of ALEX files batch-converted or merged with current settings."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._running = False
        self.setAcceptDrops(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        info = QtWidgets.QLabel("Drop .sm (or other TTTR) files to convert each or merge into one.")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self._list, 1)

        out_row = QtWidgets.QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.addWidget(QtWidgets.QLabel("Output folder"))
        self._out_edit = QtWidgets.QLineEdit()
        self._out_edit.setPlaceholderText("Output folder (for Convert / merged file)")
        self._out_edit.editingFinished.connect(
            lambda: setattr(self._model, "batch_output_folder", self._out_edit.text().strip())
        )
        out_row.addWidget(self._out_edit, 1)
        out_browse = QtWidgets.QToolButton()
        out_browse.setText("…")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(_tool_button("➕ Files", "Add TTTR files.", self._add_files))
        btn_row.addWidget(_tool_button("🗑 Clear", "Clear the list.", self._model.clear_batch))
        btn_row.addStretch(1)
        btn_row.addWidget(_tool_button("⚙️ Run batch", "Convert each / merge into one.", self._run))
        layout.addLayout(btn_row)

        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        self._model.add_observer(self._on_model_event)

    # ── drag-drop ───────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [
            u.toLocalFile()
            for u in (event.mimeData().urls() or [])
            if u.toLocalFile() and pathlib.Path(u.toLocalFile()).is_file()
        ]
        if paths:
            self._model.add_batch_files(paths)
            event.acceptProposedAction()

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        if event == "batch":
            self._list.clear()
            for p in self._model.batch_files:
                self._list.addItem(p)

    # ── actions ─────────────────────────────────────────────────────────
    def _add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add ALEX files", "", "TTTR/SM files (*.sm *.ptu *.ht3 *.spc);;All Files (*)"
        )
        if paths:
            self._model.add_batch_files(list(paths))

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._out_edit.setText(folder)
            self._model.batch_output_folder = folder

    def _run(self) -> None:
        if self._running:
            return
        reason = self._model.can_run_batch()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Cannot run batch", reason)
            return
        self._running = True
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            outputs = self._model.run_batch()
            self._status.setText(f"Wrote {len(outputs)} file(s).")
            QtWidgets.QMessageBox.information(
                self, "Batch complete", f"Wrote {len(outputs)} file(s)."
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Batch failed", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._running = False


__all__ = ["alex_actions", "alex_batch"]
