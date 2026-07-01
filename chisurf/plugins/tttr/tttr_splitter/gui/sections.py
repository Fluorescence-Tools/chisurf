"""Custom AutoForm sections for the TTTR Split / Convert tool.

The drag-drop file pickers (``splitter_io``), the run button + progress bar
(``splitter_run``) and the batch file list (``splitter_batch``) are bespoke Qt
widgets registered here; the option controls are plain built-in sections in
``splitter.view.json``. The widgets own only Qt concerns and drive the Qt-free
:class:`~..view_model.SplitterViewModel`. Imported (and thus registered) by
``gui.tool``. Mirrors the PSF tool's ``psf_controls`` pattern.
"""

from __future__ import annotations

import logging
import pathlib

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drag-drop helpers
# ---------------------------------------------------------------------------


def _enable_file_drop(line_edit: QtWidgets.QLineEdit, on_file) -> None:
    """Enable dropping a single existing file onto *line_edit* → ``on_file(path)``."""
    line_edit.setAcceptDrops(True)

    def dragEnterEvent(event: QtGui.QDragEnterEvent):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1 and pathlib.Path(urls[0].toLocalFile()).is_file():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(event: QtGui.QDropEvent):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1:
            path = urls[0].toLocalFile()
            if pathlib.Path(path).is_file():
                event.acceptProposedAction()
                on_file(path)
                return
        event.ignore()

    line_edit.dragEnterEvent = dragEnterEvent
    line_edit.dropEvent = dropEvent


def _enable_folder_drop(line_edit: QtWidgets.QLineEdit, on_folder) -> None:
    """Enable dropping a single existing folder onto *line_edit* → ``on_folder(path)``."""
    line_edit.setAcceptDrops(True)

    def dragEnterEvent(event: QtGui.QDragEnterEvent):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1 and pathlib.Path(urls[0].toLocalFile()).is_dir():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(event: QtGui.QDropEvent):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1:
            folder = urls[0].toLocalFile()
            if pathlib.Path(folder).is_dir():
                event.acceptProposedAction()
                on_folder(folder)
                return
        event.ignore()

    line_edit.dragEnterEvent = dragEnterEvent
    line_edit.dropEvent = dropEvent


# ---------------------------------------------------------------------------
# splitter_io — input file + output folder pickers (drag-drop)
# ---------------------------------------------------------------------------


@register_section("splitter_io")
def splitter_io(model, target=None, **options):
    """AutoForm factory for the input-file / output-folder picker rows."""
    return _IoSection(model)


class _IoSection(QtWidgets.QWidget):
    """Input-file and output-folder rows with browse buttons and drag-drop."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(2, 2, 2, 2)
        form.setSpacing(4)

        self._in_edit = QtWidgets.QLineEdit()
        self._in_edit.setPlaceholderText("Drop a TTTR file here or browse…")
        in_browse = QtWidgets.QToolButton()
        in_browse.setText("…")
        in_browse.clicked.connect(self._browse_input)
        form.addRow("Input file", _row(self._in_edit, in_browse))

        self._out_edit = QtWidgets.QLineEdit()
        self._out_edit.setPlaceholderText("Output folder")
        out_browse = QtWidgets.QToolButton()
        out_browse.setText("…")
        out_browse.clicked.connect(self._browse_output)
        form.addRow("Output folder", _row(self._out_edit, out_browse))

        _enable_file_drop(self._in_edit, self._load_input)
        _enable_folder_drop(self._out_edit, self._set_output)
        self._in_edit.editingFinished.connect(
            lambda: self._load_input(self._in_edit.text().strip())
        )
        self._out_edit.editingFinished.connect(
            lambda: self._set_output(self._out_edit.text().strip())
        )

        self._model.add_observer(self._on_model_event)

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        if event in ("loaded", "fields"):
            for edit, value in (
                (self._in_edit, self._model.input_file),
                (self._out_edit, self._model.output_folder),
            ):
                if edit.text() != value:
                    edit.blockSignals(True)
                    edit.setText(value)
                    edit.blockSignals(False)

    # ── actions ─────────────────────────────────────────────────────────
    def _browse_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select TTTR file")
        if path:
            self._load_input(path)

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._set_output(folder)

    def _set_output(self, folder: str) -> None:
        self._model.output_folder = folder
        self._out_edit.setText(folder)

    def _load_input(self, path: str) -> None:
        if not path:
            return
        p = pathlib.Path(path)
        if not p.is_file():
            QtWidgets.QMessageBox.warning(self, "Invalid file", f"'{path}' is not a valid file.")
            return
        import tttrlib

        from chisurf.gui.widgets.staged_loading import load_with_progress

        tttr_type = self._model.tttr_type

        def _load(local_path):
            if tttr_type is None:
                return tttrlib.TTTR(local_path)
            return tttrlib.TTTR(local_path, tttr_type)

        tttr = load_with_progress(self, _load, str(p), title="Loading TTTR file")
        if tttr is None:
            return
        self._model.set_tttr(tttr, str(p))


# ---------------------------------------------------------------------------
# splitter_run — Convert/Split button + progress bar
# ---------------------------------------------------------------------------


@register_section("splitter_run")
def splitter_run(model, target=None, **options):
    """AutoForm factory for the Convert/Split button and progress bar."""
    return _RunSection(model)


class _RunSection(QtWidgets.QWidget):
    """The Convert/Split action button and its progress bar."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._running = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._btn = _tool_button(
            "✂️ Convert / Split",
            self._run,
            "Split / convert the loaded TTTR file into the output folder.",
        )
        layout.addWidget(self._btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)

    def _run(self) -> None:
        if self._running:
            return
        reason = self._model.can_split()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Cannot split", reason)
            return
        self._running = True
        self._btn.setEnabled(False)
        try:
            self._model.do_split(progress_cb=self._on_progress)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Split failed", str(exc))
        finally:
            self._running = False
            self._btn.setEnabled(True)

    def _on_progress(self, percent: int) -> None:
        self._progress.setValue(int(percent))
        QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# splitter_batch — batch file list + controls
# ---------------------------------------------------------------------------


@register_section("splitter_batch_run")
def splitter_batch_run(model, target=None, **options):
    """AutoForm factory for the batch Start button + progress bar.

    The file list is the general ``path_list`` section (target ``batch_files``)
    and the parent-folder toggle is a declarative ``toggle``; this section only
    runs the batch.
    """
    return _BatchRunSection(model)


class _BatchRunSection(QtWidgets.QWidget):
    """Start button + progress for the batch splitter run."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._running = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        row.addWidget(self._progress, 1)
        self._btn_start = _tool_button(
            "▶ Start batch", self._start, "Process every file with the options above."
        )
        row.addWidget(self._btn_start)
        layout.addLayout(row)

    def _start(self) -> None:
        if self._running:
            return
        if not self._model.batch_files:
            QtWidgets.QMessageBox.information(self, "No files", "Add PTU files or folders first.")
            return
        self._running = True
        self._btn_start.setEnabled(False)
        try:
            count = self._model.run_batch(file_progress_cb=self._on_file)
            self._progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Batch complete", f"Processed {count} file(s).")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Batch failed", str(exc))
        finally:
            self._running = False
            self._btn_start.setEnabled(True)

    def _on_file(self, index: int, count: int, path: str) -> None:
        self._progress.setValue(int(index * 100 / max(1, count)))
        QtWidgets.QApplication.processEvents()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Wrap *widgets* in a tight horizontal row container."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    for w in widgets:
        lay.addWidget(w)
    return box


def _tool_button(text: str, slot, tooltip: str = "") -> QtWidgets.QToolButton:
    """Build a ``QToolButton`` with an emoji label (house style) and optional tip."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.clicked.connect(slot)
    return btn


__all__ = ["splitter_io", "splitter_run", "splitter_batch_run"]
