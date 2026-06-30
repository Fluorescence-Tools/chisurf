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


class _PtuFileDropList(QtWidgets.QListWidget):
    """List widget that accepts dropped files/folders and collects ``.ptu`` files."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        paths = [
            pathlib.Path(u.toLocalFile())
            for u in (event.mimeData().urls() or [])
            if u.toLocalFile()
        ]
        if paths:
            self.add_paths([p for p in paths if p.exists()])
        event.acceptProposedAction()

    def add_paths(self, paths: list[pathlib.Path]):
        files: list[pathlib.Path] = []
        for p in paths:
            if p.is_dir():
                files.extend(self._collect_ptu_files(p))
            elif p.is_file() and p.suffix.lower() == ".ptu":
                files.append(p.resolve())
        existing = {self.item(i).text() for i in range(self.count())}
        added = False
        for f in sorted(set(map(str, files))):
            if f not in existing:
                self.addItem(f)
                added = True
        if added:
            self.changed.emit()

    @staticmethod
    def _collect_ptu_files(folder: pathlib.Path) -> list[pathlib.Path]:
        try:
            return [p.resolve() for p in folder.rglob("*.ptu") if p.is_file()]
        except Exception:
            return []


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

        self._btn = QtWidgets.QPushButton("Convert / Split")
        self._btn.setToolTip("Split / convert the loaded TTTR file into the output folder.")
        self._btn.clicked.connect(self._run)
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


@register_section("splitter_batch")
def splitter_batch(model, target=None, **options):
    """AutoForm factory for the batch file list and its controls."""
    return _BatchSection(model)


class _BatchSection(QtWidgets.QWidget):
    """Drag-drop list of PTU files/folders processed with the current options."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._running = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        info = QtWidgets.QLabel(
            "Drop PTU files and/or folders below (folders are scanned recursively)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = _PtuFileDropList()
        self._list.changed.connect(self._sync_files)
        layout.addWidget(self._list, 1)

        self._chk_parent = QtWidgets.QCheckBox("Use file's parent as output folder")
        self._chk_parent.setChecked(bool(self._model.batch_use_parent))
        self._chk_parent.toggled.connect(
            lambda v: setattr(self._model, "batch_use_parent", bool(v))
        )
        layout.addWidget(self._chk_parent)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._btn_add_files = _tool_button("Add Files…", self._add_files)
        self._btn_add_folders = _tool_button("Add Folder…", self._add_folder)
        self._btn_remove = _tool_button("Remove", self._remove_selected)
        self._btn_clear = _tool_button("Clear", self._clear)
        for b in (self._btn_add_files, self._btn_add_folders, self._btn_remove, self._btn_clear):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        self._btn_start = QtWidgets.QPushButton("Start batch")
        self._btn_start.clicked.connect(self._start)
        btn_row.addWidget(self._btn_start)
        layout.addLayout(btn_row)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

    # ── list management ─────────────────────────────────────────────────
    def _sync_files(self) -> None:
        self._model.batch_files = [self._list.item(i).text() for i in range(self._list.count())]

    def _add_files(self) -> None:
        dlg = QtWidgets.QFileDialog(self, "Select PTU files")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.setNameFilter("PTU files (*.ptu)")
        if dlg.exec_():
            self._list.add_paths([pathlib.Path(f) for f in dlg.selectedFiles()])
            self._sync_files()

    def _add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder")
        if folder:
            self._list.add_paths([pathlib.Path(folder)])
            self._sync_files()

    def _remove_selected(self) -> None:
        for it in self._list.selectedItems():
            self._list.takeItem(self._list.row(it))
        self._sync_files()

    def _clear(self) -> None:
        self._list.clear()
        self._sync_files()

    # ── run ─────────────────────────────────────────────────────────────
    def _start(self) -> None:
        if self._running:
            return
        self._sync_files()
        if not self._model.batch_files:
            QtWidgets.QMessageBox.information(self, "No files", "Add PTU files or folders first.")
            return
        self._running = True
        self._set_enabled(False)
        try:
            count = self._model.run_batch(file_progress_cb=self._on_file)
            self._progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Batch complete", f"Processed {count} file(s).")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Batch failed", str(exc))
        finally:
            self._running = False
            self._set_enabled(True)

    def _on_file(self, index: int, count: int, path: str) -> None:
        self._progress.setValue(int(index * 100 / max(1, count)))
        QtWidgets.QApplication.processEvents()

    def _set_enabled(self, enabled: bool) -> None:
        for w in (
            self._list,
            self._chk_parent,
            self._btn_add_files,
            self._btn_add_folders,
            self._btn_remove,
            self._btn_clear,
            self._btn_start,
        ):
            w.setEnabled(enabled)


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


def _tool_button(text: str, slot) -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton(text)
    btn.clicked.connect(slot)
    return btn


__all__ = ["splitter_io", "splitter_run", "splitter_batch"]
