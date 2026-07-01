"""General ``path_list`` AutoForm section — a reusable file/folder drop list.

A drag-drop list of file paths bound to a model attribute, usable from any
``.view.json`` (like ``plot`` / ``image`` / ``waterfall``). Replaces the
hand-rolled drop-list widgets each batch tool used to copy.

Declare it with::

    {"type": "custom", "key": "path_list", "target": "batch_files",
     "options": {"extensions": [".sm", ".ptu"], "add_folders": true}}

``target`` names a model attribute holding a ``list[str]`` of paths; the section
reads it to populate and writes it back (then calls ``model.update()``) on every
change. Dropped folders are expanded recursively to files whose extension is in
``extensions`` (case-insensitive; empty ⇒ accept any file). Options: ``extensions``
(list), ``add_folders`` (bool, default True), ``dialog_filter`` (file-dialog
filter string), ``title`` (header label).
"""

from __future__ import annotations

import logging
import pathlib

from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.tools.chisurf_dock_tool import PathDropListWidget

from .registry import register_section

logger = logging.getLogger(__name__)


def _tool_button(text: str, tooltip: str, slot) -> QtWidgets.QToolButton:
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.clicked.connect(slot)
    return btn


class PathListWidget(QtWidgets.QWidget):
    """Drag-drop file/folder list bound to a model ``list[str]`` attribute."""

    #: marker so a hosting dock panel gives this section the spare vertical space.
    _autoform_expanding = True

    def __init__(self, model, target: str, **options):
        super().__init__()
        self._model = model
        self._target = target
        self._exts = {e.lower() for e in (options.get("extensions") or [])}
        self._add_folders = bool(options.get("add_folders", True))
        self._dialog_filter = options.get("dialog_filter") or self._default_filter()

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if options.get("title"):
            header = QtWidgets.QLabel(str(options["title"]))
            header.setStyleSheet("font-weight: bold; padding: 2px;")
            layout.addWidget(header)

        self._list = PathDropListWidget(path_filter=self._accepts)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._list.pathsDropped.connect(self._on_dropped)
        layout.addWidget(self._list, 1)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(_tool_button("➕ Files", "Add files.", self._add_files))
        if self._add_folders:
            bar.addWidget(
                _tool_button("📁 Folder", "Add a folder (scanned recursively).", self._add_folder)
            )
        bar.addWidget(_tool_button("➖ Remove", "Remove selected entries.", self._remove_selected))
        bar.addWidget(_tool_button("🗑 Clear", "Clear the list.", self._clear))
        bar.addStretch(1)
        layout.addLayout(bar)

        self._refresh_from_model()

    # ── filtering / expansion ───────────────────────────────────────────
    def _accepts(self, local_path: str) -> bool:
        p = pathlib.Path(local_path)
        if p.is_dir():
            return self._add_folders
        return not self._exts or p.suffix.lower() in self._exts

    def _default_filter(self) -> str:
        if not self._exts:
            return "All Files (*)"
        pattern = " ".join(f"*{e}" for e in sorted(self._exts))
        return f"Files ({pattern});;All Files (*)"

    def _expand(self, paths: list[str]) -> list[str]:
        out: list[str] = []
        for item in paths:
            p = pathlib.Path(item)
            if p.is_file():
                out.append(str(p))
            elif p.is_dir() and self._add_folders:
                out.extend(
                    str(f)
                    for f in sorted(p.rglob("*"))
                    if f.is_file() and (not self._exts or f.suffix.lower() in self._exts)
                )
        return out

    # ── model binding ───────────────────────────────────────────────────
    def _current(self) -> list[str]:
        value = getattr(self._model, self._target, None)
        return list(value) if isinstance(value, list) else []

    def _commit(self, paths: list[str]) -> None:
        # de-duplicate while preserving order
        seen: set[str] = set()
        unique = [p for p in paths if not (p in seen or seen.add(p))]
        setattr(self._model, self._target, unique)
        self._refresh_list(unique)
        try:
            self._model.update()
        except Exception:
            pass

    def _add(self, paths: list[str]) -> None:
        self._commit(self._current() + self._expand(paths))

    def _refresh_from_model(self) -> None:
        self._refresh_list(self._current())

    def _refresh_list(self, paths: list[str]) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._list.addItems(paths)
        self._list.blockSignals(False)

    # ── actions ─────────────────────────────────────────────────────────
    def _on_dropped(self, paths: list) -> None:
        self._add([str(p) for p in paths])

    def _add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add files", "", self._dialog_filter
        )
        if files:
            self._add(list(files))

    def _add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add a folder")
        if folder:
            self._add([folder])

    def _remove_selected(self) -> None:
        remove = {it.text() for it in self._list.selectedItems()}
        if remove:
            self._commit([p for p in self._current() if p not in remove])

    def _clear(self) -> None:
        self._commit([])

    # exposed for tests
    def expand(self, paths: list[str]) -> list[str]:
        """Expand *paths* (files + folders) to the accepted file list."""
        return self._expand(paths)


@register_section("path_list")
def _path_list_section_factory(model, target: str, **options):
    """Custom-section factory for the general file/folder drop list."""
    return PathListWidget(model, target, **options)


__all__ = ["PathListWidget"]
