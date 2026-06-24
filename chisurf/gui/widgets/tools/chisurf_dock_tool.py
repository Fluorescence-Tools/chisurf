"""Shared dockable-tool base for transformer GUIs (PRD-23 Task 1).

`ChisurfDockTool` factors the boilerplate every transformer tool re-implemented
(drag-drop of file/folder paths, a `DockArea` central widget, window-geometry
persistence, and MFDB-connectivity status) into one base, so fixes propagate and
the per-tool widget stays a thin view. `PathDropListWidget` is the byte-identical
drag-drop list both tools had copied.

The base performs **no** I/O or DB work on construction (PRD-23 Task 4): it only
wires Qt widgets. MFDB access is via the overridable `acquire_mfdb_connection`
hook, called lazily on demand — never in `__init__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets


def local_paths_from_event(
    event: QtGui.QDropEvent, *, require_exists: bool = False
) -> list[Path]:
    """Return existing-or-not local filesystem paths from a drop event's URLs."""
    paths: list[Path] = []
    for url in event.mimeData().urls():
        local_path = url.toLocalFile()
        if not local_path:
            continue
        path = Path(local_path)
        if require_exists and not path.exists():
            continue
        paths.append(path)
    return paths


class PathDropListWidget(QtWidgets.QListWidget):
    """List widget that accepts dropped file and folder paths.

    Emits :attr:`pathsDropped` with the dropped local paths (existing only).
    Previously duplicated verbatim in each transformer tool.
    """

    pathsDropped = QtCore.Signal(list)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept URL drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Accept URL moves."""
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Emit local paths from dropped URLs."""
        paths = local_paths_from_event(event, require_exists=True)
        if paths:
            self.pathsDropped.emit(paths)
        event.acceptProposedAction()

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        """Return supported drop actions."""
        return QtCore.Qt.DropAction.CopyAction


class ChisurfDockTool(QtWidgets.QMainWindow):
    """Base for dockable transformer tools (drag-drop, docks, MFDB status).

    Subclasses build their own widgets/docks/toolbar in ``__init__`` as before;
    this base adds window-level path drag-drop (dispatched to
    :meth:`on_paths_dropped`), window-geometry persistence helpers, and lazy
    MFDB-connectivity accessors. It accepts and forwards ``*args``/``**kwargs`` to
    ``QMainWindow`` so existing ``super().__init__(*args, **kwargs)`` calls keep
    working.
    """

    #: QSettings application key for window-geometry persistence; override per tool.
    tool_settings_name: str = "ChisurfDockTool"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    # -- drag & drop ----------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept file URL drops on the main window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Dispatch dropped local paths to :meth:`on_paths_dropped`."""
        paths = local_paths_from_event(event)
        if paths:
            self.on_paths_dropped(paths)
        event.acceptProposedAction()

    def on_paths_dropped(self, paths: list[Path]) -> None:
        """Handle paths dropped on the window.

        Default: forward to ``self._add_paths`` when the subclass defines it
        (the established convention); otherwise no-op. Override to customise.
        """
        add_paths = getattr(self, "_add_paths", None)
        if callable(add_paths):
            add_paths(paths)

    # -- MFDB connectivity (lazy; never on construction) ----------------------

    def acquire_mfdb_connection(self) -> Any | None:
        """Return an MFDB connection, or ``None``. Override per tool.

        The base returns ``None`` (no connection). Tools override to delegate to
        their api-layer connection helper. Called lazily — never in ``__init__``.
        """
        return None

    def mfdb_connection(self) -> Any | None:
        """Return the active MFDB connection (via :meth:`acquire_mfdb_connection`)."""
        try:
            return self.acquire_mfdb_connection()
        except Exception:
            return None

    def mfdb_connected(self) -> bool:
        """Return whether an MFDB connection is currently available."""
        return self.mfdb_connection() is not None

    # -- window geometry persistence ------------------------------------------

    def save_window_geometry(self) -> None:
        """Persist the main-window geometry to QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", self.tool_settings_name)
            settings.setValue("geometry", self.saveGeometry())
            settings.sync()
        except Exception:
            pass

    def restore_window_geometry(self) -> None:
        """Restore the main-window geometry from QSettings, if present."""
        try:
            settings = QtCore.QSettings("chisurf", self.tool_settings_name)
            geometry = settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        except Exception:
            pass
