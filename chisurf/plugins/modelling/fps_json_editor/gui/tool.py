"""New-style GUI entrypoint for the FPS JSON Editor plugin."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

from ..label_structure import LabelStructure
from .communication import FpsJsonEditorClient

_manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")


class FpsJsonEditorTool(QtWidgets.QMainWindow):
    """Main window for the FPS JSON Editor plugin."""

    def __init__(self) -> None:
        """Initialize the plugin window and editor widget."""
        super().__init__()
        title = _manifest.display_name if _manifest is not None else "FPS JSON Editor"
        self.setWindowTitle(title)
        self._client = FpsJsonEditorClient()
        self.editor = LabelStructure(client=self._client)
        self.setCentralWidget(self.editor)

    def showEvent(self, event: QtCore.QShowEvent) -> None:
        """Apply manifest statefulness once when the window is shown."""
        if _manifest is not None:
            apply_manifest_statefulness(self, _manifest)
        super().showEvent(event)


__all__ = ["FpsJsonEditorTool"]
