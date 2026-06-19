"""New-style GUI entrypoint for the FPS JSON Editor plugin."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

from ..api.client import FpsJsonEditorClient
from .editor import FpsJsonEditor

_manifest = load_manifest(Path(__file__).parents[1] / "manifest.json")


class FpsJsonEditorTool(QtWidgets.QMainWindow):
    """Main window for the FPS JSON Editor plugin."""

    def __init__(self) -> None:
        """Initialize the plugin window and editor widget."""
        super().__init__()
        title = _manifest.display_name if _manifest is not None else "FPS JSON Editor"
        self.setWindowTitle(title)
        self.resize(1200, 760)
        self._client = FpsJsonEditorClient()
        self.editor = FpsJsonEditor(client=self._client)
        self.setCentralWidget(self.editor)
        self._init_actions()

    def _init_actions(self) -> None:
        """Create window actions for file-level editor operations."""
        toolbar = self.addToolBar("FPS JSON")
        toolbar.setObjectName("fps_json_editor_toolbar")

        file_menu = self.menuBar().addMenu("&File")

        load_action = QtWidgets.QAction("📂 Load", self)
        load_action.triggered.connect(self.editor.onLoadJSON)
        file_menu.addAction(load_action)
        toolbar.addAction(load_action)

        save_action = QtWidgets.QAction("💾 Save", self)
        save_action.triggered.connect(self.editor.onSaveJSON)
        file_menu.addAction(save_action)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()

        update_action = QtWidgets.QAction("🔄 Update", self)
        update_action.setToolTip("Update UI models from JSON editor text")
        update_action.triggered.connect(self.editor.onReadTextEdit)
        file_menu.addAction(update_action)
        toolbar.addAction(update_action)

        clear_action = QtWidgets.QAction("🗑️ Clear", self)
        clear_action.triggered.connect(self.editor.onClearAll)
        file_menu.addAction(clear_action)
        toolbar.addAction(clear_action)
        
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        help_action = QtWidgets.QAction("ℹ️ Help", self)
        help_action.setToolTip("Show help and documentation")
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)

        self.actionLoad = load_action
        self.actionSave = save_action
        self.actionClear = clear_action
        self.actionUpdate = update_action

    def _show_help(self) -> None:
        """Display help information for the FPS JSON Editor in a modal window."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("FPS JSON Editor Help")
        dialog.setMinimumSize(600, 400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        help_html = (
            "<h3>FPS JSON Editor</h3>"
            "<p>This tool allows you to visually construct and edit <i>Flexible Point-Source (FPS)</i> models.</p>"
            "<ul>"
            "<li><b>Positions:</b> Define dye attachment points by selecting residues from a PDB structure. Use the table to set parameters such as linker length and width.</li>"
            "<li><b>Distances:</b> Define expected distances between your positions. You can group these into different scoring sets.</li>"
            "<li><b>FlexFit:</b> Setup rigid body or flexible fitting parameters for molecular simulations.</li>"
            "<li><b>JSON:</b> Review and manually edit the underlying raw JSON configuration. Click <b>🔄 Update</b> in the toolbar to apply manual changes to the UI.</li>"
            "</ul>"
            "<p><b>Toolbar Actions:</b></p>"
            "<ul>"
            "<li><b>📂 Load / 💾 Save:</b> Load or save the entire FPS model configuration to disk.</li>"
            "<li><b>🔄 Update:</b> Push any raw text edits from the JSON panel back to the visual editor.</li>"
            "<li><b>🗑️ Clear:</b> Remove all positions and distances to start from scratch.</li>"
            "</ul>"
        )
        text_edit.setHtml(help_html)
        layout.addWidget(text_edit)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def showEvent(self, event: QtCore.QShowEvent) -> None:
        """Apply manifest statefulness once when the window is shown."""
        if _manifest is not None:
            apply_manifest_statefulness(self, _manifest)
        super().showEvent(event)


__all__ = ["FpsJsonEditorTool"]
