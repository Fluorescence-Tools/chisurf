from __future__ import annotations

import pathlib

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.plugins.core.code_editor.editor import CodeEditor


class CodeEditorWindow(QtWidgets.QMainWindow):
    """Full code editor plugin window built around the shared ``CodeEditor``."""

    def __init__(
        self,
        *args,
        filename: str = None,
        project_root: str | pathlib.Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Code Editor")
        self.resize(1200, 800)

        self.editor = CodeEditor(filename=filename, project_root=project_root, **kwargs)
        self.setCentralWidget(self.editor)
        self.actions = self.editor.create_actions(self)

        self._create_docks()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()

        self.editor.statusChanged.connect(self._update_status)
        self.editor.lspStatusChanged.connect(self._update_lsp_status)

    def _create_docks(self) -> None:
        """Create dock widgets for the full editor window."""
        self.file_dock = QtWidgets.QDockWidget("Project", self)
        self.file_dock.setObjectName("code_editor_project_dock")
        self.file_dock.setWidget(self.editor.project_browser_widget())
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.file_dock)

        self.symbol_dock = QtWidgets.QDockWidget("Symbols", self)
        self.symbol_dock.setObjectName("code_editor_symbols_dock")
        self.symbol_dock.setWidget(self.editor.symbol_outline_widget())
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.symbol_dock)
        self.tabifyDockWidget(self.file_dock, self.symbol_dock)
        self.file_dock.raise_()

        self.diagnostics_dock = QtWidgets.QDockWidget("Diagnostics", self)
        self.diagnostics_dock.setObjectName("code_editor_diagnostics_dock")
        self.diagnostics_dock.setWidget(self.editor.diagnostics_widget())
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.diagnostics_dock)

        self.agent_dock = QtWidgets.QDockWidget("Agent", self)
        self.agent_dock.setObjectName("code_editor_agent_dock")
        self.agent_dock.setWidget(self.editor.agent_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.agent_dock)
        self.agent_dock.hide()
        try:
            self.actions["agent"].triggered.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.actions["agent"].triggered.connect(
            lambda _checked=False: self.agent_dock.setVisible(not self.agent_dock.isVisible())
        )

    def _create_menus(self) -> None:
        """Create the editor menu bar."""
        file_menu = self.menuBar().addMenu("File")
        for name in ["new", "open", "save", "save_as", "reload"]:
            file_menu.addAction(self.actions[name])
        file_menu.addSeparator()
        file_menu.addAction("Close", self.close)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.actions["completion"])

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction(self.actions["settings"])
        settings_menu.addSeparator()
        settings_menu.addAction(self.actions["toggle_line_numbers"])
        settings_menu.addAction(self.actions["toggle_lsp"])

        navigate_menu = self.menuBar().addMenu("Navigate")
        for name in ["back", "forward", "definition"]:
            navigate_menu.addAction(self.actions[name])

        view_menu = self.menuBar().addMenu("View")
        for dock in [self.file_dock, self.symbol_dock, self.diagnostics_dock, self.agent_dock]:
            view_menu.addAction(dock.toggleViewAction())

        run_menu = self.menuBar().addMenu("Run")
        run_menu.addAction(self.actions["run"])

    def _create_toolbar(self) -> None:
        """Create the main editor toolbar."""
        toolbar = QtWidgets.QToolBar("Editor", self)
        toolbar.setObjectName("code_editor_toolbar")
        toolbar.setMovable(True)
        for name in ["new", "open", "save"]:
            toolbar.addAction(self.actions[name])
        toolbar.addSeparator()
        for name in ["back", "forward", "definition", "completion"]:
            toolbar.addAction(self.actions[name])
        toolbar.addSeparator()
        toolbar.addAction(self.actions["run"])
        toolbar.addAction(self.actions["settings"])
        toolbar.addAction(self.actions["agent"])
        self.addToolBar(toolbar)

    def _create_status_bar(self) -> None:
        """Create status labels for editor state."""
        self.file_status = QtWidgets.QLabel("Untitled", self)
        self.position_status = QtWidgets.QLabel("Ln 1, Col 0", self)
        self.dirty_status = QtWidgets.QLabel("", self)
        self.lsp_status = QtWidgets.QLabel("LSP idle", self)
        self.statusBar().addPermanentWidget(self.file_status, 1)
        self.statusBar().addPermanentWidget(self.position_status)
        self.statusBar().addPermanentWidget(self.dirty_status)
        self.statusBar().addPermanentWidget(self.lsp_status)
        self.statusBar().showMessage("Ready")

    def _update_status(self, status: dict) -> None:
        """Update the status bar from shared editor status."""
        if "file" in status:
            self.file_status.setText(str(status.get("file") or "Untitled"))
        if "line" in status:
            self.position_status.setText(
                f"Ln {status.get('line', 1)}, Col {status.get('column', 0)}"
            )
        if "modified" in status:
            self.dirty_status.setText("Modified" if status.get("modified") else "")
        if "lsp" in status:
            self._update_lsp_status(str(status["lsp"]))

    def _update_lsp_status(self, status: str) -> None:
        """Update the status bar LSP state."""
        self.lsp_status.setText(status)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop editor services when the window closes."""
        if self.editor._lsp_client is not None:
            self.editor._lsp_client.stop()
        super().closeEvent(event)


__all__ = ["CodeEditorWindow"]
