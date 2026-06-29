from __future__ import annotations

import pathlib

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.plugins.core.code_editor.editor import (
    CodeEditor,
    get_editor_settings,
    save_editor_settings,
)
from chisurf.plugins.icon_utils import create_emoji_icon


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

        self.output_dock = QtWidgets.QDockWidget("Output", self)
        self.output_dock.setObjectName("code_editor_output_dock")
        self.output_dock.setWidget(self.editor.output_console_widget())
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.output_dock)
        self.tabifyDockWidget(self.diagnostics_dock, self.output_dock)
        self.output_dock.raise_()  # show output by default

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
        for name in ["new", "open", "open_folder", "save", "save_as", "reload"]:
            file_menu.addAction(self.actions[name])
        file_menu.addSeparator()
        file_menu.addAction("Close", self.close)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.actions["find"])
        edit_menu.addSeparator()
        edit_menu.addAction(self.actions["completion"])

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction(self.actions["settings"])
        settings_menu.addSeparator()
        settings_menu.addAction(self.actions["toggle_line_numbers"])
        settings_menu.addAction(self.actions["toggle_whitespace"])
        settings_menu.addAction(self.actions["toggle_lsp"])

        navigate_menu = self.menuBar().addMenu("Navigate")
        for name in ["back", "forward", "definition"]:
            navigate_menu.addAction(self.actions[name])

        view_menu = self.menuBar().addMenu("View")
        for dock in [self.file_dock, self.symbol_dock, self.diagnostics_dock, self.output_dock, self.agent_dock]:
            view_menu.addAction(dock.toggleViewAction())

        run_menu = self.menuBar().addMenu("Run")
        run_menu.addAction(self.actions["run"])
        run_menu.addAction(self.actions["ruff"])

    def _create_toolbar(self) -> None:
        """Create the main editor toolbar."""
        toolbar = QtWidgets.QToolBar("Editor", self)
        toolbar.setObjectName("code_editor_toolbar")
        toolbar.setMovable(True)
        toolbar.setIconSize(QtCore.QSize(18, 18))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        for name in ["new", "open", "open_folder", "save"]:
            toolbar.addAction(self.actions[name])
        toolbar.addSeparator()
        for name in ["back", "forward", "definition", "completion"]:
            toolbar.addAction(self.actions[name])
        toolbar.addSeparator()

        run_btn = QtWidgets.QToolButton(toolbar)
        run_btn.setIcon(create_emoji_icon("▶", size=24))
        run_btn.setText("Run")
        run_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        run_btn.setToolTip("Run the current script in the selected endpoint")
        run_btn.clicked.connect(lambda _checked=False: self.editor.run_macro(None))
        toolbar.addWidget(run_btn)

        stop_btn = QtWidgets.QToolButton(toolbar)
        stop_btn.setIcon(create_emoji_icon("⏹", size=24))
        stop_btn.setText("Stop")
        stop_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        stop_btn.setToolTip("Stop the running script")
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(self.editor.stop_macro)
        toolbar.addWidget(stop_btn)

        self.editor.runStateChanged.connect(
            lambda running: (
                run_btn.setEnabled(not running),
                stop_btn.setEnabled(running),
                self.output_dock.raise_() if running else None,
            )
        )

        endpoint = QtWidgets.QComboBox(toolbar)
        _endpoint_keys = ["console", "process", "ipython"]
        endpoint.addItem(create_emoji_icon("🖥", size=16), "Console")
        endpoint.addItem(create_emoji_icon("⚙", size=16), "Process")
        endpoint.addItem(create_emoji_icon("🐍", size=16), "IPython")
        settings = get_editor_settings()
        current = settings.get("run_endpoint", "process")
        endpoint.setCurrentIndex(_endpoint_keys.index(current) if current in _endpoint_keys else 1)
        endpoint.setToolTip(
            "Console — exec() in-process (cs in scope)\n"
            "Process — separate subprocess\n"
            "IPython — send to the ChiSurf IPython console (%run)"
        )
        endpoint.currentIndexChanged.connect(
            lambda idx: self._set_run_endpoint(_endpoint_keys[idx])
        )
        toolbar.addWidget(endpoint)
        self._endpoint_combo = endpoint
        self._endpoint_keys = _endpoint_keys

        # When the active file has a shebang, pre-select the matching endpoint.
        # The user can still change the combo before pressing Run.
        self.editor.endpointHint.connect(self._apply_endpoint_hint)

        toolbar.addSeparator()
        toolbar.addAction(self.actions["ruff"])
        toolbar.addAction(self.actions["toggle_whitespace"])
        ws_btn = toolbar.widgetForAction(self.actions["toggle_whitespace"])
        if ws_btn is not None:
            ws_btn.setStyleSheet(
                "QToolButton:checked { background-color: palette(highlight);"
                " color: palette(highlighted-text); border-radius: 3px; }"
            )
        toolbar.addAction(self.actions["settings"])

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        toolbar.addWidget(spacer)

        toolbar.addSeparator()
        toolbar.addAction(self.actions["agent"])
        self.addToolBar(toolbar)

    def _set_run_endpoint(self, mode: str) -> None:
        """Persist the selected execution endpoint."""
        settings = get_editor_settings()
        settings["run_endpoint"] = mode
        save_editor_settings(settings)

    def _apply_endpoint_hint(self, endpoint: str) -> None:
        """Pre-select the endpoint combo to match a script's shebang (user can still override)."""
        if endpoint not in self._endpoint_keys:
            return
        idx = self._endpoint_keys.index(endpoint)
        # Block the combo's signal so this doesn't overwrite the persisted preference.
        self._endpoint_combo.blockSignals(True)
        self._endpoint_combo.setCurrentIndex(idx)
        self._endpoint_combo.blockSignals(False)

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
        if self.editor._rpc_server is not None:
            self.editor._rpc_server.stop()
        super().closeEvent(event)


__all__ = ["CodeEditorWindow"]
