from __future__ import annotations

import ctypes
import os
import pathlib
import queue
import sys
import tempfile
import threading

from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.fio as io
import chisurf.gui.widgets
from chisurf import logging
from chisurf.gui.widgets.dock_area import DockArea
from chisurf.plugins.core.code_editor.agent_panel import AgentPanelWidget
from chisurf.plugins.core.code_editor.document_store import DocumentStore
from chisurf.plugins.core.code_editor.lsp_client import PythonLspClient
from chisurf.plugins.core.code_editor.rpc_server import EditorRpcServer
from chisurf.plugins.core.code_editor.ruff_runner import RuffRunner
from chisurf.plugins.core.code_editor.symbols import CodeSymbol, find_project_root
from chisurf.plugins.core.code_editor.text_editor import (
    EditorSettingsDialog,
    TextEditor,
    editor_language_key,
    get_editor_settings,
    make_editor_font,
    save_editor_settings,
)
from chisurf.plugins.icon_utils import create_emoji_icon


def _async_raise(tid: int, exc_type: type) -> None:
    """Raise an exception in a thread by its thread id."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exc_type)
    )
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:

    def persist_plugin_state(_name):
        """Return an identity decorator when plugin-state persistence is unavailable."""
        return lambda cls: cls


@persist_plugin_state("code_editor")
class CodeEditor(QtWidgets.QWidget):
    """Tabbed text editor with DockArea tabs and an AI agent side panel."""

    settings_changed = QtCore.Signal(dict)
    statusChanged = QtCore.Signal(dict)
    currentFileChanged = QtCore.Signal(str)
    symbolsChanged = QtCore.Signal(list)
    lspStatusChanged = QtCore.Signal(str)
    runStateChanged = QtCore.Signal(bool)

    def __init__(
        self,
        *args,
        filename: str = None,
        language: str = "Python",
        can_load: bool = True,
        project_root: str | pathlib.Path | None = None,
        enable_lsp: bool | None = None,
        show_tab_bar: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.filename = filename
        self._can_load = can_load
        self._open_files: dict[str, QtWidgets.QWidget] = {}
        self._document_ids: dict[QtWidgets.QWidget, str] = {}
        self._applying_remote_document = False
        self.document_store = DocumentStore()
        self.ruff_runner = RuffRunner()
        self._rpc_event_queue: queue.Queue[dict] = queue.Queue()
        self._rpc_event_timer = QtCore.QTimer(self)
        self._rpc_event_timer.setSingleShot(False)
        self._rpc_event_timer.setInterval(50)
        self._rpc_event_timer.timeout.connect(self._drain_rpc_events)
        self._rpc_server: EditorRpcServer | None = None
        self._agent_panel_visible = False
        self._run_process: QtCore.QProcess | None = None
        self.project_root = find_project_root(project_root or filename or pathlib.Path.cwd())
        editor_settings = get_editor_settings()
        if enable_lsp is None:
            enable_lsp = bool(editor_settings.get("enable_lsp", True))
        self._enable_lsp = enable_lsp
        self._lsp_client: PythonLspClient | None = None
        self._lsp_sync_timer = QtCore.QTimer(self)
        self._lsp_sync_timer.setSingleShot(True)
        self._lsp_sync_timer.setInterval(350)
        self._lsp_sync_timer.timeout.connect(self._sync_current_editor_to_lsp)
        self.setLayout(main_layout)

        self.agent_panel = AgentPanelWidget(
            parent=None, get_context_callback=self._get_editor_context
        )
        self.agent_panel.editor = self

        self.tab_widget = DockArea()
        self.tab_widget.tabActionRequested.connect(self._on_tab_action)
        self.tab_widget.newTabRequested.connect(self._add_new_editor_tab)
        self.tab_widget.currentChanged.connect(self._on_current_tab_changed)
        self.tab_widget.setCloseTabCallback(self._close_tab)
        main_layout.addWidget(self.tab_widget)
        self._create_navigation_widgets()
        self._actions: dict[str, QtWidgets.QAction] = {}

        if can_load or filename:
            self.agent_panel.clear_history()
            self._create_editor_tab(filename=filename, language=language)
        self.tab_widget.setTabsClosable(True)
        # The '+' new-tab button creates blank "Untitled" tabs, which only makes
        # sense for the standalone editor. In embedded read/fixed-file hosts
        # (can_load=False) it has no purpose and rendered as a stray corner box.
        self.tab_widget.setNewTabButtonVisible(can_load)
        self.tab_widget.setTabBarVisible(show_tab_bar)
        self.tab_widget.setContextMenuEnabled(True)

        self._sync_agent_font()
        self._sync_rpc_server()

    def _create_navigation_widgets(self) -> None:
        """Create reusable project, symbol, and diagnostics widgets."""
        self.file_model = QtWidgets.QFileSystemModel(self)
        self.file_model.setRootPath(str(self.project_root))
        self.file_model.setNameFilters(
            [
                "*.py",
                "*.pyw",
                "*.json",
                "*.yaml",
                "*.yml",
                "*.txt",
                "*.md",
            ]
        )
        self.file_model.setNameFilterDisables(False)

        # No parent: these are "shared" widgets a host reparents into its own
        # docks/panels. Parenting them to ``self`` here makes them render as a
        # stray box at (0, 0) over the tab bar in hosts that never use them
        # (e.g. the embedded fit-model editor). With no parent they stay hidden
        # until a host adds them to a layout, which reparents and shows them.
        self.file_tree = QtWidgets.QTreeView()
        self.file_tree.setObjectName("code_editor_file_tree")
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(str(self.project_root)))
        self.file_tree.setHeaderHidden(False)
        for column in range(1, self.file_model.columnCount()):
            self.file_tree.hideColumn(column)
        self.file_tree.doubleClicked.connect(self._on_file_tree_activated)
        self.file_tree.clicked.connect(self._on_file_tree_activated)

        self.symbol_tree = QtWidgets.QTreeWidget()
        self.symbol_tree.setObjectName("code_editor_symbol_tree")
        self.symbol_tree.setHeaderLabels(["Symbol", "Line"])
        self.symbol_tree.itemActivated.connect(self._on_symbol_item_activated)
        self.symbol_tree.itemClicked.connect(self._on_symbol_item_activated)

        self.diagnostics_list = QtWidgets.QListWidget()
        self.diagnostics_list.setObjectName("code_editor_diagnostics_list")
        self.diagnostics_list.setMaximumHeight(100)

    def project_browser_widget(self) -> QtWidgets.QWidget:
        """Return the shared project file browser widget."""
        return self.file_tree

    def symbol_outline_widget(self) -> QtWidgets.QWidget:
        """Return the shared symbol outline widget."""
        return self.symbol_tree

    def diagnostics_widget(self) -> QtWidgets.QWidget:
        """Return the shared diagnostics widget."""
        return self.diagnostics_list

    def set_project_root(self, root: str | pathlib.Path) -> None:
        """Set the project root used by navigation and LSP."""
        self.project_root = find_project_root(root)
        self.file_model.setRootPath(str(self.project_root))
        self.file_tree.setRootIndex(self.file_model.index(str(self.project_root)))
        if self._lsp_client is not None:
            self._lsp_client.stop()
            self._lsp_client = None

    def create_actions(self, parent=None) -> dict[str, QtWidgets.QAction]:
        """Create shared editor actions for menus and toolbars."""
        parent = parent or self
        actions = {
            "new": QtWidgets.QAction(create_emoji_icon("📄", size=20), "New", parent),
            "open": QtWidgets.QAction(create_emoji_icon("📂", size=20), "Open...", parent),
            "save": QtWidgets.QAction(create_emoji_icon("💾", size=20), "Save", parent),
            "save_as": QtWidgets.QAction(create_emoji_icon("💾", size=20), "Save As...", parent),
            "reload": QtWidgets.QAction(create_emoji_icon("🔄", size=20), "Reload", parent),
            "run": QtWidgets.QAction(create_emoji_icon("▶", size=20), "Run Macro", parent),
            "ruff": QtWidgets.QAction(create_emoji_icon("🧹", size=20), "Run Ruff", parent),
            "back": QtWidgets.QAction(create_emoji_icon("◀", size=20), "Back", parent),
            "forward": QtWidgets.QAction(create_emoji_icon("▶", size=20), "Forward", parent),
            "definition": QtWidgets.QAction(create_emoji_icon("🔍", size=20), "Go to Definition", parent),
            "completion": QtWidgets.QAction(create_emoji_icon("✨", size=20), "Complete", parent),
            "settings": QtWidgets.QAction(create_emoji_icon("⚙", size=20), "Editor Settings...", parent),
            "agent": QtWidgets.QAction(create_emoji_icon("🤖", size=20), "Agent", parent),
            "toggle_line_numbers": QtWidgets.QAction("Show Line Numbers", parent),
            "toggle_lsp": QtWidgets.QAction("Enable Python LSP", parent),
        }
        actions["toggle_line_numbers"].setCheckable(True)
        actions["toggle_line_numbers"].setObjectName("toggle_line_numbers")
        actions["toggle_lsp"].setCheckable(True)
        actions["toggle_lsp"].setObjectName("toggle_lsp")
        actions["new"].setShortcut(QtGui.QKeySequence.New)
        actions["open"].setShortcut(QtGui.QKeySequence.Open)
        actions["save"].setShortcut(QtGui.QKeySequence.Save)
        actions["save_as"].setShortcut(QtGui.QKeySequence.SaveAs)
        actions["back"].setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Back))
        actions["forward"].setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Forward))
        actions["definition"].setShortcut(QtGui.QKeySequence("F12"))
        actions["completion"].setShortcut(QtGui.QKeySequence("Ctrl+Space"))
        actions["settings"].setShortcut(QtGui.QKeySequence.Preferences)

        actions["new"].triggered.connect(self._add_new_editor_tab)
        actions["open"].triggered.connect(lambda _checked=False: self.load_file())
        actions["save"].triggered.connect(self.save_text)
        actions["save_as"].triggered.connect(self.save_current_as)
        actions["reload"].triggered.connect(self.reload_current)
        actions["run"].triggered.connect(lambda _checked=False: self.run_macro(None))
        actions["ruff"].triggered.connect(lambda _checked=False: self.run_ruff_current())
        actions["back"].triggered.connect(self.navigate_back)
        actions["forward"].triggered.connect(self.navigate_forward)
        actions["definition"].triggered.connect(self.go_to_definition_current)
        actions["completion"].triggered.connect(self.complete_current)
        actions["settings"].triggered.connect(self.show_editor_settings)
        actions["agent"].triggered.connect(self._toggle_agent_panel)
        actions["toggle_line_numbers"].triggered.connect(self._toggle_line_numbers)
        actions["toggle_lsp"].triggered.connect(self._toggle_lsp)
        self._actions = actions
        self._sync_editor_action_states()
        return actions

    def action(self, name: str) -> QtWidgets.QAction | None:
        """Return a previously created shared action by name."""
        return self._actions.get(name)

    def create_settings_button(self, parent=None):
        """Create a gear button that opens the editor settings dialog."""
        button = QtWidgets.QToolButton(parent)
        button.setText("⚙")
        button.setToolTip("Settings")
        button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
        button.clicked.connect(self.show_editor_settings)
        menu = self._create_settings_menu(button)
        menu.aboutToShow.connect(lambda: self._position_settings_menu(menu, button))
        button.setMenu(menu)
        button.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(
            lambda _pos: self._show_settings_menu(menu, button)
        )
        return button

    def _show_settings_menu(self, menu: QtWidgets.QMenu, button: QtWidgets.QToolButton) -> None:
        """Show the gear-button menu above the button without the built-in arrow."""
        menu.adjustSize()
        offset = QtCore.QPoint(0, -menu.sizeHint().height())
        menu.exec_(button.mapToGlobal(offset))

    def _position_settings_menu(self, menu: QtWidgets.QMenu, button: QtWidgets.QToolButton) -> None:
        """Position the gear-button menu above the button to avoid toolbar overlap."""
        menu.adjustSize()
        offset = QtCore.QPoint(0, -menu.sizeHint().height())
        menu.move(button.mapToGlobal(offset))

    def _create_settings_menu(self, parent=None):
        """Create a compact settings menu for the gear button."""
        settings = get_editor_settings()
        menu = QtWidgets.QMenu(parent)
        menu.addAction("Editor Settings...", self.show_editor_settings)

        line_action = QtWidgets.QAction("Show Line Numbers", parent)
        line_action.setCheckable(True)
        line_action.setChecked(bool(settings.get("line_numbers_visible", True)))
        line_action.toggled.connect(
            lambda checked: self._set_editor_setting_from_action("line_numbers_visible", checked)
        )
        menu.addAction(line_action)

        lsp_action = QtWidgets.QAction("Enable Python LSP", parent)
        lsp_action.setCheckable(True)
        lsp_action.setChecked(bool(settings.get("enable_lsp", True)))
        lsp_action.toggled.connect(
            lambda checked: self._set_editor_setting_from_action("enable_lsp", checked)
        )
        menu.addAction(lsp_action)
        return menu

    def show_editor_settings(self) -> None:
        """Open the persistent editor settings dialog."""
        dialog = EditorSettingsDialog(self)
        dialog.settings_applied.connect(self._save_and_apply_editor_settings)
        dialog.exec_()

    def _save_and_apply_editor_settings(self, settings: dict) -> None:
        """Persist editor settings and apply them to editor widgets."""
        if save_editor_settings(settings):
            self._apply_editor_settings(settings, apply_to_all=True)
            self._sync_editor_action_states()
            self.settings_changed.emit(settings)

    def _apply_editor_settings(self, settings: dict, apply_to_all: bool = True) -> None:
        """Apply editor settings to existing editor tabs and the agent panel."""
        editors = self._iter_editor_tabs() if apply_to_all else [self._get_current_editor()]
        for editor in editors:
            if editor is not None:
                editor.set_editor_settings(settings)
        if hasattr(self, "agent_panel"):
            self._sync_agent_font()
        if "enable_lsp" in settings:
            self._enable_lsp = bool(settings["enable_lsp"])
            if not self._enable_lsp and self._lsp_client is not None:
                self._lsp_client.stop()
                self._lsp_client = None
            self.lspStatusChanged.emit("LSP disabled" if not self._enable_lsp else "LSP idle")
        if any(
            key in settings for key in ["enable_rpc", "rpc_host", "rpc_cmd_port", "rpc_pub_port"]
        ):
            self._sync_rpc_server()

    def _iter_editor_tabs(self):
        """Yield open text editor tabs."""
        for index in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(index)
            if widget is not None and widget is not getattr(self, "agent_panel", None):
                yield widget

    def _sync_editor_action_states(self) -> None:
        """Update checkable editor action states from global settings."""
        settings = get_editor_settings()
        line_action = self._actions.get("toggle_line_numbers")
        if line_action is not None:
            line_action.setChecked(bool(settings.get("line_numbers_visible", True)))
        lsp_action = self._actions.get("toggle_lsp")
        if lsp_action is not None:
            lsp_action.setChecked(self._enable_lsp)

    def _toggle_line_numbers(self, checked=None) -> None:
        """Toggle the global line-number visibility setting."""
        self._set_editor_setting_from_action("line_numbers_visible", checked)

    def _toggle_lsp(self, checked=None) -> None:
        """Toggle the global Python LSP setting."""
        self._set_editor_setting_from_action("enable_lsp", checked)

    def _set_editor_setting_from_action(self, key: str, checked=None) -> None:
        """Persist and apply a checkable editor setting."""
        settings = get_editor_settings()
        settings[key] = bool(checked)
        self._save_and_apply_editor_settings(settings)

    def _sync_rpc_server(self) -> None:
        """Start or stop the GUI-owned editor RPC server from settings."""
        settings = get_editor_settings()
        enabled = bool(settings.get("enable_rpc", False))
        if enabled and self._rpc_server is None:
            self._rpc_server = EditorRpcServer(
                host=str(settings.get("rpc_host", "127.0.0.1")),
                cmd_port=int(settings.get("rpc_cmd_port", 8775)),
                pub_port=int(settings.get("rpc_pub_port", 8776)),
                store=self.document_store,
                runner=self.ruff_runner,
            )
            try:
                self._rpc_server.start()
                self._rpc_event_timer.start()
                self.lspStatusChanged.emit("Editor RPC ready")
            except OSError as exc:
                self._rpc_server = None
                self.lspStatusChanged.emit(f"Editor RPC unavailable: {exc}")
        elif not enabled and self._rpc_server is not None:
            self._rpc_server.stop()
            self._rpc_server = None
            self._rpc_event_timer.stop()
            self.lspStatusChanged.emit("Editor RPC stopped")

    def _document_id_for_editor(self, editor: QtWidgets.QWidget) -> str:
        """Return or create the document id for an editor widget."""
        existing = self._document_ids.get(editor)
        if isinstance(editor, TextEditor) and editor.current_file:
            document_id = DocumentStore.document_id_for_path(editor.current_file)
            if existing and existing != document_id:
                self.document_store.remove(existing)
            self._document_ids[editor] = document_id
            return document_id
        if existing:
            return existing
        document_id = f"untitled:{id(editor)}"
        self._document_ids[editor] = document_id
        return document_id

    def _sync_editor_document(self, editor: QtWidgets.QWidget) -> None:
        """Sync an editor widget into the JSON document store."""
        if not isinstance(editor, TextEditor):
            return
        document_id = self._document_id_for_editor(editor)
        path = editor.current_file or self._get_current_filename() or document_id
        name = pathlib.Path(path).name if path else document_id
        self.document_store.upsert(
            document_id=document_id,
            path=path,
            name=name,
            language=editor.language,
            content=editor.toPlainText(),
            modified=editor.document().isModified(),
        )

    def _remove_editor_document(self, editor: QtWidgets.QWidget) -> None:
        """Remove an editor widget from the JSON document store."""
        document_id = self._document_ids.pop(editor, None)
        if document_id is not None:
            self.document_store.remove(document_id)

    def _add_new_editor_tab(self):
        """Create a new blank editor tab with a unique name."""
        self.agent_panel.clear_history()
        base = "Untitled"
        used = set()
        for i in range(self.tab_widget.count()):
            if self.tab_widget.widget(i) is self.agent_panel:
                continue
            text = self.tab_widget.tabText(i)
            used.add(text[:-2] if text.endswith(" *") else text)
        if base not in used:
            name = base
        else:
            n = 1
            while f"{base}-{n}" in used:
                n += 1
            name = f"{base}-{n}"
        self._create_editor_tab(filename=name)

    def _create_editor_tab(self, filename: str = None, language: str = "Python"):
        """Create a new editor tab."""
        editor = TextEditor(parent=self, language=language or get_editor_settings()["language"])
        editor._definition_uses_host = True
        self.tab_widget.addTab(editor, filename or "Untitled")
        editor.document().modificationChanged.connect(
            lambda modified, e=editor: self._on_modification_changed(e, modified)
        )
        editor.statusChanged.connect(self._on_editor_status_changed)
        editor.symbolsChanged.connect(self._on_editor_symbols_changed)
        editor.filePathChanged.connect(self.currentFileChanged.emit)
        editor.definitionRequested.connect(self.go_to_definition)
        editor.textChanged.connect(self._on_editor_text_changed)
        if self._is_real_file(filename):
            editor.set_current_file(str(pathlib.Path(filename).resolve()))
        tab_index = self.tab_widget.indexOf(editor)
        if hasattr(self, "_on_editor_created"):
            self._on_editor_created(editor)
        self._sync_editor_document(editor)
        return editor, tab_index

    def _get_current_editor(self):
        """Get the current editor widget."""
        w = self.tab_widget.currentWidget()
        if w is getattr(self, "agent_panel", None):
            return None
        return w

    def _get_current_filename(self):
        """Get the filename of the current tab (without dirty marker)."""
        idx = self.tab_widget.currentIndex()
        if idx >= 0:
            text = self.tab_widget.tabText(idx)
            return text[:-2] if text.endswith(" *") else text
        return None

    @staticmethod
    def _is_real_file(filename: str | pathlib.Path | None) -> bool:
        """Return whether *filename* looks like a filesystem path."""
        if not filename:
            return False
        text = str(filename)
        if text.startswith("Untitled"):
            return False
        return os.path.isabs(text) or pathlib.Path(text).exists()

    def _current_path(self) -> str:
        """Return the current editor path if available."""
        editor = self._get_current_editor()
        if editor is not None and getattr(editor, "current_file", None):
            return str(editor.current_file)
        filename = self._get_current_filename()
        return str(filename or "")

    def _on_current_tab_changed(self, _index: int) -> None:
        """Refresh navigation widgets when the active editor changes."""
        editor = self._get_current_editor()
        if editor is None:
            return
        doc_id = self._document_id_for_editor(editor)
        self.document_store.active_document_id = doc_id
        self._on_editor_status_changed(
            {
                "file": editor.current_file or self._get_current_filename() or "",
                "line": editor.line_column()[0],
                "column": editor.line_column()[1],
                "modified": editor.document().isModified(),
                "language": editor.language,
            }
        )
        self._on_editor_symbols_changed(editor.refresh_symbols())

    def _on_editor_status_changed(self, status: dict) -> None:
        """Forward current editor status to hosts."""
        self.statusChanged.emit(status)

    def _on_editor_symbols_changed(self, symbols: list) -> None:
        """Refresh the shared symbol outline."""
        if self.sender() is not self._get_current_editor() and self.sender() is not None:
            return
        self._populate_symbol_tree(symbols)
        self.symbolsChanged.emit(symbols)

    def _on_editor_text_changed(self) -> None:
        """Schedule LSP synchronization after edits."""
        editor = self.sender()
        if editor is self._get_current_editor() and not self._applying_remote_document:
            self._sync_editor_document(editor)
            self._lsp_sync_timer.start()

    def _populate_symbol_tree(self, symbols: list[CodeSymbol]) -> None:
        """Populate the symbol outline tree from *symbols*."""
        self.symbol_tree.blockSignals(True)
        self.symbol_tree.clear()
        parents: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for symbol in symbols:
            item = QtWidgets.QTreeWidgetItem([symbol.display_name, str(symbol.line)])
            item.setData(0, QtCore.Qt.UserRole, symbol)
            if symbol.kind == "class":
                item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
            elif symbol.kind in {"function", "method"}:
                item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            if symbol.parent and symbol.parent in parents:
                parents[symbol.parent].addChild(item)
            else:
                self.symbol_tree.addTopLevelItem(item)
            if symbol.kind == "class":
                parents[symbol.display_name] = item
                parents[symbol.name] = item
        self.symbol_tree.expandAll()
        self.symbol_tree.resizeColumnToContents(0)
        self.symbol_tree.blockSignals(False)

    def _on_file_tree_activated(self, index: QtCore.QModelIndex) -> None:
        """Open a file when the shared project browser is activated."""
        path = pathlib.Path(self.file_model.filePath(index))
        if path.is_file():
            self.open_file(str(path))

    def _on_symbol_item_activated(self, item: QtWidgets.QTreeWidgetItem, _column: int = 0) -> None:
        """Jump to a symbol selected in the outline."""
        symbol = item.data(0, QtCore.Qt.UserRole)
        if isinstance(symbol, CodeSymbol):
            self.goto_symbol(symbol)

    def goto_symbol(self, symbol: CodeSymbol | dict) -> None:
        """Open and jump to *symbol*."""
        path = symbol.get("path", "") if isinstance(symbol, dict) else symbol.path
        line = int(symbol.get("line", 1)) if isinstance(symbol, dict) else symbol.line
        column = int(symbol.get("column", 0)) if isinstance(symbol, dict) else symbol.column
        if path:
            self.open_file(path, line=line, col=column)
            return
        editor = self._get_current_editor()
        if editor is not None:
            editor.goto_line_column(line, column)

    def navigate_back(self) -> None:
        """Navigate the active editor backward."""
        editor = self._get_current_editor()
        if editor is not None:
            editor.navigate_back()

    def navigate_forward(self) -> None:
        """Navigate the active editor forward."""
        editor = self._get_current_editor()
        if editor is not None:
            editor.navigate_forward()

    def go_to_definition_current(self) -> None:
        """Go to the definition for the active cursor word."""
        editor = self._get_current_editor()
        if editor is None:
            return
        word = editor.current_word()
        line, column = editor.line_column()
        self.go_to_definition(word, line, column)

    def go_to_definition(self, word: str, line: int, column: int) -> None:
        """Use LSP definition when available, otherwise use local fallback."""
        editor = self._get_current_editor()
        if editor is None:
            return
        path = editor.current_file or self._current_path()
        if self._enable_lsp and path and self._ensure_lsp():
            self._notify_lsp_open(editor)
            self._lsp_client.definition(
                path,
                line,
                column,
                lambda result, w=word, e=editor: self._handle_definition_result(result, w, e),
            )
            return
        editor.jump_to_definition(word)

    def complete_current(self) -> None:
        """Request and show LSP completions for the active cursor."""
        editor = self._get_current_editor()
        path = self._current_path()
        if editor is None or not path or not self._enable_lsp or not self._ensure_lsp():
            return
        line, column = editor.line_column()
        self._notify_lsp_open(editor)
        self._lsp_client.completion(
            path,
            line,
            column,
            lambda result, e=editor: self._show_completion_result(result, e),
        )

    def _show_completion_result(self, result: object, editor: TextEditor) -> None:
        """Display completion labels from an LSP result."""
        items = result.get("items", result) if isinstance(result, dict) else result
        if not isinstance(items, list):
            return
        labels = []
        for item in items:
            if isinstance(item, dict) and item.get("label"):
                labels.append(str(item["label"]))
        if not labels:
            return
        completer = QtWidgets.QCompleter(sorted(set(labels)), editor)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        completer.activated.connect(lambda text, e=editor: e.insertPlainText(text))
        editor._lsp_completer = completer
        rect = editor.cursorRect()
        rect.setWidth(300)
        completer.complete(rect)

    def _handle_definition_result(self, result: object, word: str, editor: TextEditor) -> None:
        """Open an LSP definition result or fall back locally."""
        location = result[0] if isinstance(result, list) and result else result
        if not isinstance(location, dict):
            editor.jump_to_definition(word)
            return
        uri = location.get("uri") or location.get("targetUri")
        selection = location.get("range") or location.get("targetSelectionRange") or {}
        start = selection.get("start", {}) if isinstance(selection, dict) else {}
        if not uri:
            editor.jump_to_definition(word)
            return
        path = self._path_from_uri(uri)
        self.open_file(path, line=int(start.get("line", 0)) + 1, col=int(start.get("character", 0)))

    @staticmethod
    def _path_from_uri(uri: str) -> str:
        """Convert a file URI returned by LSP into a local path."""
        import urllib.parse
        import urllib.request

        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "file":
            return uri
        return urllib.request.url2pathname(parsed.path)

    def _ensure_lsp(self) -> bool:
        """Start the Python LSP client lazily."""
        if not self._enable_lsp:
            self.lspStatusChanged.emit("LSP disabled")
            return False
        if self._lsp_client is not None:
            return True
        self._lsp_client = PythonLspClient(self.project_root, parent=self)
        self._lsp_client.status_changed.connect(self._on_lsp_status_changed)
        self._lsp_client.diagnostics_received.connect(self._on_lsp_diagnostics)
        if not self._lsp_client.start():
            self._lsp_client = None
            return False
        return True

    def _notify_lsp_open(self, editor: TextEditor) -> None:
        """Open or update the current editor document in LSP."""
        if self._lsp_client is None or not editor.current_file:
            return
        language = "python" if editor_language_key(editor.language) == "python" else "plaintext"
        self._lsp_client.open_document(editor.current_file, editor.toPlainText(), language)

    def _sync_current_editor_to_lsp(self) -> None:
        """Synchronize the current editor contents to LSP."""
        editor = self._get_current_editor()
        if self._lsp_client is None or editor is None or not editor.current_file:
            return
        self._lsp_client.change_document(editor.current_file, editor.toPlainText())

    def _on_lsp_status_changed(self, status: str) -> None:
        """Forward LSP status to hosts."""
        self.lspStatusChanged.emit(status)
        status_payload = {"lsp": status, "file": self._current_path()}
        self.statusChanged.emit(status_payload)

    def _on_lsp_diagnostics(self, uri: str, diagnostics: list) -> None:
        """Show LSP diagnostics in the shared diagnostics widget."""
        self.diagnostics_list.clear()
        path = self._path_from_uri(uri) if uri else ""
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            message = str(diagnostic.get("message", ""))
            range_info = diagnostic.get("range", {})
            start = range_info.get("start", {}) if isinstance(range_info, dict) else {}
            line = int(start.get("line", 0)) + 1
            item = QtWidgets.QListWidgetItem(f"{path}:{line}: {message}")
            item.setData(QtCore.Qt.UserRole, {"path": path, "line": line})
            self.diagnostics_list.addItem(item)

    def _on_modification_changed(self, editor: QtWidgets.QWidget, modified: bool) -> None:
        """Update the `` *`` marker on the tab when the document's modified state changes."""
        idx = self.tab_widget.indexOf(editor)
        if idx < 0:
            return
        text = self.tab_widget.tabText(idx)
        base = text[:-2] if text.endswith(" *") else text
        self.tab_widget.setTabText(idx, base + " *" if modified else base)
        if editor is self._get_current_editor() and hasattr(editor, "_emit_status_changed"):
            editor._emit_status_changed()
        if not self._applying_remote_document:
            self._sync_editor_document(editor)

    def _toggle_agent_panel(self):
        """Toggle the AI agent panel visibility."""
        if self._agent_panel_visible:
            idx = self.tab_widget.indexOf(self.agent_panel)
            if idx >= 0:
                self.tab_widget.removeTab(idx)
            self._agent_panel_visible = False
        else:
            idx = self.tab_widget.indexOf(self.agent_panel)
            if idx < 0:
                self.tab_widget.addTab(self.agent_panel, "Agent")
            self.tab_widget.setCurrentWidget(self.agent_panel)
            self.agent_panel.show()
            self._agent_panel_visible = True

    def _sync_agent_font(self):
        """Apply the editor font to the agent panel."""
        self.agent_panel.set_editor_font(make_editor_font(get_editor_settings()))

    def _get_editor_context(self) -> str:
        """Get the context of all open documents for the agent."""
        context_parts = []
        current_editor = self._get_current_editor()

        for index in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(index)
            if widget is not None and widget is not getattr(self, "agent_panel", None):
                filename = getattr(widget, "current_file", None) or self.tab_widget.tabText(index)
                if filename.endswith(" *"):
                    filename = filename[:-2]
                content = widget.toPlainText()
                is_active = " (active)" if widget is current_editor else ""
                context_parts.append(f"File: {filename}{is_active}\n\n```{content}\n```")

        return "\n\n---\n\n".join(context_parts)

    def _close_tab(self, index: int):
        """Close a tab at the given absolute index."""
        widget = self.tab_widget.widget(index)

        # Agent panel close = toggle off
        if widget is self.agent_panel:
            self._agent_panel_visible = False
            self.tab_widget.removeTab(index)
            return

        # Confirm close if the editor document has unsaved changes.
        if widget is not self.agent_panel:
            if not self._confirm_close_editor(widget, index):
                return

        # Remove from _open_files by matching the widget
        self._remove_editor_document(widget)
        to_remove = [k for k, v in self._open_files.items() if v is widget]
        for k in to_remove:
            if self._lsp_client is not None:
                self._lsp_client.close_document(k)
            del self._open_files[k]

        # Compute editor count BEFORE removing
        editor_count = sum(
            1
            for i in range(self.tab_widget.count())
            if self.tab_widget.widget(i) not in (self.agent_panel, None)
        )

        self.tab_widget.removeTab(index)
        if widget:
            widget.deleteLater()

        # Open a blank Untitled tab if the last editor was just closed
        if editor_count <= 1 and self._can_load:
            self._add_new_editor_tab()

    def _confirm_close_editor(self, editor: QtWidgets.QWidget, index: int) -> bool:
        """Return whether an editor tab with unsaved changes may close."""
        modified = bool(
            hasattr(editor, "document")
            and editor.document() is not None
            and editor.document().isModified()
        )
        tab_text = self.tab_widget.tabText(index)
        if not modified and not tab_text.endswith(" *"):
            return True

        name = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Unsaved Changes")
        msg.setText(f"Save changes to {name} before closing?")
        msg.setInformativeText("Choose Discard to close without saving.")
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setStandardButtons(
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel
        )
        msg.setDefaultButton(QtWidgets.QMessageBox.Save)
        reply = msg.exec_()
        if reply == QtWidgets.QMessageBox.Save:
            return self._save_tab(editor, name, index)
        return reply == QtWidgets.QMessageBox.Discard

    def _on_tab_action(self, action: str, index: int):
        """Handle context-menu actions on tabs.

        Parameters
        ----------
        action : str
            One of ``"save"``, ``"save_as"``, ``"rename"``, ``"reload"``,
            ``"copy_path"``, ``"copy_name"``, ``"copy_dir"``.
        index : int
            The absolute tab index.
        """
        editor = self.tab_widget.widget(index)
        if editor is None or editor is self.agent_panel:
            return
        tab_text = self.tab_widget.tabText(index)
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text

        if action == "save":
            self._save_tab(editor, clean, index)
        elif action == "save_as":
            self._save_tab_as(editor, clean, index)
        elif action == "rename":
            self._rename_tab(editor, clean, index)
        elif action == "reload":
            self._reload_tab(editor, clean, index)
        elif action == "copy_path":
            self._copy_to_clipboard(clean)
        elif action == "copy_name":
            self._copy_to_clipboard(pathlib.Path(clean).name)
        elif action == "copy_dir":
            self._copy_to_clipboard(str(pathlib.Path(clean).parent))

    def _save_tab(self, editor, tab_text: str, index: int) -> bool:
        """Save the tab content to its file."""
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        if clean and clean != "Untitled":
            clean = str(pathlib.Path(clean).resolve()) if self._is_real_file(clean) else clean
            try:
                with io.zipped.open_maybe_zipped(clean, "w") as f:
                    f.write(editor.text())
            except OSError as e:
                logging.log(1, f"Error saving {clean}: {e}")
                return False
        else:
            return self._save_tab_as(editor, clean, index)
        if hasattr(editor, "set_current_file"):
            editor.set_current_file(clean)
        editor.document().setModified(False)
        self._sync_editor_document(editor)
        if get_editor_settings().get("run_ruff_on_save", False):
            self.run_ruff_current()
        return True

    def _save_tab_as(self, editor, tab_text: str, index: int) -> bool:
        """Open a save-as dialog and save the tab content."""
        new_filename = cs.gui.widgets.save_file(file_type="Python script (*.py)")
        if not new_filename:
            return False
        new_path = str(new_filename)
        try:
            with io.zipped.open_maybe_zipped(new_path, "w") as f:
                f.write(editor.text())
        except OSError as e:
            logging.log(1, f"Error saving {new_path}: {e}")
            return False
        self.tab_widget.setTabText(index, new_path)
        editor.set_current_file(new_path)
        old_key = next((k for k, v in self._open_files.items() if v is editor), None)
        if old_key:
            del self._open_files[old_key]
        self._open_files[new_path] = editor
        editor.document().setModified(False)
        self._sync_editor_document(editor)
        if get_editor_settings().get("run_ruff_on_save", False):
            self.run_ruff_current()
        return True

    def _rename_tab(self, editor, tab_text: str, index: int):
        """Prompt for a new tab name and update accordingly."""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Tab", "New name:", text=tab_text
        )
        if not ok or not new_name or new_name == tab_text:
            return
        self.tab_widget.setTabText(index, new_name)
        if editor.document().isModified():
            self.tab_widget.setTabText(index, new_name + " *")
        old_key = next((k for k, v in self._open_files.items() if v is editor), None)
        if old_key:
            del self._open_files[old_key]
        self._open_files[new_name] = editor
        self._sync_editor_document(editor)

    def _reload_tab(self, editor, tab_text: str, index: int):
        """Re-read the file from disk and replace editor content."""
        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
        if clean == "Untitled":
            return
        try:
            with open(clean, encoding="utf-8") as f:
                editor.blockSignals(True)
                editor.setText(f.read())
                editor.blockSignals(False)
        except OSError as e:
            logging.log(1, f"Error reloading {clean}: {e}")
            return
        editor.document().setModified(False)
        self._sync_editor_document(editor)
        if hasattr(editor, "refresh_symbols"):
            editor.refresh_symbols()
        self._notify_lsp_open(editor)

    def save_current_as(self) -> None:
        """Save the current tab under a new path."""
        editor = self._get_current_editor()
        if editor is None:
            return
        idx = self.tab_widget.currentIndex()
        self._save_tab_as(editor, self._get_current_filename() or "Untitled", idx)

    def reload_current(self) -> None:
        """Reload the active editor tab from disk."""
        editor = self._get_current_editor()
        if editor is None:
            return
        self._reload_tab(editor, self._get_current_filename() or "", self.tab_widget.currentIndex())

    @staticmethod
    def _copy_to_clipboard(text: str):
        """Copy *text* to the system clipboard."""
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(text)

    def load_file_event(self, event, filename: str = None, **kwargs):
        """Load a file from an event-compatible callback."""
        self.load_file(filename)

    def load_file(self, filename: str = None, **kwargs):
        """Load a file into the current or a new tab."""
        filename = filename or cs.gui.widgets.get_filename()
        if not filename:
            return

        filename_str = str(pathlib.Path(filename).resolve())

        if filename_str in self._open_files:
            editor = self._open_files[filename_str]
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
            return

        try:
            logging.log(0, f"Loading file: {filename_str}")
            with open(filename_str, encoding="utf-8") as file:
                content = file.read()
        except OSError as e:
            logging.log(1, f"Error loading file {filename_str}: {e}")
            return

        editor, _ = self._create_editor_tab(
            filename=filename_str,
            language=self._language_from_path(filename_str),
        )
        editor.blockSignals(True)
        editor.setText(content)
        editor.blockSignals(False)
        editor.set_current_file(filename_str)
        editor.document().setModified(False)
        self._open_files[filename_str] = editor
        self._sync_editor_document(editor)
        self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
        editor.refresh_symbols()
        self._notify_lsp_open(editor)

    def open_file(self, path: str, line: int = None, col: int = None):
        """Open a file in a new tab or switch to existing tab, optionally jump to line."""
        path_str = str(pathlib.Path(path).resolve())

        if path_str in self._open_files:
            editor = self._open_files[path_str]
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
        else:
            try:
                with open(path_str, encoding="utf-8") as file:
                    content = file.read()
            except OSError as e:
                logging.log(1, f"Error opening file {path_str}: {e}")
                return

            editor, _ = self._create_editor_tab(
                filename=path_str,
                language=self._language_from_path(path_str),
            )
            editor.blockSignals(True)
            editor.setText(content)
            editor.blockSignals(False)
            editor.set_current_file(path_str)
            editor.document().setModified(False)
            self._sync_editor_document(editor)
            self._open_files[path_str] = editor
            self.tab_widget.setCurrentIndex(self.tab_widget.indexOf(editor))
            editor.refresh_symbols()
            self._notify_lsp_open(editor)

        if line and line > 0:
            self.goto_line(line, col=col or 0)

    @staticmethod
    def _language_from_path(path: str) -> str:
        """Infer editor language from a path suffix."""
        suffix = pathlib.Path(path).suffix.lower()
        if suffix == ".json":
            return "JSON"
        if suffix in {".yaml", ".yml"}:
            return "YAML"
        if suffix in {".txt", ".md"}:
            return "Plain text"
        return "Python"

    def goto_line(self, line: int, col: int = 0):
        """Move the cursor to a specific line number in the current editor."""
        editor = self._get_current_editor()
        if editor is None:
            return

        if line < 1:
            return

        editor.goto_line_column(line, col)
        self._highlight_line_temporarily(editor, line)

    def _highlight_line_temporarily(self, editor, line: int, duration_ms: int = 2000):
        """Temporarily highlight a line in the given editor."""
        doc = editor.document()
        block = doc.findBlockByNumber(line - 1)
        if not block.isValid():
            return

        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.format.setBackground(QtGui.QColor(255, 255, 0, 100))
        selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
        selection.cursor = editor.textCursor()
        selection.cursor.setPosition(block.position())
        selection.cursor.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)

        extra_selections = editor.extraSelections() + [selection]
        editor.setExtraSelections(extra_selections)

        QtCore.QTimer.singleShot(
            duration_ms, lambda: self._clear_temporary_highlight(editor, selection)
        )

    def _clear_temporary_highlight(self, editor, selection: QtWidgets.QTextEdit.ExtraSelection):
        """Clear a temporary line highlight."""
        extra_selections = editor.extraSelections()
        if selection in extra_selections:
            extra_selections.remove(selection)
            editor.setExtraSelections(extra_selections)

    def _drain_rpc_events(self) -> None:
        """Apply editor document changes requested through RPC."""
        if self._rpc_server is not None:
            events = self._rpc_server.drain_events()
        else:
            events = []
            while True:
                try:
                    events.append(self._rpc_event_queue.get_nowait())
                except queue.Empty:
                    break
        for event in events:
            self._apply_rpc_event(event)

    def _apply_rpc_event(self, event: dict) -> None:
        """Apply a remote document-change event to the matching editor."""
        document_id = event.get("document_id")
        editor = None
        for widget, widget_id in self._document_ids.items():
            if widget_id == document_id:
                editor = widget
                break
        if editor is None or not isinstance(editor, TextEditor):
            return
        content = event.get("content")
        if content is None:
            return
        self._applying_remote_document = True
        try:
            editor.blockSignals(True)
            if event.get("path"):
                editor.set_current_file(event["path"])
            editor.setText(content)
            editor.blockSignals(False)
            editor.document().setModified(True)
            self._sync_editor_document(editor)
            editor.refresh_symbols()
            self._on_current_tab_changed(self.tab_widget.currentIndex())
        finally:
            self._applying_remote_document = False

    def run_ruff_current(self) -> None:
        """Run Ruff on the current editor document."""
        settings = get_editor_settings()
        if not bool(settings.get("enable_ruff", True)):
            self.lspStatusChanged.emit("Ruff disabled")
            return
        editor = self._get_current_editor()
        if editor is None or not isinstance(editor, TextEditor):
            return
        document_id = self._document_id_for_editor(editor)
        path = editor.current_file or document_id
        result = self.ruff_runner.check(
            path=path,
            content=editor.toPlainText(),
            extra_args=list(settings.get("ruff_extra_args", [])),
            timeout_ms=int(settings.get("ruff_timeout_ms", 5000)),
        )
        self._show_ruff_result(result)

    def _show_ruff_result(self, result: dict) -> None:
        """Display Ruff diagnostics in the shared diagnostics widget."""
        self.diagnostics_list.clear()
        if not isinstance(result, dict):
            self.lspStatusChanged.emit("Ruff failed")
            return
        if result.get("ok") is False:
            item = QtWidgets.QListWidgetItem(f"Ruff failed: {result.get('error', 'unknown error')}")
            item.setForeground(QtGui.QColor("#b00020"))
            self.diagnostics_list.addItem(item)
            self.lspStatusChanged.emit("Ruff failed")
            return
        diagnostics = result.get("diagnostics", [])
        if not diagnostics:
            item = QtWidgets.QListWidgetItem("Ruff: no issues found")
            item.setForeground(QtGui.QColor("#2e7d32"))
            self.diagnostics_list.addItem(item)
            self.lspStatusChanged.emit("Ruff clean")
            return
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            path = diagnostic.get("path") or result.get("path", "")
            line = int(diagnostic.get("line", 0) or 0)
            column = int(diagnostic.get("column", 0) or 0)
            code = diagnostic.get("code", "")
            message = diagnostic.get("message", "")
            prefix = f"{path}:{line}:{column}" if path and line else path
            item = QtWidgets.QListWidgetItem(f"{prefix} {code}: {message}")
            item.setData(QtCore.Qt.UserRole, diagnostic)
            self.diagnostics_list.addItem(item)
        self.lspStatusChanged.emit(f"Ruff: {len(diagnostics)} issue(s)")

    def _write_and_get_filepath(self, content: str, filename: str) -> str:
        """Write editor content to a file and return its path."""
        if filename and filename != "Untitled" and self._is_real_file(filename):
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                logging.log(1, f"Error writing {filename}: {e}")
                return ""
            return str(pathlib.Path(filename).resolve())
        fd, filepath = tempfile.mkstemp(suffix=".py", prefix="chisurf_run_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def run_macro(self, event):
        """Execute the current editor content without requiring a prior save."""
        editor = self._get_current_editor()
        if editor is None:
            return

        content = editor.toPlainText()
        filename = self._get_current_filename()
        filepath = self._write_and_get_filepath(content, filename)
        if not filepath:
            return

        settings = get_editor_settings()
        mode = settings.get("run_endpoint", "process")

        if mode == "console":
            self._run_console(content, filepath)
        else:
            self._run_process_impl(filepath)

    def _run_process_impl(self, filepath: str) -> None:
        """Run a file as a subprocess via QProcess."""
        self._run_process = QtCore.QProcess(self)
        self._run_process.finished.connect(self._on_run_finished)
        self._run_process.errorOccurred.connect(self._on_run_error)
        self._run_process.setProcessChannelMode(QtCore.QProcess.ForwardedChannels)
        self._run_process.started.connect(lambda: self.runStateChanged.emit(True))
        self._run_process.start(sys.executable, [filepath])

    def _run_console(self, content: str, filepath: str) -> None:
        """Run content in-process via exec wrapped in a stoppable thread."""
        self._run_thread = threading.Thread(
            target=self._exec_in_thread,
            args=(content, filepath),
            daemon=True,
        )
        self._run_thread.start()
        self.runStateChanged.emit(True)

    def _exec_in_thread(self, content: str, filepath: str) -> None:
        """Execute code in a thread, catching KeyboardInterrupt for stop."""
        namespace = {
            "__name__": "__main__",
            "__file__": filepath,
            "cs": cs,
            "np": __import__("numpy"),
            "os": os,
            "sys": sys,
        }
        try:
            exec(compile(content, filepath, "exec"), namespace)
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        finally:
            self._run_thread = None
            self.runStateChanged.emit(False)

    def stop_macro(self):
        """Stop a currently running script."""
        if self._run_process is not None and self._run_process.state() != QtCore.QProcess.NotRunning:
            self._run_process.kill()
            self._run_process.waitForFinished(3000)
            self._run_process = None
            self.runStateChanged.emit(False)
            return
        if self._run_thread is not None and self._run_thread.is_alive():
            _async_raise(self._run_thread.ident, KeyboardInterrupt)
            self._run_thread = None
            self.runStateChanged.emit(False)

    def _on_run_finished(self, _exit_code: int, _exit_status: QtCore.QProcess.ExitStatus) -> None:
        """Clean up after a script process finishes."""
        self._run_process = None
        self.runStateChanged.emit(False)

    def _on_run_error(self, _error: QtCore.QProcess.ProcessError) -> None:
        """Handle errors from the script process."""
        self._run_process = None
        self.runStateChanged.emit(False)
        logging.log(1, "Failed to start script process.")

    def save_text(self, event=None):
        """Save the current tab's text to a file."""
        editor = self._get_current_editor()
        if editor is None:
            return

        filename = self._get_current_filename()
        if filename == "Untitled" or not filename:
            new_filename = cs.gui.widgets.save_file(file_type="Python script (*.py)")
            if not new_filename:
                return
            filename = str(new_filename)
            idx = self.tab_widget.currentIndex()
            self.tab_widget.setTabText(idx, filename)
            self._open_files[str(filename)] = editor

        try:
            with io.zipped.open_maybe_zipped(filename, "w") as file:
                file.write(editor.text())
        except OSError as e:
            logging.log(1, f"Error saving file {filename}: {e}")
            return
        editor.set_current_file(str(filename))
        editor.document().setModified(False)
        self._sync_editor_document(editor)
        if get_editor_settings().get("run_ruff_on_save", False):
            self.run_ruff_current()


__all__ = ["CodeEditor"]
