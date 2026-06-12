from qtpy import QtCore

import chisurf as cs


class TestCodeEditor:
    """Tests for the shared and standalone code editor widgets."""

    def test_creation(self, qapp):
        """The shared editor widget can be created."""
        from chisurf.plugins.core.code_editor import CodeEditor
        widget = CodeEditor()
        assert widget is not None

    def test_creation_with_filename(self, qapp):
        """The shared editor accepts a filename at construction."""
        from chisurf.plugins.core.code_editor import CodeEditor
        widget = CodeEditor(filename="test.py", language="Python")
        assert widget is not None

    def test_creation_uses_global_lsp_setting(self, qapp, monkeypatch):
        """The shared editor reads the global LSP setting when not overridden."""
        from chisurf.plugins.core.code_editor import CodeEditor

        monkeypatch.setattr(cs.core.settings, "cs_settings", {"gui": {"editor": {"enable_lsp": False}}})
        monkeypatch.setattr(cs.core.settings, "gui", {"editor": {"enable_lsp": False}})

        widget = CodeEditor(can_load=False, enable_lsp=None)

        assert widget._enable_lsp is False

    def test_load_returns_main_window(self, qapp):
        """The plugin load function returns the full main-window shell."""
        from chisurf.gui import QtWidgets
        from chisurf.plugins.core import code_editor

        widget = code_editor.load()

        assert isinstance(widget, QtWidgets.QMainWindow)
        assert widget.menuBar() is not None
        assert widget.statusBar() is not None
        assert widget.findChild(QtWidgets.QToolBar, "code_editor_toolbar") is not None

        menus = [action.menu() for action in widget.menuBar().actions() if action.menu()]
        settings_menu = next(menu for menu in menus if menu.title() == "Settings")
        action_names = {action.objectName() for action in settings_menu.actions()}
        assert "toggle_line_numbers" in action_names
        assert "toggle_lsp" in action_names

    def test_text_editor_extracts_symbols_and_jumps(self, qapp):
        """The single-document editor extracts symbols and jumps to them."""
        from chisurf.plugins.core.code_editor.text_editor import TextEditor

        editor = TextEditor(language="Python")
        editor.setText(
            "class Alpha:\n"
            "    def beta(self):\n"
            "        pass\n\n"
            "def gamma():\n"
            "    pass\n"
        )
        symbols = editor.refresh_symbols()

        assert [symbol.display_name for symbol in symbols] == ["Alpha", "Alpha.beta", "gamma"]

        editor.goto_symbol(symbols[-1])
        assert editor.line_column()[0] == 5

    def test_text_editor_status_tracks_dirty_line_column_and_file(self, qapp, tmp_path):
        """The single-document editor emits file, cursor, and dirty state."""
        from chisurf.plugins.core.code_editor.text_editor import TextEditor

        path = tmp_path / "sample.py"
        editor = TextEditor(language="Python")
        statuses = []
        editor.statusChanged.connect(statuses.append)

        editor.set_current_file(str(path))
        editor.setText("a = 1\nb = 2\n")
        editor.insertPlainText("c")
        editor.goto_line_column(2, 1)

        assert statuses[-1]["file"] == str(path)
        assert statuses[-1]["line"] == 2
        assert statuses[-1]["column"] == 1
        assert editor.document().isModified()

    def test_code_editor_opens_files_and_symbol_click_jumps(self, qapp, tmp_path):
        """The shared editor opens files and jumps from outline items."""
        from chisurf.plugins.core.code_editor import CodeEditor

        path = tmp_path / "sample.py"
        path.write_text(
            "class Alpha:\n"
            "    def beta(self):\n"
            "        pass\n\n"
            "def gamma():\n"
            "    pass\n",
            encoding="utf-8",
        )
        widget = CodeEditor(can_load=False, project_root=tmp_path, enable_lsp=False)

        widget.open_file(str(path), line=1)
        editor = widget._get_current_editor()

        assert editor is not None
        assert editor.current_file == str(path)
        assert str(path) in widget._open_files

        items = widget.symbol_tree.findItems("gamma", QtCore.Qt.MatchExactly, 0)
        assert items
        widget._on_symbol_item_activated(items[0], 0)

        assert editor.line_column()[0] == 5

    def test_code_editor_file_tree_click_opens_file(self, qapp, tmp_path):
        """The shared project tree opens clicked files."""
        from chisurf.plugins.core.code_editor import CodeEditor

        path = tmp_path / "clicked.py"
        path.write_text("x = 1\n", encoding="utf-8")
        widget = CodeEditor(can_load=False, project_root=tmp_path, enable_lsp=False)
        qapp.processEvents()

        index = widget.file_model.index(str(path))
        assert index.isValid()

        widget._on_file_tree_activated(index)

        assert str(path) in widget._open_files

    def test_lsp_client_handles_fake_response(self, qapp, tmp_path):
        """The LSP client dispatches JSON-RPC responses to callbacks."""
        from chisurf.plugins.core.code_editor.lsp_client import PythonLspClient

        client = PythonLspClient(root_path=tmp_path, command=["pylsp"])
        results = []

        request_id = client.request("fake/method", {}, results.append)
        client._handle_message({"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}})

        assert results == [{"ok": True}]
