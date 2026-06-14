from __future__ import annotations

from qtpy import QtGui, QtWidgets

from chisurf.plugins.core.code_editor import CodeEditor
from chisurf.plugins.core.code_editor.context_retriever import retrieve_context
from chisurf.plugins.core.code_editor.validation import validate_writes
from chisurf.plugins.core.code_editor.wiki_indexer import build_api_index


def test_code_editor_syncs_open_document_to_store(qapp, tmp_path) -> None:
    """CodeEditor mirrors open tabs into the JSON document store."""
    path = tmp_path / "sample.py"
    path.write_text("x = 1\n", encoding="utf-8")
    widget = CodeEditor(can_load=False, enable_lsp=False)

    widget.open_file(str(path))
    editor = widget._get_current_editor()
    editor.moveCursor(QtGui.QTextCursor.End)
    editor.insertPlainText("y = 2\n")

    document = widget.document_store.get(path=str(path))

    assert document is not None
    assert document.content == "x = 1\ny = 2\n"
    assert document.modified is True


def test_code_editor_run_ruff_current_displays_result(qapp, monkeypatch) -> None:
    """The Run Ruff action uses the current editor content."""
    widget = CodeEditor(enable_lsp=False)
    editor = widget._get_current_editor()
    editor.setText("import os\n")

    def fake_check(self, path, content, extra_args, timeout_ms):
        assert content == "import os\n"
        return {
            "ok": True,
            "path": str(path),
            "diagnostics": [
                {
                    "path": str(path),
                    "line": 1,
                    "column": 0,
                    "code": "F401",
                    "message": "`os` imported but unused",
                }
            ],
        }

    monkeypatch.setattr(type(widget.ruff_runner), "check", fake_check)

    widget.run_ruff_current()

    assert widget.diagnostics_list.count() == 1


def test_agent_write_file_current_updates_untitled_document(qapp) -> None:
    """Agent WRITE_FILE blocks targeting current update the open untitled tab."""
    widget = CodeEditor(enable_lsp=False)
    editor = widget._get_current_editor()
    response = """```markdown
# WRITE_FILE: current
# Theory of FRET

Forster resonance energy transfer depends on donor-acceptor distance.
```"""

    widget.agent_panel._apply_file_writes(response)

    assert editor.toPlainText().startswith("# Theory of FRET")
    assert "Forster resonance energy transfer" in editor.toPlainText()
    assert widget.tab_widget.currentWidget() is editor


def test_agent_write_file_parser_does_not_fold_content_into_filename(qapp) -> None:
    """The WRITE_FILE target is limited to the first header line."""
    widget = CodeEditor(enable_lsp=False)
    editor = widget._get_current_editor()
    response = """```python
# WRITE_FILE: untitled document
print("written")
```"""

    widget.agent_panel._apply_file_writes(response)

    assert editor.toPlainText() == 'print("written")'


def test_close_dirty_tab_cancel_keeps_tab_open(qapp, monkeypatch) -> None:
    """Closing a modified tab asks first and honors Cancel."""
    widget = CodeEditor(enable_lsp=False)
    editor = widget._get_current_editor()
    editor.setPlainText("changed")
    editor.document().setModified(True)
    widget.tab_widget.setTabText(widget.tab_widget.currentIndex(), "Untitled")
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec_",
        lambda self: QtWidgets.QMessageBox.Cancel,
    )

    widget._close_tab(widget.tab_widget.currentIndex())

    assert widget._get_current_editor() is editor
    assert editor.document().isModified() is True


def test_close_dirty_untitled_save_cancel_keeps_tab_open(qapp, monkeypatch) -> None:
    """Cancelling Save As while closing a dirty untitled tab does not close it."""
    widget = CodeEditor(enable_lsp=False)
    editor = widget._get_current_editor()
    editor.setPlainText("changed")
    editor.document().setModified(True)
    widget.tab_widget.setTabText(widget.tab_widget.currentIndex(), "Untitled")
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec_",
        lambda self: QtWidgets.QMessageBox.Save,
    )
    monkeypatch.setattr(
        "chisurf.plugins.core.code_editor.editor.cs.gui.widgets.save_file",
        lambda file_type: None,
    )

    widget._close_tab(widget.tab_widget.currentIndex())

    assert widget._get_current_editor() is editor
    assert editor.document().isModified() is True


def test_agent_no_write_response_does_not_validate(qapp, monkeypatch) -> None:
    """Responses without WRITE_FILE do not trigger compile or ruff validation."""
    widget = CodeEditor(enable_lsp=False)

    def fail_validate(*_args, **_kwargs):
        raise AssertionError("validate_writes should not be called")

    monkeypatch.setattr(
        "chisurf.plugins.core.code_editor.agent_panel.validate_writes",
        fail_validate,
    )

    writes = widget.agent_panel._apply_file_writes("This is just an explanation.")
    assert writes == []


def test_validate_writes_detects_py_compile_errors(tmp_path) -> None:
    """Generated code is checked with py_compile."""
    issues = validate_writes([("broken.py", "def broken(:\n    pass\n")], editor=None)

    assert issues
    assert issues[0][0] == "broken.py"
    assert issues[0][1][0]["code"] == "E999"


def test_validate_writes_uses_ruff_diagnostics(qapp, tmp_path, monkeypatch) -> None:
    """Generated code is checked with the editor ruff runner."""
    widget = CodeEditor(enable_lsp=False)

    def fake_check(self, path, content, extra_args, timeout_ms):
        assert content == "import os\n"
        return {
            "ok": True,
            "path": str(path),
            "diagnostics": [
                {
                    "path": str(path),
                    "line": 1,
                    "column": 0,
                    "code": "F401",
                    "message": "`os` imported but unused",
                }
            ],
        }

    monkeypatch.setattr(type(widget.ruff_runner), "check", fake_check)
    issues = validate_writes([("sample.py", "import os\n")], editor=widget)

    assert issues
    assert issues[0][1][0]["code"] == "F401"


def test_api_retriever_builds_context_from_source_index(tmp_path) -> None:
    """The retriever indexes real source symbols and returns them as context."""
    package = tmp_path / "chisurf" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api.py").write_text(
        "class Runner:\n"
        "    \"\"\"Runs a demo analysis.\"\"\"\n"
        "    def run(self, name):\n"
        "        return name\n",
        encoding="utf-8",
    )

    build_api_index(tmp_path)
    context = retrieve_context("run demo runner", repo_root=tmp_path)

    assert "Verified ChiSurf API Context" in context
    assert "demo.api.Runner.run" in context
    assert "Runs a demo analysis" in context
