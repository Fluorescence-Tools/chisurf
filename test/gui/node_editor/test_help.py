"""Tests for the node editor help dialog."""

import pytest
from qtpy import QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def editor(app):  # noqa: ARG001 - ensures QApplication exists
    return NodeEditorWidget()


def test_show_help_dialog_loads_readme(editor):
    """Help dialog should open and contain README text."""
    dlg = editor.show_help_dialog()
    try:
        assert dlg is not None
        # Find the QTextEdit child and ensure it has some content
        edits = dlg.findChildren(QtWidgets.QTextEdit)
        assert edits, "No QTextEdit found in help dialog"
        text = edits[0].toPlainText()
        assert "Node Editor" in text
    finally:
        dlg.close()
