import pytest
from qtpy import QtWidgets

pytest.importorskip("chinet")

from chisurf.gui.widgets import node_editor as gui_node_editor


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def editor(app):  # noqa: ARG001 - ensures QApplication exists
    return gui_node_editor.NodeEditorWidget()


def test_gui_widgets_node_edit_round_trip(editor, tmp_path):
    editor._build_example_graph()

    json_str = editor.to_json()
    assert isinstance(json_str, str)
    assert json_str

    editor.clear_graph()
    editor.load_graph_from_json(json_str)

    items = list(editor.scene.items())
    assert items


def test_gui_widgets_node_edit_file_io(editor, tmp_path):
    editor._build_example_graph()

    path = tmp_path / "graph.json"
    editor.save_graph_to_file(str(path))
    assert path.exists()

    editor.clear_graph()
    editor.load_graph_from_file(str(path))

    items = list(editor.scene.items())
    assert items


def test_gui_widgets_node_edit_help_dialog(editor):
    dlg = editor.show_help_dialog()
    try:
        assert dlg is not None
        edits = dlg.findChildren(QtWidgets.QTextEdit)
        assert edits
        text = edits[0].toPlainText()
        assert "Node Editor" in text
    finally:
        dlg.close()
