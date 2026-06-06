"""Unit tests for node editor JSON round-trip serialization."""

import pytest
from qtpy import QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem


@pytest.fixture
def app():
    """Create a QApplication for testing.

    Reuse an existing instance if present to avoid multiple QApplication
    creations, which are not supported by Qt in one process.
    """

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def editor(app):
    """Create a NodeEditorWidget for testing."""
    return NodeEditorWidget()


def test_scene_to_dict_from_dict_round_trip(editor):
    """Test that to_dict -> from_dict preserves scene structure."""
    # Build a simple scene
    editor._build_example_graph()

    # Get initial state
    initial_items = [it for it in editor.scene.items() if isinstance(it, NodeGraphicsItem)]
    initial_edges = len(editor.scene.edges)

    # Serialize
    data = editor.scene.to_dict()

    # Clear and reload
    editor.scene.clear()
    editor.scene.from_dict(data)

    # Check restored state
    restored_items = [it for it in editor.scene.items() if isinstance(it, NodeGraphicsItem)]
    restored_edges = len(editor.scene.edges)

    assert len(restored_items) == len(initial_items)
    assert restored_edges == initial_edges

    # Check node properties (positions may vary due to sorting)
    initial_titles = sorted(it.model.title for it in initial_items)
    restored_titles = sorted(it.model.title for it in restored_items)
    assert restored_titles == initial_titles


def test_editor_json_round_trip(editor):
    """Test that to_json -> load_graph_from_json preserves scene."""
    # Build example
    editor._build_example_graph()

    # Serialize
    json_str = editor.to_json()

    # Clear and reload
    editor.clear_graph()
    editor.load_graph_from_json(json_str)

    # Check items are restored
    items = [it for it in editor.scene.items() if isinstance(it, NodeGraphicsItem)]
    assert len(items) > 0
    assert len(editor.scene.edges) > 0


def test_file_io(editor, tmp_path):
    """Test save_graph_to_file and load_graph_from_file."""
    # Build example
    editor._build_example_graph()

    filepath = tmp_path / "test_graph.json"

    # Save
    editor.save_graph_to_file(str(filepath))
    assert filepath.exists()

    # Clear and load
    editor.clear_graph()
    editor.load_graph_from_file(str(filepath))

    # Check restored
    items = [it for it in editor.scene.items() if isinstance(it, NodeGraphicsItem)]
    assert len(items) > 0
    assert len(editor.scene.edges) > 0


def test_empty_scene_round_trip(app):  # noqa: ARG001 - ensures QApplication exists
    """Test round-trip with an actually empty scene."""

    scene = NodeScene()
    data = scene.to_dict()
    assert data == {"nodes": [], "edges": [], "version": 1}

    scene.from_dict(data)
    items = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]
    assert len(items) == 0
    assert len(scene.edges) == 0


def test_invalid_json_handling(editor):
    """Test that invalid JSON is handled gracefully."""
    # Invalid JSON
    editor.load_graph_from_json("{invalid")

    # Scene should remain empty or unchanged
    items = [it for it in editor.scene.items() if isinstance(it, NodeGraphicsItem)]
    # May have example graph, but no crash

    # Invalid data
    editor.load_graph_from_json('{"nodes": "invalid"}')
    # Should not crash
