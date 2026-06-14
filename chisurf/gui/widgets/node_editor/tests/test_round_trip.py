"""Unit tests for node editor JSON round-trip serialization."""

import pytest
from qtpy import QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.scene import NodeScene


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


def test_node_ids_and_edge_metadata_round_trip(editor):
    """Round-trip a graph with explicit node IDs and edge metadata."""
    graph = {
        "version": 1,
        "meta": {"purpose": "provenance_view", "schema_name": "test"},
        "nodes": [
            {
                "id": "raw:1",
                "title": "Raw",
                "inputs": [{"name": "in", "type": "mfdb"}],
                "outputs": [{"name": "out", "type": "mfdb"}],
                "type": "mfdb_record",
                "config": {"record": {"node_type": "raw_data", "node_id": "1"}, "extra": 1},
                "pos": [0.0, 0.0],
                "collapsed": False,
                "z": 1.0,
            },
            {
                "id": "product:1",
                "title": "Product",
                "inputs": [{"name": "in", "type": "mfdb"}],
                "outputs": [{"name": "out", "type": "mfdb"}],
                "type": "mfdb_record",
                "config": {"record": {"node_type": "processed_data", "node_id": "1"}, "extra": 2},
                "pos": [260.0, 0.0],
                "collapsed": False,
                "z": 1.0,
            },
        ],
        "edges": [
            {
                "source": "raw:1",
                "source_port": 1,
                "target": "product:1",
                "target_port": 0,
                "config": {
                    "edge_id": "edge-1",
                    "relationship_type": "produced",
                    "metadata": {"source": "mfdb"},
                    "color": [70, 180, 100],
                },
            }
        ],
    }

    editor.clear_graph()
    editor.load_graph_dict(graph)

    loaded = editor.graph_dict()
    assert {node["id"] for node in loaded["nodes"]} == {"raw:1", "product:1"}
    assert loaded["meta"] == graph["meta"]
    assert loaded["edges"][0]["config"]["edge_id"] == "edge-1"
    assert loaded["edges"][0]["config"]["metadata"] == {"source": "mfdb"}
    assert loaded["edges"][0]["config"]["color"] == [70, 180, 100]


def test_load_graph_dict_loads_valid_graph(editor):
    """load_graph_dict should load a valid graph and fit the view."""
    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "title": "Node",
                "inputs": [],
                "outputs": [],
                "type": "text_note",
                "config": {"label": "Note", "text": "hello"},
                "pos": [0.0, 0.0],
                "collapsed": False,
            }
        ],
        "edges": [],
    }

    editor.clear_graph()
    editor.load_graph_dict(graph)

    assert len([item for item in editor.scene.items() if isinstance(item, NodeGraphicsItem)]) == 1


def test_invalid_json_handling(editor):
    """Test that invalid JSON is handled gracefully."""
    editor.load_graph_from_json("{invalid")

    editor.load_graph_from_json('{"nodes": "invalid"}')
