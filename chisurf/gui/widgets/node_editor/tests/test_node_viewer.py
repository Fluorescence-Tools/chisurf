"""Tests for the read-only NodeViewerWidget abstraction."""

from qtpy import QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.node_viewer import NodeViewerWidget


def _graph() -> dict:
    return {
        "version": 1,
        "meta": {"purpose": "provenance_view"},
        "nodes": [
            {
                "id": "raw:1",
                "title": "Raw",
                "inputs": [{"name": "in", "type": "mfdb"}],
                "outputs": [{"name": "out", "type": "mfdb"}],
                "type": "mfdb_record",
                "config": {"node_type": "raw_data", "node_id": "raw_1"},
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
                "config": {"node_type": "processed_data", "node_id": "prod_1"},
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
                "config": {"relationship_type": "produced"},
            }
        ],
    }


def test_node_viewer_widget_is_read_only_graph_component(qapp):
    """NodeViewerWidget loads graphs without editor-only demo UI."""
    del qapp
    viewer = NodeViewerWidget(read_only=True)

    assert isinstance(viewer, QtWidgets.QWidget)
    assert viewer.scene.read_only is True
    assert not hasattr(viewer, "widget_palette")
    assert not hasattr(viewer, "json_edit")
    assert not hasattr(viewer, "undo_stack")


def test_node_viewer_widget_api_signals_and_client(qapp):
    """NodeViewerWidget exposes the promised abstract viewer API."""
    del qapp
    fake_client = object()
    viewer = NodeViewerWidget(show_toolbar=True, graph_purpose="workflow", client=fake_client)
    loaded = []
    failed = []
    viewer.graphLoaded.connect(lambda data: loaded.append(data))
    viewer.graphLoadFailed.connect(lambda data: failed.append(data))

    assert viewer.graph_purpose == "workflow"
    assert viewer.client is fake_client
    assert viewer.toolbar is not None
    assert viewer.selected_node() == {}
    assert viewer.selected_edge() == {}

    viewer.load_graph_dict(_graph())
    assert loaded[-1]["meta"] == {"purpose": "provenance_view"}

    viewer.load_graph_from_json("{invalid")
    assert failed[-1]["phase"] == "parse_graph_json"

    viewer.set_toolbar_visible(False)
    assert viewer.toolbar.isVisible() is False


def test_node_editor_default_layout_has_managed_side_panel(qapp, capsys):
    """Default editor construction installs one layout and exposes side-panel widgets."""
    del qapp
    editor = NodeEditorWidget(build_example=False, show_side_panel=True, show_timeline=False)

    output = capsys.readouterr()
    assert "Attempting to add QLayout" not in output.err
    assert editor.layout() is not None
    assert editor.widget_palette is not None
    assert editor.json_edit is not None
    assert hasattr(editor, "widget_palette")
    assert hasattr(editor, "json_edit")


def test_node_viewer_widget_loads_and_serializes_graph(qapp):
    """NodeViewerWidget exposes the same graph dict API as NodeEditorWidget."""
    del qapp
    viewer = NodeViewerWidget()
    graph = _graph()

    viewer.load_graph_dict(graph)
    loaded = viewer.graph_dict()

    assert {node["id"] for node in loaded["nodes"]} == {"raw:1", "product:1"}
    assert loaded["edges"][0]["source"] == "raw:1"
    assert loaded["meta"] == graph["meta"]


def test_node_viewer_widget_emits_selection_signals(qapp):
    """NodeViewerWidget emits node and edge selection payloads."""
    del qapp
    viewer = NodeViewerWidget()
    viewer.load_graph_dict(_graph())

    selected_node = {}
    selected_edge = {}
    viewer.nodeSelected.connect(lambda data: selected_node.update(data))
    viewer.edgeSelected.connect(lambda data: selected_edge.update(data))

    node = next(
        item
        for item in viewer.scene.items()
        if item.__class__.__name__ == "NodeGraphicsItem" and getattr(item.model, "id", None) == "raw:1"
    )
    edge = viewer.scene.edges[0]
    node.setSelected(True)
    edge.setSelected(True)

    assert selected_node["id"] == "raw:1"
    assert viewer.selected_node()["id"] == "raw:1"
    assert selected_edge["source"] == "raw:1"
    assert viewer.selected_edge()["target"] == "product:1"
    assert selected_edge["target"] == "product:1"


def test_node_editor_widget_remains_node_viewer_subclass(qapp):
    """NodeEditorWidget remains compatible with the NodeViewer API."""
    del qapp
    editor = NodeEditorWidget(build_example=False, show_side_panel=False, show_timeline=False)

    assert isinstance(editor, NodeViewerWidget)
    assert editor.load_graph_dict({"version": 1, "nodes": [], "edges": []}) is None
    assert editor.graph_dict() == {"nodes": [], "edges": [], "version": 1}
