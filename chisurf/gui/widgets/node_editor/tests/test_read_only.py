"""Tests for read-only node editor behavior."""

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem
from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _node_count(scene) -> int:
    return len([item for item in scene.items() if isinstance(item, NodeGraphicsItem)])


def _edge_count(scene) -> int:
    return len([item for item in scene.items() if isinstance(item, EdgeGraphicsItem)])


def _first_node(editor: NodeEditorWidget) -> NodeGraphicsItem:
    return next(item for item in editor.scene.items() if isinstance(item, NodeGraphicsItem))


def _first_edge(editor: NodeEditorWidget) -> EdgeGraphicsItem:
    return next(item for item in editor.scene.items() if isinstance(item, EdgeGraphicsItem))


def _graph() -> dict:
    return {
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "title": "Node 1",
                "inputs": [{"name": "in", "type": "mfdb"}],
                "outputs": [{"name": "out", "type": "mfdb"}],
                "type": "mfdb_record",
                "config": {"node_type": "raw_data", "node_id": "raw_1"},
                "pos": [0.0, 0.0],
                "collapsed": False,
                "z": 1.0,
            },
            {
                "id": "n2",
                "title": "Node 2",
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
                "source": "n1",
                "source_port": 1,
                "target": "n2",
                "target_port": 0,
                "config": {"relationship_type": "produced"},
            }
        ],
    }


def test_editor_constructor_options_hide_demo_parts(qapp):
    """Constructor flags keep the editor reusable without demo UI."""
    del qapp
    editor = NodeEditorWidget(build_example=False, show_side_panel=False, show_timeline=False)

    assert _node_count(editor.scene) == 0
    assert not hasattr(editor, "widget_palette")
    assert not hasattr(editor, "json_edit")
    assert editor.timeline is None
    assert editor.undo_stack is not None


def test_read_only_blocks_edge_creation(qapp, monkeypatch):
    """Read-only scenes do not start temporary edge creation from ports."""
    del qapp
    editor = NodeEditorWidget(build_example=False, read_only=True)
    editor.load_graph_dict(_graph())
    node = _first_node(editor)

    def noop_mouse_press(_scene, _event):
        return None

    monkeypatch.setattr(QtWidgets.QGraphicsScene, "mousePressEvent", noop_mouse_press)

    class MockMouseEvent:
        def scenePos(self):
            return node.port_items[1].scene_pos()

        def button(self):
            return QtCore.Qt.LeftButton

        def accept(self):
            return None

    editor.scene.mousePressEvent(MockMouseEvent())

    assert editor.scene._current_edge is None


def test_read_only_delete_keeps_nodes_and_edges(qapp):
    """Delete key is ignored in read-only mode while selection remains available."""
    del qapp
    editor = NodeEditorWidget(build_example=False, read_only=True)
    editor.load_graph_dict(_graph())
    node = _first_node(editor)
    edge = _first_edge(editor)
    node.setSelected(True)
    edge.setSelected(True)

    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Delete,
        QtCore.Qt.NoModifier,
    )
    editor.scene.keyPressEvent(event)

    assert _node_count(editor.scene) == 2
    assert len(editor.scene.edges) == 1


def test_read_only_blocks_paste_and_duplicate(qapp):
    """Paste and Shift+D duplicate shortcuts are blocked in read-only mode."""
    del qapp
    editor = NodeEditorWidget(build_example=False, read_only=True)
    editor.load_graph_dict(_graph())
    node = _first_node(editor)
    node.setSelected(True)

    copy_event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_C,
        QtCore.Qt.ControlModifier,
    )
    editor.scene.keyPressEvent(copy_event)
    before_paste = _node_count(editor.scene)

    paste_event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_V,
        QtCore.Qt.ControlModifier,
    )
    editor.scene.keyPressEvent(paste_event)
    duplicate_event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_D,
        QtCore.Qt.ShiftModifier,
    )
    editor.scene.keyPressEvent(duplicate_event)
    after = _node_count(editor.scene)

    assert after == before_paste


def test_read_only_keeps_node_selection(qapp):
    """Read-only mode still allows selecting nodes."""
    del qapp
    editor = NodeEditorWidget(build_example=False, read_only=True)
    editor.load_graph_dict(_graph())
    node = _first_node(editor)

    node.setSelected(True)

    assert node.isSelected()


def test_read_only_loaded_nodes_are_not_movable(qapp):
    """Loaded nodes in read-only scenes cannot be moved."""
    del qapp
    editor = NodeEditorWidget(build_example=False, read_only=True)
    editor.load_graph_dict(_graph())
    node = _first_node(editor)

    assert not node.flags() & QtWidgets.QGraphicsItem.ItemIsMovable
