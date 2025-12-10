"""Tests for safe deletion of nodes and edges (no crashes/segfaults)."""

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.model import NodeModel, PortSpec
from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def scene(app):  # noqa: ARG001 - app fixture ensures QApplication exists
    return NodeScene()


def _make_two_nodes_with_edge(scene: NodeScene):
    """Helper to create two nodes connected by a single edge."""
    model1 = NodeModel(
        title="Node 1",
        inputs=[PortSpec(name="In", is_output=False)],
        outputs=[PortSpec(name="Out", is_output=True)],
        node_type="test",
        config={},
    )
    model2 = NodeModel(
        title="Node 2",
        inputs=[PortSpec(name="In", is_output=False)],
        outputs=[PortSpec(name="Out", is_output=True)],
        node_type="test",
        config={},
    )

    item1 = NodeGraphicsItem(model1)
    item2 = NodeGraphicsItem(model2)
    scene.addItem(item1)
    scene.addItem(item2)

    # One input, one output each -> output is index 1, input is index 0
    edge = EdgeGraphicsItem(item1.port_items[1], item2.port_items[0])
    scene.addItem(edge)
    scene.register_edge(edge)

    return item1, item2, edge


def test_delete_key_removes_edges_and_nodes(scene):
    """Pressing Delete on selected node+edge removes both safely."""
    item1, item2, edge = _make_two_nodes_with_edge(scene)

    # Select node and edge
    item1.setSelected(True)
    edge.setSelected(True)

    # Simulate Delete key press
    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Delete,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(event)

    # No edges left in bookkeeping or scene items
    assert len(scene.edges) == 0
    assert not any(isinstance(it, EdgeGraphicsItem) for it in scene.items())


def test_delete_key_with_only_node_selected(scene):
    """Deleting a node with attached edge should also remove the edge safely."""
    item1, item2, edge = _make_two_nodes_with_edge(scene)

    item1.setSelected(True)

    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Delete,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(event)

    assert len(scene.edges) == 0
    assert not any(isinstance(it, EdgeGraphicsItem) for it in scene.items())


def test_delete_key_with_only_edge_selected(scene):
    """Deleting only an edge should not crash and must update bookkeeping."""
    item1, item2, edge = _make_two_nodes_with_edge(scene)

    edge.setSelected(True)

    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Delete,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(event)

    assert len(scene.edges) == 0
    assert not any(isinstance(it, EdgeGraphicsItem) for it in scene.items())
