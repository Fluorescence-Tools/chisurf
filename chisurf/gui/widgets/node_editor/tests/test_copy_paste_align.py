"""Tests for copy/paste and alignment behaviour in the node editor scene."""

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.model import NodeModel, PortSpec


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def scene(app):  # noqa: ARG001 - ensures QApplication exists
    return NodeScene()


def _count_nodes(scene: NodeScene) -> int:
    return len([it for it in scene.items() if isinstance(it, NodeGraphicsItem)])


def _make_simple_node(scene: NodeScene, x: float, y: float, title: str = "N") -> NodeGraphicsItem:
    model = NodeModel(
        title=title,
        inputs=[PortSpec(name="In", is_output=False)],
        outputs=[PortSpec(name="Out", is_output=True)],
        node_type="test",
        config={"value": 42.0},
    )
    item = NodeGraphicsItem(model)
    scene.addItem(item)
    item.setPos(x, y)
    return item


def test_copy_paste_single_node(scene):
    """Ctrl+C / Ctrl+V on a single node should duplicate it with same config."""
    n0 = _count_nodes(scene)
    node = _make_simple_node(scene, 0.0, 0.0, title="A")
    n1 = _count_nodes(scene)
    assert n1 == n0 + 1

    # Select and copy
    node.setSelected(True)
    ev_copy = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_C,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_copy)

    clipboard = getattr(scene, "_clipboard", [])
    assert len(clipboard) == 1
    model_copy, offset = clipboard[0]
    assert model_copy.title == "A"
    assert model_copy.config.get("value") == 42.0

    # Paste
    ev_paste = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_V,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_paste)

    n2 = _count_nodes(scene)
    assert n2 == n1 + 1


def test_copy_paste_multiple_nodes(scene):
    """Copy/paste with multiple selected nodes should preserve count and titles."""
    n0 = _count_nodes(scene)
    n1_item = _make_simple_node(scene, 0.0, 0.0, title="A")
    n2_item = _make_simple_node(scene, 50.0, 20.0, title="B")

    for it in (n1_item, n2_item):
        it.setSelected(True)

    ev_copy = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_C,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_copy)

    clipboard = getattr(scene, "_clipboard", [])
    assert len(clipboard) == 2

    ev_paste = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_V,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_paste)

    n_after = _count_nodes(scene)
    assert n_after == n0 + 4  # two originals + two pasted


def _get_rects(scene: NodeScene):
    nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]
    return nodes, [n.sceneBoundingRect() for n in nodes]


def test_align_left_right_top_bottom(scene):
    """_align_nodes should correctly align by edges."""
    a = _make_simple_node(scene, 0.0, 0.0)
    b = _make_simple_node(scene, 50.0, 30.0)
    c = _make_simple_node(scene, 100.0, -10.0)

    nodes = [a, b, c]

    # Left
    scene._align_nodes(nodes, mode="left")
    nodes, rects = _get_rects(scene)
    lefts = {round(r.left(), 4) for r in rects}
    assert len(lefts) == 1

    # Right
    scene._align_nodes(nodes, mode="right")
    nodes, rects = _get_rects(scene)
    rights = {round(r.right(), 4) for r in rects}
    assert len(rights) == 1

    # Top
    scene._align_nodes(nodes, mode="top")
    nodes, rects = _get_rects(scene)
    tops = {round(r.top(), 4) for r in rects}
    assert len(tops) == 1

    # Bottom
    scene._align_nodes(nodes, mode="bottom")
    nodes, rects = _get_rects(scene)
    bottoms = {round(r.bottom(), 4) for r in rects}
    assert len(bottoms) == 1


def test_align_centers(scene):
    """_align_nodes vcenter/hcenter should align centers along respective axes."""
    a = _make_simple_node(scene, 0.0, 0.0)
    b = _make_simple_node(scene, 80.0, 40.0)
    c = _make_simple_node(scene, -30.0, -20.0)

    nodes = [a, b, c]

    # Vertical center (x-centers equal)
    scene._align_nodes(nodes, mode="vcenter")
    nodes, rects = _get_rects(scene)
    x_centers = {round(r.center().x(), 4) for r in rects}
    assert len(x_centers) == 1

    # Horizontal center (y-centers equal)
    scene._align_nodes(nodes, mode="hcenter")
    nodes, rects = _get_rects(scene)
    y_centers = {round(r.center().y(), 4) for r in rects}
    assert len(y_centers) == 1
