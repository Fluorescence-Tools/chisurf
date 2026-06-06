"""Tests for undo/redo behaviour in the node editor."""

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
from chisurf.gui.widgets.node_editor.registry import registry


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def editor(app):  # noqa: ARG001 - ensures QApplication exists
    # Ensure a clean registry so built-in node types can be (re)registered
    registry._types.clear()
    return NodeEditorWidget()


def _count_nodes(scene):
    return len([it for it in scene.items() if isinstance(it, NodeGraphicsItem)])


def test_undo_redo_add_node_via_shortcut(editor):
    scene = editor.scene
    before = _count_nodes(scene)
    # Use the editor's node creation callback directly to avoid depending on
    # focus/key handling details in this test.
    editor._on_add_node_requested("constant", QtCore.QPointF(0.0, 0.0))  # type: ignore[attr-defined]

    after_add = _count_nodes(scene)
    assert after_add == before + 1

    editor.undo()
    after_undo = _count_nodes(scene)
    assert after_undo == before

    editor.redo()
    after_redo = _count_nodes(scene)
    assert after_redo == after_add


def test_undo_redo_delete_node(editor):
    scene = editor.scene
    # Ensure at least one node exists; if not, add one
    nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]
    if not nodes:
        ev_add = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_C,
            QtCore.Qt.NoModifier,
        )
        scene.keyPressEvent(ev_add)
        nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]

    before = _count_nodes(scene)
    node = nodes[0]
    node.setSelected(True)

    ev_del = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Delete,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(ev_del)

    after_del = _count_nodes(scene)
    assert after_del == before - 1

    editor.undo()
    after_undo = _count_nodes(scene)
    assert after_undo == before

    editor.redo()
    after_redo = _count_nodes(scene)
    assert after_redo == after_del


def test_undo_redo_paste_nodes(editor):
    scene = editor.scene
    nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]
    if not nodes:
        ev_add = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_C,
            QtCore.Qt.NoModifier,
        )
        scene.keyPressEvent(ev_add)
        nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]

    # Select one node and copy
    node = nodes[0]
    node.setSelected(True)
    before = _count_nodes(scene)

    ev_copy = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_C,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_copy)

    ev_paste = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_V,
        QtCore.Qt.ControlModifier,
    )
    scene.keyPressEvent(ev_paste)

    after_paste = _count_nodes(scene)
    assert after_paste == before + 1

    editor.undo()
    after_undo = _count_nodes(scene)
    assert after_undo == before

    editor.redo()
    after_redo = _count_nodes(scene)
    assert after_redo == after_paste
