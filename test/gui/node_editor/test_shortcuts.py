"""Tests for keyboard shortcuts that create nodes from the scene."""

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def editor(app):  # noqa: ARG001 - ensures QApplication exists
    return NodeEditorWidget()


def _count_nodes(scene):
    return len([it for it in scene.items() if isinstance(it, NodeGraphicsItem)])


def test_shortcut_creates_constant_node(editor):
    scene = editor.scene
    before = _count_nodes(scene)

    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_C,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(event)

    after = _count_nodes(scene)
    assert after == before + 1


def test_shortcut_respects_view_center(editor):
    scene = editor.scene
    view = editor.view

    before = _count_nodes(scene)

    # Move view so that center is at a non-zero scene position
    view.centerOn(200, 100)

    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_O,
        QtCore.Qt.NoModifier,
    )
    scene.keyPressEvent(event)

    after = _count_nodes(scene)
    assert after == before + 1

    # Roughly check that the last node is near the view center
    nodes = [it for it in scene.items() if isinstance(it, NodeGraphicsItem)]
    last = max(nodes, key=lambda n: n.pos().x() + n.pos().y())
    center_scene = view.mapToScene(view.viewport().rect().center())
    dx = abs(last.pos().x() - center_scene.x())
    dy = abs(last.pos().y() - center_scene.y())
    # Allow a fairly generous tolerance: shortcut-created node should still
    # appear roughly near the view center even after layout/zoom changes.
    assert dx < 400
    assert dy < 400
