"""Tests for node creation UX and menu grouping based on the registry."""

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


def test_grouping_uses_registry_categories(editor):
    """Scene._group_node_types_for_menu should reflect registry categories."""
    scene = editor.scene

    groups = scene._group_node_types_for_menu()
    # Expect at least the built-in categories
    all_ids = {t for ids in groups.values() for t in ids}
    assert {"constant", "binary_op", "output", "controls"}.issubset(all_ids)
    # There should be at least two categories when registry is populated
    assert len(groups) >= 2


def test_grouping_respects_available_node_types(editor):
    """available_node_types should restrict what appears in the menu grouping."""
    editor.available_node_types = ["constant", "output"]
    scene = editor.scene

    groups = scene._group_node_types_for_menu()
    all_ids = sorted({t for ids in groups.values() for t in ids})
    assert all_ids == ["constant", "output"]
