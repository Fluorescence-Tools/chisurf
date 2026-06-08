import pytest
from qtpy import QtWidgets


def test_tags_editor_creation(qapp, qtbot):
    pytest.importorskip("tttrlib")
    from chisurf.plugins.tttr.ptu_header_edit.wizard import TagsEditor
    widget = TagsEditor(json_data={})
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert hasattr(widget, "table_widget")
    assert hasattr(widget, "json_display")
