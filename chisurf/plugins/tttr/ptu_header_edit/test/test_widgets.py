import pytest
from qtpy import QtWidgets


def test_tags_editor_creation(qapp, qtbot):
    pytest.importorskip("tttrlib")
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.tttr.ptu_header_edit.gui.tool import TagsEditor

    widget = TagsEditor()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert isinstance(widget.auto_form, AutoForm)
    assert hasattr(widget.model, "tags")
