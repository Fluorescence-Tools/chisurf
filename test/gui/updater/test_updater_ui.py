from qtpy import QtWidgets


def test_updater_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.updater import UpdaterWidget
    widget = UpdaterWidget()
    qtbot.addWidget(widget)
    assert "Update" in widget.windowTitle() or "Updater" in widget.windowTitle()
