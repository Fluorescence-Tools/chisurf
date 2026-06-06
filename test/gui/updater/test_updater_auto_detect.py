from qtpy import QtWidgets


def test_updater_auto_detect(qapp, qtbot):
    from chisurf.plugins.chisurf.updater import UpdaterWidget
    widget = UpdaterWidget()
    qtbot.addWidget(widget)

    browse_buttons = widget.findChildren(QtWidgets.QPushButton)
    assert len(browse_buttons) >= 1, "Expected at least one push button"
