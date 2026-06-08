from qtpy import QtWidgets


def test_updater_default_remote(qapp, qtbot):
    from chisurf.plugins.core.updater import UpdaterWidget
    widget = UpdaterWidget()
    qtbot.addWidget(widget)

    radios = widget.findChildren(QtWidgets.QRadioButton)
    assert len(radios) >= 1, "Expected at least one radio button"

    remote_radio = radios[0]
    assert remote_radio.isChecked()
