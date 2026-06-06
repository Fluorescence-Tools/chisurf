from qtpy import QtWidgets


def test_updater_source_change(qapp, qtbot):
    from chisurf.plugins.chisurf.updater import UpdaterWidget
    widget = UpdaterWidget()
    qtbot.addWidget(widget)

    radios = widget.findChildren(QtWidgets.QRadioButton)
    assert len(radios) >= 2, "Expected at least two radio buttons (Remote URL / Local Folder)"

    remote_radio = radios[0]
    local_radio = radios[1]

    local_radio.setChecked(True)
    assert local_radio.isChecked()
    assert not remote_radio.isChecked()

    remote_radio.setChecked(True)
    assert remote_radio.isChecked()
    assert not local_radio.isChecked()
