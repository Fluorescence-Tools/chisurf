"""The AutoForm MFDB login dialog: single-line password + collapsible advanced."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_login_dialog_password_masked_and_advanced_collapsible(qapp):
    from qtpy import QtWidgets

    from chisurf.gui.widgets.collapsible_box import CollapsibleBox
    from chisurf.plugins.core.mfdb_admin.gui.connection_dialog import ConnectionAuthDialog

    d = ConnectionAuthDialog(user="admin", host="127.0.0.1", cmd_port=8765, pub_port=8766)
    # password is a masked single-line field
    masked = [
        e for e in d.findChildren(QtWidgets.QLineEdit)
        if e.echoMode() == QtWidgets.QLineEdit.Password
    ]
    assert len(masked) == 1
    # the advanced (user + connection) block is a collapsible panel
    assert len(d.findChildren(CollapsibleBox)) == 1

    # values round-trip through the AutoForm model
    d._model.password = "pw"
    d._model.host = "10.0.0.5"
    vals = d.values()
    assert vals == {
        "user": "admin", "password": "pw",
        "host": "10.0.0.5", "cmd_port": 8765, "pub_port": 8766,
    }
