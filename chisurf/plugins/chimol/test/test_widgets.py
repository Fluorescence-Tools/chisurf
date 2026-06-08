import pytest
from qtpy import QtWidgets


def test_molview(qtbot):
    pytest.importorskip("OpenGL")
    try:
        from chisurf.plugins.chimol import MolView
    except ImportError:
        pytest.skip("MolView import failed (missing dependencies)")
    widget = MolView()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)


def test_molview_plugin_window(qtbot):
    pytest.importorskip("OpenGL")
    try:
        from chisurf.plugins.chimol import MolViewPluginWindow
    except ImportError:
        pytest.skip("MolViewPluginWindow import failed (missing dependencies)")
    widget = MolViewPluginWindow()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
