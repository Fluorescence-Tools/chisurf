import pytest
from qtpy import QtWidgets


def test_hydro_gui_creation(qapp, qtbot):
    from chisurf.plugins.modelling.hydropro.hydrogui import HydroGui
    widget = HydroGui()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "HYDRO" in widget.windowTitle()
    assert hasattr(widget, "select_button")
    assert hasattr(widget, "run_button")
    assert hasattr(widget, "table")


def test_settings_dialog_creation(qapp, qtbot):
    from chisurf.plugins.modelling.hydropro.hydrogui import SettingsDialog
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "Settings" in dialog.windowTitle()


def test_ucfret_wizard_import(qapp):
    try:
        from chisurf.plugins.modelling.ucfret.wizard import UCFRETWizard
        assert UCFRETWizard is not None
    except ImportError:
        pytest.skip("ucfret package not available")
    except RuntimeError:
        pytest.skip("ucfret package not available at runtime")
