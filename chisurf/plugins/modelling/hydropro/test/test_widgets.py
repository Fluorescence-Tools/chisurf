import pytest
from qtpy import QtWidgets


def test_hydropro_tool_creation(qapp, qtbot):
    from chisurf.plugins.modelling.hydropro.gui.tool import HydroGui, HydroProTool

    widget = HydroProTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "HYDRO" in widget.windowTitle()
    # AutoForm + controls are present.
    assert hasattr(widget, "_form")
    assert hasattr(widget, "table")
    assert HydroGui is HydroProTool  # backwards-compat alias


def test_view_spec_loads(qapp, qtbot):
    from chisurf.plugins.modelling.hydropro.gui.tool import _HydroModel

    view = _HydroModel().view_spec()
    titles = [s.title for s in view.sections]
    assert "Executable & input" in titles
    assert "Primary model" in titles


def test_model_to_settings_roundtrip():
    from chisurf.plugins.modelling.hydropro.core import HydroProSettings
    from chisurf.plugins.modelling.hydropro.gui.tool import _HydroModel

    m = _HydroModel()
    m.indmode = "4"
    m.idif = False
    s = m.to_settings()
    assert s.indmode == 4
    assert s.idif == 0
    # load back
    m.load_settings(HydroProSettings(indmode=2, idif=1))
    assert m.indmode == "2"
    assert m.idif is True


def test_settings_validation():
    from chisurf.plugins.modelling.hydropro.core import HydroProSettings

    HydroProSettings().validate()  # defaults are valid
    with pytest.raises(ValueError):
        HydroProSettings(indmode=3).validate()
    with pytest.raises(ValueError):
        HydroProSettings(nq=5, qmax=0.0).validate()


def test_parse_diffusion_coefficient(tmp_path):
    from chisurf.plugins.modelling.hydropro.core import parse_diffusion_coefficient

    res = tmp_path / "case-res.txt"
    res.write_text("Translational diffusion coefficient:  3.760E-07 cm^2/s\n")
    assert parse_diffusion_coefficient(res) == pytest.approx(3.760e-07)
    assert parse_diffusion_coefficient(tmp_path / "missing.txt") is None
