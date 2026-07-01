"""Tests for the AutoForm-based ALEX Creator tool."""

from __future__ import annotations

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PTU = _REPO_ROOT / "test" / "data" / "clsm" / "Leica_SP5.ptu"


def test_view_model_defaults_and_options():
    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    m = AlexViewModel()
    assert m.input_format == "Auto"
    assert m.output_format == "PTU"
    assert m.alex_period == 8000
    assert m.input_format_options()[0] == "Auto"
    assert "PTU" in m.output_format_options()
    assert m.can_save() is not None  # nothing loaded yet


def test_view_spec_loads():
    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    assert AlexViewModel().view_spec() is not None


def test_tool_builds_with_autoform(qapp, qtbot):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.registry import get_section_factory
    from chisurf.plugins.tttr.ptu_alex_creator.gui.tool import AlexPTUCreator

    w = AlexPTUCreator()
    qtbot.addWidget(w)
    assert isinstance(w.auto_form, AutoForm)
    assert get_section_factory("alex_actions") is not None


@pytest.mark.skipif(not _PTU.exists(), reason="sample PTU not available")
def test_alex_histogram_and_save(tmp_path):
    import tttrlib

    from chisurf.plugins.tttr.ptu_alex_creator.gui.view_model import AlexViewModel

    m = AlexViewModel()
    m.load(str(_PTU))
    assert m.has_data

    m.alex_period = 4000
    series = m.histogram_series()
    assert series and len(series[0]["y"]) == 4000

    out = tmp_path / "alex_out.ptu"
    m.output_format = "PTU"
    m.save(str(out))
    assert out.exists()
    assert len(tttrlib.TTTR(str(out))) > 0
