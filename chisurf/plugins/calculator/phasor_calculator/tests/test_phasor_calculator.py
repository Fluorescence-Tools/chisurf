"""Headless + offscreen-GUI tests for the phasor calculator (PRD-56)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_model_overlays_track_toggles():
    from chisurf.plugins.calculator.phasor_calculator.gui.tool import _PhasorCalcModel

    m = _PhasorCalcModel()
    m.show_grid = False
    m.show_ticks = True
    m.show_fret = False
    m.show_component = False
    names = {o.get("name") for o in m.phasor_overlays()}
    assert "lifetime ticks" in names

    m.show_fret = True
    m.show_component = True
    names = {o.get("name") for o in m.phasor_overlays()}
    assert "FRET trajectory" in names
    assert "component line" in names


def test_model_tau_parsing_is_robust():
    from chisurf.plugins.calculator.phasor_calculator.gui.tool import _PhasorCalcModel

    m = _PhasorCalcModel()
    m.taus = "1, 2 ; 4 , bad,"
    assert m._tau_list() == [1.0, 2.0, 4.0]
    m.taus = "garbage"
    assert m._tau_list() == [1.0]  # falls back to a sane default


def test_results_html_lists_reference_points():
    from chisurf.plugins.calculator.phasor_calculator.gui.tool import _PhasorCalcModel

    m = _PhasorCalcModel()
    html = m.results_html()
    assert "MHz" in html and "<table>" in html


def test_registered_in_hub():
    from chisurf.plugins.calculator.hub.core.registry import default_calculators

    ids = {e.id for e in default_calculators()}
    assert "phasor" in ids


def test_view_spec_loads():
    from chisurf.plugins.calculator.phasor_calculator.gui.tool import _PhasorCalcModel

    spec = _PhasorCalcModel().view_spec()
    assert spec is not None


def test_gui_tool_constructs_and_renders_offscreen(qtbot):
    from chisurf.plugins.calculator.phasor_calculator.gui.tool import PhasorCalculatorTool

    tool = PhasorCalculatorTool()
    qtbot.addWidget(tool)
    tool.show()
    qtbot.waitExposed(tool)
    # Toggle overlays and refresh without error.
    tool._model.show_fret = True
    tool._model.show_component = True
    tool._form.refresh_plots()
    pixmap = tool.grab()
    assert pixmap.width() > 0 and pixmap.height() > 0
