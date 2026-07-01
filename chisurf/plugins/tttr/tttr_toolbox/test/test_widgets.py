"""Tests for the TTTR Tools window (shared NavigationPanelTool shell)."""


def test_tttr_tools_uses_shared_navigation_shell(qapp, qtbot):
    from chisurf.gui.widgets.navigation import NavigationPanelTool
    from chisurf.plugins.tttr.tttr_toolbox.gui.tool import TTTR_PANELS, TttrToolboxTool

    w = TttrToolboxTool()
    qtbot.addWidget(w)

    # same base / look as FCS Tools, Structure Tools, Burst Analysis, ...
    assert isinstance(w, NavigationPanelTool)
    assert "TTTR Tools" in w.windowTitle()

    # one navigation entry per panel (incl. the separator)
    assert w.nav_list.count() == len(TTTR_PANELS)
    names = [w.nav_list.item(i).text() for i in range(w.nav_list.count())]
    assert any("ALEX Creator" in n for n in names)
    assert any("Split / Convert" in n for n in names)
    assert any("Count Rate Analysis" in n for n in names)

    # panels load lazily: only the default (first) panel is instantiated initially
    assert all(p.get("instance") is None for p in w.panels[1:])


def test_panels_are_data_driven_from_json():
    """The panel list is built from panels.json (entrypoint strings), not Python."""
    import json
    import pathlib

    from chisurf.plugins.tttr.tttr_toolbox.gui import tool as tool_mod

    spec = json.loads((pathlib.Path(tool_mod.__file__).with_name("panels.json")).read_text())
    tool_panels = [p for p in spec["panels"] if not p.get("separator")]
    # every non-separator entry declares a resolvable "module:Class" entrypoint
    for p in tool_panels:
        module_name, _, attr = p["entrypoint"].partition(":")
        assert module_name and attr, p
    # and each becomes a panel with a generated factory
    built = [p for p in tool_mod.TTTR_PANELS if not p.get("separator")]
    assert len(built) == len(tool_panels)
    assert all(callable(p["factory"]) for p in built)


def test_included_plugins_are_menu_hidden():
    import importlib

    for mod in (
        "chisurf.plugins.tttr.ptu_alex_creator",
        "chisurf.plugins.tttr.tttr_microtime_shifter",
        "chisurf.plugins.tttr.ptu_header_edit",
        "chisurf.plugins.tttr.tttr_splitter",
        "chisurf.plugins.tttr.tttr_count_rate_analysis",
    ):
        m = importlib.import_module(mod)
        assert getattr(m, "menu_hidden", False) is True, mod
