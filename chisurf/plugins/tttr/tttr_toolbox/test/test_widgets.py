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

    # panels load lazily: only the default (first) panel is instantiated initially
    assert all(p.get("instance") is None for p in w.panels[1:])


def test_included_plugins_are_menu_hidden():
    import importlib

    for mod in (
        "chisurf.plugins.tttr.ptu_alex_creator",
        "chisurf.plugins.tttr.tttr_microtime_shifter",
        "chisurf.plugins.tttr.ptu_header_edit",
        "chisurf.plugins.tttr.tttr_splitter",
    ):
        m = importlib.import_module(mod)
        assert getattr(m, "menu_hidden", False) is True, mod
