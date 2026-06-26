"""Tests for the FCS Tools window (shared NavigationPanelTool shell)."""


def test_fcs_tools_uses_shared_navigation_shell(qapp, qtbot):
    from chisurf.gui.widgets.navigation import NavigationPanelTool
    from chisurf.plugins.fcs.fcs_toolbox.tool import FCS_PANELS, FcsToolboxTool

    w = FcsToolboxTool()
    qtbot.addWidget(w)

    # same base / look as Burst Analysis, Decay Analysis, Imaging Tools
    assert isinstance(w, NavigationPanelTool)
    assert w.windowTitle() == "FCS Tools"

    # one navigation entry per FCS panel
    assert len(FCS_PANELS) == 7
    assert w.nav_list.count() == len(FCS_PANELS)
    names = [w.nav_list.item(i).text() for i in range(w.nav_list.count())]
    assert any("Diffusion Calc" in n for n in names)

    # panels load lazily: only the default (first) panel is instantiated initially
    assert all(p.get("instance") is None for p in w.panels[1:])
    w.nav_list.setCurrentRow(4)  # Diffusion Calc
    assert w.panels[4].get("instance") is not None


def test_included_plugins_are_menu_hidden():
    import importlib

    for mod in (
        "chisurf.plugins.fcs.flc_2d",
        "chisurf.plugins.fcs.fcs_calculator",
        "chisurf.plugins.fcs.fcs_filter_calculator",
        "chisurf.plugins.fcs.fcs_merger",
        "chisurf.plugins.burst.burst_fcs_correlator",
    ):
        m = importlib.import_module(mod)
        assert getattr(m, "menu_hidden", False) is True, mod
