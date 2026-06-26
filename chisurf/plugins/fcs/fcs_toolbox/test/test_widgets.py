"""Tests for the FCS Toolbox meta tool."""


def test_toolbox_builds_with_rail_and_lazy_tools(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_toolbox.tool import TOOLS, FcsToolboxTool

    w = FcsToolboxTool()
    qtbot.addWidget(w)

    # one rail button per real tool (SEPARATOR rows have no factory/button)
    n_tools = sum(1 for *_, factory in TOOLS if factory is not None)
    assert len(w._group.buttons()) == n_tools == 7
    assert len(w._factories) == n_tools
    # there is exactly one separator in the spec
    assert sum(1 for *_, factory in TOOLS if factory is None) == 1

    # the stack has one page per tool; the first is instantiated, the rest lazy
    assert w._stack.count() == n_tools
    assert w._instances[0] is not None
    assert all(inst is None for inst in w._instances[1:])

    # selecting the Diffusion Calc (tool index 4 after the two setup tools,
    # 2D-FLCS and Burst-wise FCS) instantiates it on demand
    w._select_tool(4)
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    assert isinstance(w._instances[4], ConfocalCalcWidget)
    assert w._stack.currentIndex() == 4


def test_included_plugins_are_menu_hidden():
    import importlib

    for mod in (
        "chisurf.plugins.fcs.fcs_2d",
        "chisurf.plugins.fcs.fcs_calculator",
        "chisurf.plugins.fcs.fcs_filter_calculator",
        "chisurf.plugins.fcs.fcs_merger",
        "chisurf.plugins.burst.burst_fcs_correlator",
    ):
        m = importlib.import_module(mod)
        assert getattr(m, "menu_hidden", False) is True, mod
