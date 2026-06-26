"""Tests for the FCS Toolbox meta tool."""


def test_toolbox_builds_with_rail_and_lazy_tools(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_toolbox.tool import TOOLS, FcsToolboxTool

    w = FcsToolboxTool()
    qtbot.addWidget(w)

    # one rail button per tool
    assert len(w._group.buttons()) == len(TOOLS) == 5
    assert {b.text().split("\n")[0] for b in w._group.buttons()} == {"🟦", "🔬", "🧮", "🧪", "🔗"}

    # the stack has one page per tool; the first is instantiated, the rest lazy
    assert w._stack.count() == len(TOOLS)
    assert w._instances[0] is not None
    assert all(inst is None for inst in w._instances[1:])

    # selecting another tool instantiates it on demand
    w._select_tool(2)  # Diffusion Calc
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    assert isinstance(w._instances[2], ConfocalCalcWidget)
    assert w._stack.currentIndex() == 2


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
