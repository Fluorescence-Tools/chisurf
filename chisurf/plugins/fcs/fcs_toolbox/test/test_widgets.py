"""Tests for the FCS Toolbox meta tool."""


def test_toolbox_builds_with_rail_and_lazy_tools(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_toolbox.tool import TOOLS, FcsToolboxTool

    w = FcsToolboxTool()
    qtbot.addWidget(w)

    # one rail button per tool
    assert len(w._group.buttons()) == len(TOOLS) == 3
    assert {b.text().split("\n")[0] for b in w._group.buttons()} == {"🟦", "🔬", "🧮"}

    # the stack has one page per tool; the first is instantiated, the rest lazy
    assert w._stack.count() == len(TOOLS)
    assert w._instances[0] is not None
    assert w._instances[1] is None and w._instances[2] is None

    # selecting another tool instantiates it on demand
    w._select_tool(2)  # Diffusion Calc
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    assert isinstance(w._instances[2], ConfocalCalcWidget)
    assert w._stack.currentIndex() == 2
