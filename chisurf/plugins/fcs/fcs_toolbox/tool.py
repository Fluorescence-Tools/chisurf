"""FCS Tools — a meta tool that hosts several FCS tools behind a left icon rail.

Built on the reusable :class:`chisurf.gui.widgets.meta_tool.MetaToolWindow`
(the rail + lazy-loaded stack + separators), so it shares the exact construction
of the other category tool windows.
"""

from __future__ import annotations

from qtpy import QtWidgets

from chisurf.gui.widgets.meta_tool import SEPARATOR, MetaToolWindow


def _make_detector_def() -> QtWidgets.QWidget:
    from chisurf.plugins.core.setup_channel_definition.gui.tool import (
        SetupChannelDefinitionWidget,
    )
    return SetupChannelDefinitionWidget()


def _make_correlation_def() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_channel_preset.gui.tool import FCSChannelWidget
    return FCSChannelWidget()


def _make_2dflcs() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.flc_2d import TwoDFCSPlugin
    return TwoDFCSPlugin()


def _make_burst_fcs() -> QtWidgets.QWidget:
    from chisurf.plugins.burst.burst_fcs_correlator.gui.tool import BurstFcsTool
    return BurstFcsTool()


def _make_diffusion_calc() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    return ConfocalCalcWidget()


def _make_filter_calc() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_filter_calculator.gui_parts.main_window import (
        FcsFilterCalculatorWidget,
    )
    return FcsFilterCalculatorWidget()


def _make_merger() -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_merger.wizard import ChisurfWizard
    return ChisurfWizard()


# (emoji, label, factory); SEPARATOR rows render a horizontal divider.
TOOLS = [
    ("🎛️", "Detector\nDef", _make_detector_def),
    ("🎚️", "Correlation\nCh Def", _make_correlation_def),
    ("🟦", "2D-FLCS", _make_2dflcs),
    ("🔬", "Burst-wise\nFCS", _make_burst_fcs),
    SEPARATOR,
    ("🧮", "Diffusion\nCalc", _make_diffusion_calc),
    ("🧪", "Filter\nCalc", _make_filter_calc),
    ("🔗", "FCS\nMerger", _make_merger),
]


class FcsToolboxTool(MetaToolWindow):
    """FCS Tools window (left icon rail + the selected tool on the right)."""

    def __init__(self, parent=None):
        super().__init__("🧰 FCS Tools", TOOLS, parent=parent)
