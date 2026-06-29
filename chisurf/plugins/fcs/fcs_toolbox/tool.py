"""FCS Tools — a unified FCS toolbox built on the shared navigation shell.

Uses :class:`chisurf.gui.widgets.navigation.NavigationPanelTool` — the same base
the Burst Analysis, Decay Analysis and Imaging Tools windows use — so FCS Tools
shares their look and codebase instead of a bespoke layout.
"""

from __future__ import annotations

import pathlib

from qtpy import QtWidgets

from chisurf.core.plugin.manifest import load_manifest
from chisurf.gui.widgets.navigation import NavigationPanelTool

#: chisurf/plugins directory (this file is plugins/fcs/fcs_toolbox/tool.py).
_PLUGINS_DIR = pathlib.Path(__file__).resolve().parents[2]


def _apply_manifest_flags(panels: list[dict]) -> list[dict]:
    """Enrich panels with the ``experimental`` flag from each tool's manifest.json.

    A panel may declare ``"manifest": "<rel path under chisurf/plugins>"``; the manifest's
    ``experimental`` / ``experimental_message`` then drive the navigation marker and banner,
    so the flag lives in one place (the manifest) rather than being duplicated here.
    """
    for panel in panels:
        rel = panel.get("manifest")
        if not rel:
            continue
        manifest = load_manifest(_PLUGINS_DIR / rel / "manifest.json")
        if manifest is not None and manifest.experimental:
            panel["experimental"] = True
            if manifest.experimental_message:
                panel["experimental_message"] = manifest.experimental_message
    return panels


def _make_detector_def(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.core.setup_channel_definition.gui.tool import (
        SetupChannelDefinitionWidget,
    )

    w = SetupChannelDefinitionWidget()
    w.setParent(parent)
    return w


def _make_correlation_def(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_channel_preset.gui.tool import FCSChannelWidget

    w = FCSChannelWidget()
    w.setParent(parent)
    return w


def _make_2dflcs(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.flc_2d import TwoDFCSPlugin

    w = TwoDFCSPlugin()
    w.setParent(parent)
    return w


def _make_burst_fcs(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.burst.burst_fcs_correlator.gui.tool import BurstFcsTool

    w = BurstFcsTool()
    w.setParent(parent)
    return w


def _make_diffusion_calc(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget

    w = ConfocalCalcWidget()
    w.setParent(parent)
    return w


def _make_filter_calc(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_filter_calculator.gui_parts.main_window import (
        FcsFilterCalculatorWidget,
    )

    w = FcsFilterCalculatorWidget()
    w.setParent(parent)
    return w


def _make_merger(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.plugins.fcs.fcs_merger.wizard import ChisurfWizard

    w = ChisurfWizard()
    w.setParent(parent)
    return w


FCS_PANELS = [
    {
        "name": "Detector Def",
        "icon": "🎛️",
        "factory": _make_detector_def,
        "role": "detector_def",
        "description": "Define detector setups (routing channels, PIE windows).",
    },
    {
        "name": "Correlation Ch Def",
        "icon": "🎚️",
        "factory": _make_correlation_def,
        "role": "correlation_def",
        "description": "Define FCS correlation channel presets.",
    },
    {
        "name": "2D-FLCS",
        "icon": "🟦",
        "factory": _make_2dflcs,
        "role": "flc_2d",
        "manifest": "fcs/flc_2d",
        "description": "Two-dimensional fluorescence lifetime correlation spectroscopy.",
    },
    {
        "name": "Burst-wise FCS",
        "icon": "🔬",
        "factory": _make_burst_fcs,
        "role": "burst_fcs",
        "description": "Per-burst fluorescence correlation.",
    },
    {
        "name": "Diffusion Calc",
        "icon": "🧮",
        "factory": _make_diffusion_calc,
        "role": "diffusion_calc",
        "description": "Confocal diffusion / volume calculator.",
    },
    {
        "name": "Filter Calc",
        "icon": "🧪",
        "factory": _make_filter_calc,
        "role": "filter_calc",
        "manifest": "fcs/fcs_filter_calculator",
        "description": "Filtered-FCS lifetime filter calculator.",
    },
    {
        "name": "FCS Merger",
        "icon": "🔗",
        "factory": _make_merger,
        "role": "merger",
        "description": "Merge / average FCS correlation curves.",
    },
]

# Pull the experimental flag from each tool's manifest (single source of truth).
_apply_manifest_flags(FCS_PANELS)


class FcsToolboxTool(NavigationPanelTool):
    """FCS Tools window (left navigation panel + the selected tool on the right)."""

    def __init__(self, parent=None):
        super().__init__(
            title="FCS Tools",
            panels=FCS_PANELS,
            parent=parent,
            minimum_size=(950, 600),
            initial_size=(1180, 740),
            navigation_width=210,
        )
