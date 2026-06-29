"""Integrated fluorescence lifetime analysis GUI."""

from __future__ import annotations

from qtpy import QtWidgets

from chisurf.gui.widgets.navigation import NavigationPanelTool


def _irf_estimator(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the IRF estimation panel."""
    from chisurf.plugins.fluorescence_decay.irf_estimator.gui.tool import (
        IRFEstimatorTool,
    )

    return IRFEstimatorTool(parent=parent)


def _maxent_mem(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the MaxEnt MEM panel."""
    from chisurf.plugins.fluorescence_decay.maxent_decay.gui.gui import (
        MaxentDecayWidget,
    )

    widget = MaxentDecayWidget()
    widget.setParent(parent)
    return widget


def _lazy_lifetime(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the Lazy Lifetime Analysis panel."""
    from chisurf.plugins.fluorescence_decay.lltf.lltf_gui import LLTFGUIWizard

    return LLTFGUIWizard(parent=parent)


def _microtime_histogram(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the microtime histogram panel."""
    from chisurf.plugins.tttr.microtime_histogram.wizard import MicrotimeHistogram

    return MicrotimeHistogram(parent=parent)


def _jordi_g_factor(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the Jordi G-factor panel."""
    from chisurf.plugins.jordi_g_factor.gui.tool import JordiGFactorCalculator

    widget = JordiGFactorCalculator()
    widget.setParent(parent)
    return widget


LIFETIME_PANELS = [
    {
        "name": "1. IRF Estimation",
        "icon": "🌊",
        "description": "Estimate instrument response functions from fluorescence decays.",
        "factory": _irf_estimator,
        "role": "irf",
    },
    {
        "name": "2. MaxEnt MEM",
        "icon": "📈",
        "description": "Run maximum entropy lifetime and FRET-distance analysis.",
        "factory": _maxent_mem,
        "role": "maxent",
    },
    {
        "name": "3. Lazy Lifetime Analysis",
        "icon": "⏱️",
        "description": "Analyze TCSPC decays with the LLTF workflow.",
        "factory": _lazy_lifetime,
        "role": "lazy_lifetime",
    },
    {
        "name": "4. Histogram-Microtime",
        "icon": "📊",
        "description": "Build TTTR microtime histograms.",
        "factory": _microtime_histogram,
        "role": "microtime_histogram",
    },
    {
        "name": "5. Jordi G-Factor",
        "icon": "⚖️",
        "description": "Calculate detector G-factors from Jordi decays.",
        "factory": _jordi_g_factor,
        "role": "jordi_g_factor",
    },
]


class LifetimeAnalysisTool(NavigationPanelTool):
    """Integrated fluorescence lifetime analysis tool."""

    def __init__(self, parent=None):
        """Create the integrated lifetime analysis tool."""
        super().__init__(
            title="Decay Analysis",
            panels=LIFETIME_PANELS,
            parent=parent,
            minimum_size=(950, 620),
            initial_size=(1180, 760),
            navigation_width=270,
            navigation_min_width=250,
        )
