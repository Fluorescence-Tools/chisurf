"""Tests for the integrated lifetime analysis tool."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from chisurf.core.plugin.manifest import load_manifest


def test_lifetime_panel_order() -> None:
    """Integrated lifetime tool exposes the requested ordered tools."""
    from chisurf.plugins.fluorescence_decay.lifetime_analysis.gui.tool import (
        LIFETIME_PANELS,
    )

    labels = [f"{panel.get('icon', '')} {panel['name']}".strip() for panel in LIFETIME_PANELS]
    assert labels == [
        "🌊 1. IRF Estimation",
        "📈 2. MaxEnt MEM",
        "⏱️ 3. Lazy Lifetime Analysis",
        "📊 4. Histogram-Microtime",
        "⚖️ 5. Jordi G-Factor",
    ]


def test_lifetime_panel_factories_import_expected_widgets(monkeypatch) -> None:
    """Panel factories instantiate the existing standalone widget classes."""
    from qtpy import QtWidgets

    from chisurf.plugins.fluorescence_decay.lifetime_analysis.gui import tool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Widget(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)

    modules = {
        "chisurf.plugins.fluorescence_decay.irf_estimator.gui.tool": (
            "IRFEstimatorTool",
            MainWindow,
        ),
        "chisurf.plugins.fluorescence_decay.maxent_decay.gui.gui": (
            "MaxentDecayWidget",
            MainWindow,
        ),
        "chisurf.plugins.fluorescence_decay.lltf.lltf_gui": ("LLTFGUIWizard", MainWindow),
        "chisurf.plugins.tttr.microtime_histogram.wizard": (
            "MicrotimeHistogram",
            Widget,
        ),
        "chisurf.plugins.jordi_g_factor.gui.tool": ("JordiGFactorCalculator", Widget),
    }
    for module_name, (class_name, widget_class) in modules.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, widget_class)
        monkeypatch.setitem(sys.modules, module_name, module)

    parent = QtWidgets.QWidget()
    widgets = [
        tool._irf_estimator(parent),
        tool._maxent_mem(parent),
        tool._lazy_lifetime(parent),
        tool._microtime_histogram(parent),
        tool._jordi_g_factor(parent),
    ]

    assert all(isinstance(widget, QtWidgets.QWidget) for widget in widgets)
    assert all(widget.parent() is parent for widget in widgets)
    for widget in widgets:
        widget.close()
    parent.close()
    app.processEvents()


def test_lifetime_analysis_menu_metadata() -> None:
    """Only the grouped lifetime tool remains visible in plugin menus."""
    root = Path(__file__).resolve().parents[1]
    visible = load_manifest(root / "manifest.json")
    assert visible is not None
    assert visible.display_name == "Spectroscopy:Fluorescence decay:Lifetime Analysis"
    assert visible.menu_hidden is False

    hidden_manifests = [
        root.parent / "irf_estimator" / "manifest.json",
        root.parent / "maxent_decay" / "manifest.json",
        root.parent / "lltf" / "manifest.json",
        root.parents[1] / "tttr" / "microtime_histogram" / "manifest.json",
        root.parents[1] / "jordi_g_factor" / "manifest.json",
    ]
    for manifest_path in hidden_manifests:
        manifest = load_manifest(manifest_path)
        assert manifest is not None, manifest_path
        assert manifest.menu_hidden is True, manifest_path
