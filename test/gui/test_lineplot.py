import os
import sys
import pytest
import numpy as np
from pathlib import Path
from chisurf.gui.plots.lineplot.lineplot import LinePlot

def _lineplot_source() -> str:
    path = Path(__file__).resolve().parents[2] / "chisurf" / "plots" / "lineplot" / "lineplot.py"
    return path.read_text(encoding="utf-8")

def test_group_display_methods_exist():
    src = _lineplot_source()
    assert "def _plot_group_curves(self" in src
    assert "def _plot_single_fit_curves(self" in src
    assert "def _plot_active_fit_only(self" in src

def test_group_display_alpha_and_setalpha_contract():
    src = _lineplot_source()
    assert "line.setAlpha(int(alpha * 255), auto=False)" in src
    assert "alpha = 1.0" in src
    assert "alpha = 0.4" in src

def test_group_display_uses_selected_fit_when_available():
    src = _lineplot_source()
    assert "selected_fit" in src
    assert "hasattr(self.fit, 'grouped_fits')" in src or "hasattr(fit, 'grouped_fits')" in src

def test_lineplot_reference_no_attribute(qtbot):
    """
    Verify that LinePlot handles models without a 'reference' attribute 
    gracefully when reference_curve=True is set.
    """
    # Create simple test data
    x = np.linspace(0, 10, 100)
    y = np.exp(-x)
    
    # Mock data class
    class MockData:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Mock fit class
    class MockFit:
        def __init__(self, x, y):
            self.data = MockData(x, y)
            self.xmin = 0
            self.xmax = len(x)
            
        def get_curves(self):
            return {'data': MockData(self.data.x, self.data.y)}
    
    # Mock model without reference attribute
    class MockModelWithoutReference:
        def __init__(self, fit):
            self.fit = fit
            
    fit = MockFit(x, y)
    fit.model = MockModelWithoutReference(fit)
    
    # Create a LinePlot with reference_curve=True
    # This usually creates internal widgets, so we use qtbot to track it
    plot = LinePlot(fit, reference_curve=True)
    qtbot.addWidget(plot)
    
    # Update the plot - this should not raise an error
    plot.update()
    
    # Check if the reference checkbox is disabled/available
    assert plot.plot_controller.checkBox_5.isEnabled() is False
    assert plot.plot_controller.use_reference is False