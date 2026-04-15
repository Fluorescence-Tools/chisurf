import pytest
import numpy as np
import chinet
from qtpy import QtWidgets, QtCore
import chisurf.data
import chisurf.fitting.fit
from chisurf.fitting.parameter import FittingParameter
from chisurf.models.model import ModelCurve
from chisurf.gui.widgets.fitting.parameter_widgets import FittingParameterWidget

class SimpleModel(ModelCurve):
    name = "SimpleModel"
    def __init__(self, fit):
        super().__init__(fit)
        self.p1 = FittingParameter(name="p1", value=1.0)
        self.find_parameters()

    def update_model(self, **kwargs):
        self.y = np.ones_like(self.x) * self.p1.value

@pytest.fixture
def mock_fit():
    data = chisurf.data.DataCurve(x=np.arange(10), y=np.arange(10))
    fit = chisurf.fitting.fit.Fit(model_class=SimpleModel, data=data)
    import chisurf
    chisurf.fits = [fit]
    return fit

def test_parameter_debounce_data_loss(qtbot, mock_fit):
    """
    Demonstrate that quick successive updates result in data loss due to 
    leading-edge debouncing in ActionDispatcher.
    """
    param = mock_fit.model.p1
    widget = FittingParameterWidget(param)
    qtbot.addWidget(widget)
    
    # Ensure initial value is 1.0
    assert param.value == 1.0
    
    # 1. First update: 5.0
    widget.widget_value.setValue(5.0)
    widget.widget_value.editingFinished.emit()
    
    # This should succeed immediately (leading edge)
    assert param.value == 5.0
    
    # 2. Second update: 6.0 IMMEDIATELY
    widget.widget_value.setValue(6.0)
    widget.widget_value.editingFinished.emit()
    
    # This will likely be suppressed because it's within 200ms of the first
    # and the ActionDispatcher does not queue a trailing-edge update.
    print(f"Param value after first update: {param.value}")
    
    # Wait for more than the debounce period (200ms)
    qtbot.wait(300)
    
    print(f"Param value after 300ms wait: {param.value}")
    
    # If Bug 2 exists, param.value will still be 5.0, NOT 6.0
    assert param.value == 5.0, "BUG: The 6.0 update was NOT lost, so either the bug is fixed or we didn't reproduce it."
    assert float(param._port.value) == 5.0

if __name__ == "__main__":
    pytest.main([__file__])
