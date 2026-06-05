import sys
import time
import chisurf

import pytest
import chisurf.core.actions
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.models.model import Model
from chisurf.core.fitting.fit import Fit

class MockController:
    def __init__(self):
        self.finalize_called = 0
        self.last_value = None

    def finalize(self):
        self.finalize_called += 1

def test_fitting_parameter_update_trigger():
    """
    Test that calling FittingParameter.update() triggers controller.finalize().
    """
    p = FittingParameter(name="p1", value=1.0)
    ctrl = MockController()
    p.controller = ctrl
    
    # This is the call that should trigger the refresh
    p.update()
    
    # ASSERTION: finalize should have been called
    assert ctrl.finalize_called == 1

def test_parameter_sync_via_dispatch():
    """
    Test that changing a parameter value via dispatch triggers UI refresh.
    """
    # 1. Setup a Model and Fit
    class SimpleModel(Model):
        def update_model(self):
            pass

    p = FittingParameter(name="p1", value=1.0)
    # Important: p MUST be an attribute of pg for find_parameters to find it
    pg = FittingParameterGroup(name="group1")
    pg.p1 = p
    
    fit = Fit(name="fit1")
    model = SimpleModel(fit=fit)
    model.pg = pg
    fit.model = model
    
    # Discovery
    model.find_parameters()
    print(f"DEBUG: parameters_all: {model.parameters_all}")
    
    # Register the fit in chisurf.fits so the action can find it
    chisurf.fits = [fit]
    
    # 2. Attach a mock controller to the parameter
    ctrl = MockController()
    p.controller = ctrl
    
    # 3. Dispatch the action (this is what the GUI does)
    # The action handler calls fit.set_parameter_value(p.name, value)
    chisurf.core.actions.dispatch(
        name="parameter.value",
        payload={
            "parameter_name": "p1",
            "value": 5.0,
            "fit_index": 0
        }
    )
    
    # 4. Assertions
    # a) Value should be updated in the parameter object
    print(f"DEBUG: p.value after dispatch: {p.value}")
    assert p.value == 5.0
    
    # b) UI refresh should have been triggered
    assert ctrl.finalize_called >= 1

def test_rapid_parameter_updates():
    """
    Test that rapid parameter updates (simulating a slider drag) 
    eventually synchronize to the final value.
    """
    class SimpleModel(Model):
        def update_model(self):
            pass

    p = FittingParameter(name="tau", value=1.0)
    pg = FittingParameterGroup(name="group1", parameters=[p])
    fit = Fit(name="fit2")
    model = SimpleModel(fit=fit)
    model.parameter_groups = [pg]
    fit.model = model
    model.find_parameters()
    chisurf.fits = [fit]
    
    # Fire multiple updates rapidly
    values = [1.1, 1.2, 1.3, 1.4, 1.5]
    for v in values:
        chisurf.core.actions.dispatch(
            name="parameter.value",
            payload={"parameter_name": "tau", "value": v, "fit_index": 0}
        )
    
    # Wait for debounce (200ms) plus a buffer
    time.sleep(0.4)
    
    # The final value should be 1.5
    print(f"DEBUG: Final p.value after rapid updates: {p.value}")
    assert p.value == 1.5

if __name__ == "__main__":
    print("DEBUG: STARTING TEST")
    try:
        test_fitting_parameter_update_trigger()
        print("test_fitting_parameter_update_trigger PASSED")
        test_parameter_sync_via_dispatch()
        print("test_parameter_sync_via_dispatch PASSED")
        test_rapid_parameter_updates()
        print("test_rapid_parameter_updates PASSED")
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
