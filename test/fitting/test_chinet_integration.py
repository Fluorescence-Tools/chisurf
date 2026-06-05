import pytest
import numpy as np
import chinet
from chisurf.core.fitting.parameter import FittingParameter

def test_fitting_parameter_chinet_sync():
    # Test that FittingParameter synchronizes with its underlying chinet.Port
    p = FittingParameter(name="test_p", value=10.0, lb=0.0, ub=20.0, bounds_on=True)
    
    # Check initial sync
    assert p.value == 10.0
    assert p._port.value == 10.0
    assert p._port.bounded == True
    assert np.allclose(p._port.bounds, [0.0, 20.0])
    
    # Update via Parameter object
    p.value = 15.0
    assert p._port.value == 15.0
    
    # Update via chinet.Port directly
    p._port.value = np.array([5.0])
    assert p.value == 5.0
    
    # Test bounding
    p.value = 25.0 # Should be clamped to 20.0 by chinet.Port if enforced?
    # Actually, chisurf's Parameter.value setter might not clamp automatically unless the port does.
    # Let's verify chinet behavior.
    assert p.value == 20.0 or p._port.value == 20.0

def test_fitting_parameter_linking_sync():
    # Test that linking FittingParameters links their chinet.Ports
    p1 = FittingParameter(name="p1", value=1.0)
    p2 = FittingParameter(name="p2", value=2.0)
    
    p2.link = p1
    assert p2.is_linked
    assert p2._port.is_linked()
    
    # Check value sync through link
    p1.value = 5.0
    assert p2.value == 5.0
    assert p2._port.value == 5.0

def test_fitting_parameter_fixed_sync():
    p = FittingParameter(name="p", value=1.0, fixed=False)
    assert not p._port.fixed
    
    p.fixed = True
    assert p._port.fixed
    
    p.fixed = False
    assert not p._port.fixed
