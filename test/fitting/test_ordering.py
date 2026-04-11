
import pytest
import numpy as np
from chisurf.fitting.parameter import FittingParameter, FittingParameterGroup

def test_find_parameters_ordering_stability():
    """
    Ensures that FittingParameterGroup.find_parameters() preserves the discovery 
    order of parameters and nested groups, and does not shuffle them on subsequent 
    calls (as list(set(...)) would).
    """
    # Create some parameters in a specific discovery order (by assigning to __dict__)
    # Python 3.7+ preserves dict insertion order by default.
    p_names = [f"p{i}" for i in range(10)]
    params = [FittingParameter(name=name, value=float(i)) for i, name in enumerate(p_names)]
    
    group = FittingParameterGroup()
    # Define parameters on the group in a fixed order
    for p in params:
        setattr(group, p.name, p)

    # First call to find_parameters should discover them in insertion order
    group.find_parameters()
    order1 = [p.name for p in group.parameters_all]
    assert order1 == p_names

    # Subsequent calls should not change the order
    for _ in range(10):
        group.find_parameters()
        assert [p.name for p in group.parameters_all] == order1, "Parameter order changed on subsequent find_parameters call"

def test_aggregated_parameters_ordering_stability():
    """
    Ensures that group.aggregated_parameters preserves discovery order.
    """
    subgroups = [FittingParameterGroup(name=f"sub{i}") for i in range(5)]
    group = FittingParameterGroup()
    
    for sg in subgroups:
        setattr(group, sg.name, sg)
        
    group.find_parameters()
    # aggregated_parameters uses find_objects which used to use set()
    order1 = [sg.name for sg in group.aggregated_parameters]
    
    # Since find_objects now preserves order, it should match discovery order
    expected = [f"sub{i}" for i in range(5)]
    assert order1 == expected
    
    # Check stability
    for _ in range(10):
        order_n = [sg.name for sg in group.aggregated_parameters]
        assert order_n == order1

def test_find_objects_order_preservation():
    """Directly test chisurf.base.find_objects for order preservation."""
    from chisurf.base import find_objects
    
    class Mock:
        pass
        
    objs = [Mock() for _ in range(20)]
    # Mix them up in a list with duplicates
    mixed = []
    for i in range(len(objs)):
        mixed.append(objs[i])
        if i % 3 == 0:
            mixed.append(objs[i]) # add duplicate
            
    # find_objects should return unique objects in discovery order
    discovered = find_objects(mixed, Mock, remove_doublets=True)
    assert len(discovered) == len(objs)
    assert discovered == objs # Exact sequence match
