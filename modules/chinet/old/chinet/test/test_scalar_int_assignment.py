import sys
import os
import numpy as np

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_scalar_int_assignment():
    print("\nTesting scalar int assignment...")
    
    # Create a port
    p = cn.Port()
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    
    # Assign an integer value
    p.value = 2
    print(f"After assigning 2, value: {p.value}")
    print(f"After assigning 2, value_type: {p.get_value_type()}")
    
    # Verify that the port's value_type is set to 0 (scalar int)
    assert p.get_value_type() == 0, f"Expected value_type 0 (scalar int), got {p.get_value_type()}"
    
    # Verify that the port's value is the integer we assigned
    assert p.value == 2, f"Expected value 2, got {p.value}"
    
    print("Scalar int assignment test passed!")

if __name__ == "__main__":
    test_scalar_int_assignment()
    print("\nAll tests completed!")