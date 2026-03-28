import sys
import os
import numpy as np

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_empty_vector():
    print("\nTesting empty vector handling...")
    
    # Create a port
    p = cn.Port()
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    
    # Assign an empty vector
    p.value = np.array([])
    print(f"After assigning empty vector, value: {p.value}")
    print(f"After assigning empty vector, value_type: {p.get_value_type()}")
    
    # Verify that the port's value_type is set to 2 (vector int)
    assert p.get_value_type() == 2, f"Expected value_type 2 (vector int), got {p.get_value_type()}"
    
    # Verify that the port's value is an empty vector
    assert isinstance(p.value, (tuple, list, np.ndarray)), f"Expected array-like value, got {type(p.value)}"
    assert len(p.value) == 0, f"Expected empty vector, got vector of length {len(p.value)}"
    
    print("Empty vector test passed!")

def test_empty_float_vector():
    print("\nTesting empty float vector handling...")
    
    # Create a port with float type
    p = cn.Port(value=3.14)
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    
    # Assign an empty vector with float dtype
    p.value = np.array([], dtype=np.float64)
    print(f"After assigning empty float vector, value: {p.value}")
    print(f"After assigning empty float vector, value_type: {p.get_value_type()}")
    
    # Verify that the port's value_type is set to 3 (vector float)
    assert p.get_value_type() == 3, f"Expected value_type 3 (vector float), got {p.get_value_type()}"
    
    # Verify that the port's value is an empty vector
    assert isinstance(p.value, (tuple, list, np.ndarray)), f"Expected array-like value, got {type(p.value)}"
    assert len(p.value) == 0, f"Expected empty vector, got vector of length {len(p.value)}"
    
    print("Empty float vector test passed!")

if __name__ == "__main__":
    test_empty_vector()
    test_empty_float_vector()
    print("\nAll tests completed!")