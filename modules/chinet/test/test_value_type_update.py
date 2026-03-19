import sys
import os
import numpy as np

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_int_to_float():
    print("\nTesting int to float conversion...")
    
    # Create a port with int value
    p = cn.Port(value=42)
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    
    # Assign a float value, should upcast to float
    p.value = 3.14
    print(f"After assigning float, value: {p.value}")
    print(f"After assigning float, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    assert isinstance(p.value, (float, np.floating)), f"Expected float value, got {type(p.value)}"

def test_float_to_int():
    print("\nTesting float to int conversion...")
    
    # Create a port with float value
    p = cn.Port(value=3.14)
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    
    # Assign an int value, should downcast to int
    p.value = 42
    print(f"After assigning int, value: {p.value}")
    print(f"After assigning int, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    assert isinstance(p.value, (int, np.integer)), f"Expected int value, got {type(p.value)}"

def test_vector_int_to_float():
    print("\nTesting vector int to float conversion...")
    
    # Create a port with vector int value
    p = cn.Port(value=np.array([1, 2, 3, 4, 5]))
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    
    # Assign a vector float value, should upcast to float
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    print(f"After assigning vector float, value: {p.value}")
    print(f"After assigning vector float, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"

def test_vector_float_to_int():
    print("\nTesting vector float to int conversion...")
    
    # Create a port with vector float value
    p = cn.Port(value=np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    
    # Assign a vector int value, should downcast to int
    p.value = np.array([1, 2, 3, 4, 5])
    print(f"After assigning vector int, value: {p.value}")
    print(f"After assigning vector int, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"

def test_scalar_to_vector():
    print("\nTesting scalar to vector conversion...")
    
    # Create a port with scalar int value
    p = cn.Port(value=42)
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    
    # Assign a vector int value, should change to vector int
    p.value = np.array([1, 2, 3, 4, 5])
    print(f"After assigning vector int, value: {p.value}")
    print(f"After assigning vector int, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    
    # Create a port with scalar float value
    p = cn.Port(value=3.14)
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    
    # Assign a vector float value, should change to vector float
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    print(f"After assigning vector float, value: {p.value}")
    print(f"After assigning vector float, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"

def test_vector_to_scalar():
    print("\nTesting vector to scalar conversion...")
    
    # Create a port with vector int value
    p = cn.Port(value=np.array([1, 2, 3, 4, 5]))
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    
    # Assign a scalar int value, should change to scalar int
    p.value = 42
    print(f"After assigning scalar int, value: {p.value}")
    print(f"After assigning scalar int, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    
    # Create a port with vector float value
    p = cn.Port(value=np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    print(f"Initial value: {p.value}")
    print(f"Initial value_type: {p.get_value_type()}")
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    
    # Assign a scalar float value, should change to scalar float
    p.value = 3.14
    print(f"After assigning scalar float, value: {p.value}")
    print(f"After assigning scalar float, value_type: {p.get_value_type()}")
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"

if __name__ == "__main__":
    test_int_to_float()
    test_float_to_int()
    test_vector_int_to_float()
    test_vector_float_to_int()
    test_scalar_to_vector()
    test_vector_to_scalar()
    print("\nAll tests completed!")