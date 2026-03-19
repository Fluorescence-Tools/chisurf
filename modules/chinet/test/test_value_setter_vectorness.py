import sys
import os
import numpy as np

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_scalar_int():
    print("\nTesting scalar int assignment...")
    p = cn.Port()
    p.value = 42
    print(f"Value: {p.value}")
    print(f"Value type: {p.get_value_type()}")
    print(f"Expected value type: 0 (scalar int)")
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    assert isinstance(p.value, (int, np.integer)), f"Expected int value, got {type(p.value)}"

def test_scalar_float():
    print("\nTesting scalar float assignment...")
    p = cn.Port()
    p.value = 3.14
    print(f"Value: {p.value}")
    print(f"Value type: {p.get_value_type()}")
    print(f"Expected value type: 1 (scalar float)")
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    assert isinstance(p.value, (float, np.floating)), f"Expected float value, got {type(p.value)}"

def test_vector_int():
    print("\nTesting vector int assignment...")
    p = cn.Port()
    p.value = np.array([1, 2, 3, 4, 5])
    print(f"Value: {p.value}")
    print(f"Value type: {p.get_value_type()}")
    print(f"Expected value type: 2 (vector int)")
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    assert isinstance(p.value, (tuple, list, np.ndarray)), f"Expected array-like value, got {type(p.value)}"

def test_vector_float():
    print("\nTesting vector float assignment...")
    p = cn.Port()
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    print(f"Value: {p.value}")
    print(f"Value type: {p.get_value_type()}")
    print(f"Expected value type: 3 (vector float)")
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    assert isinstance(p.value, (tuple, list, np.ndarray)), f"Expected array-like value, got {type(p.value)}"

def test_type_conversion():
    print("\nTesting type conversion...")
    
    # Scalar int to scalar float
    p = cn.Port()
    p.value = 42
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    p.value = 3.14
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    
    # Scalar float to scalar int
    p = cn.Port()
    p.value = 3.14
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    p.value = 42
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    
    # Vector int to vector float
    p = cn.Port()
    p.value = np.array([1, 2, 3, 4, 5])
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    
    # Vector float to vector int
    p = cn.Port()
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    p.value = np.array([1, 2, 3, 4, 5])
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    
    # Scalar int to vector int
    p = cn.Port()
    p.value = 42
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    p.value = np.array([1, 2, 3, 4, 5])
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    
    # Vector int to scalar int
    p = cn.Port()
    p.value = np.array([1, 2, 3, 4, 5])
    assert p.get_value_type() == 2, f"Expected value_type 2, got {p.get_value_type()}"
    p.value = 42
    assert p.get_value_type() == 0, f"Expected value_type 0, got {p.get_value_type()}"
    
    # Scalar float to vector float
    p = cn.Port()
    p.value = 3.14
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    
    # Vector float to scalar float
    p = cn.Port()
    p.value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert p.get_value_type() == 3, f"Expected value_type 3, got {p.get_value_type()}"
    p.value = 3.14
    assert p.get_value_type() == 1, f"Expected value_type 1, got {p.get_value_type()}"
    
    print("All type conversion tests passed!")

if __name__ == "__main__":
    test_scalar_int()
    test_scalar_float()
    test_vector_int()
    test_vector_float()
    test_type_conversion()
    print("\nAll tests completed!")