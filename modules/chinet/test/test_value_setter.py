import sys
import os
import numpy as np

# Add the parent directory to the path so we can import chinet
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)

import chinet as cn

def test_integer_assignment():
    print("Testing integer assignment...")
    p1 = cn.Port()
    print(f"Initial value: {p1.value}")
    print(f"Initial value_type: {p1.get_value_type()}")
    
    # Assign an integer value
    p1.value = 3
    print(f"After assigning 3, value: {p1.value}")
    print(f"After assigning 3, value_type: {p1.get_value_type()}")
    
    # Verify that the value is still an integer
    print(f"Is the value an integer? {isinstance(p1.value, int) or (isinstance(p1.value, np.ndarray) and np.issubdtype(p1.value.dtype, np.integer))}")
    
    # Test with value_type 2
    p2 = cn.Port()
    p2.set_value_type(2)
    print(f"p2 initial value_type: {p2.get_value_type()}")
    p2.value = 5
    print(f"After assigning 5 to p2, value: {p2.value}")
    print(f"After assigning 5 to p2, value_type: {p2.get_value_type()}")

def test_float_assignment():
    print("\nTesting float assignment...")
    p1 = cn.Port()
    print(f"Initial value: {p1.value}")
    print(f"Initial value_type: {p1.get_value_type()}")
    
    # Assign a float value
    p1.value = 3.14
    print(f"After assigning 3.14, value: {p1.value}")
    print(f"After assigning 3.14, value_type: {p1.get_value_type()}")
    
    # Test with value_type 3
    p2 = cn.Port()
    p2.set_value_type(3)
    print(f"p2 initial value_type: {p2.get_value_type()}")
    p2.value = 5.5
    print(f"After assigning 5.5 to p2, value: {p2.value}")
    print(f"After assigning 5.5 to p2, value_type: {p2.get_value_type()}")

def test_type_preservation():
    print("\nTesting type preservation...")
    # Test int -> float -> int
    p1 = cn.Port(5)
    print(f"p1 initial value: {p1.value}, value_type: {p1.get_value_type()}")
    p1.value = 3.14
    print(f"p1 after float assignment: {p1.value}, value_type: {p1.get_value_type()}")
    p1.value = 7
    print(f"p1 after int assignment: {p1.value}, value_type: {p1.get_value_type()}")
    
    # Test with value_type 2 and 3
    p2 = cn.Port()
    p2.set_value_type(2)
    p2.value = 10
    print(f"p2 with value_type 2: {p2.value}, value_type: {p2.get_value_type()}")
    p2.value = 10.5
    print(f"p2 after float assignment: {p2.value}, value_type: {p2.get_value_type()}")
    p2.value = 12
    print(f"p2 after int assignment: {p2.value}, value_type: {p2.get_value_type()}")

if __name__ == "__main__":
    test_integer_assignment()
    test_float_assignment()
    test_type_preservation()