import chinet as cn
import numpy as np

def test_initial_garbage():
    print("Testing initial Port value...")
    p = cn.Port()
    val = p.value
    print(f"Initial value: {val}")
    # It should probably be empty or zeroed, but definitely not random garbage of size 8
    # Based on Port.h, it resizes to 64 bytes.
    if isinstance(val, np.ndarray):
        print(f"Value size: {len(val)}")
        if len(val) > 0 and np.any(val != 0):
             print("Found garbage in initial value!")
    else:
        print(f"Value is not an array: {val}")

if __name__ == "__main__":
    test_initial_garbage()
