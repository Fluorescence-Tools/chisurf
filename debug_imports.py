import sys
import os

print("--- sys.path ---")
for p in sys.path:
    print(p)
print("----------------")

try:
    import pyarrow as pa
    print(f"PyArrow imported from: {pa.__file__}")
    print(f"PyArrow version: {pa.__version__}")
except Exception as e:
    print(f"FAILED to import pyarrow: {e}")
    import traceback
    traceback.print_exc()

try:
    import pandas as pd
    print(f"Pandas imported from: {pd.__file__}")
    print(f"Pandas version: {pd.__version__}")
except Exception as e:
    print(f"FAILED to import pandas: {e}")
    import traceback
    traceback.print_exc()
