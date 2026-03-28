import os, sys
TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, TOPDIR)
import numpy as np
import chinet as cn

p = cn.Port()
p.value = 2
print(f"p.value = {p.value}")
v = p.value
print("type:", type(v))
if isinstance(v, np.ndarray):
    print("dtype:", v.dtype)
print("buffer pointer:", p.get_buffer_ptr())
print("get_value_type:", p.get_value_type())
