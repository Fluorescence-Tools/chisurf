import os
import sys
import tttrlib

def get_short_path_name(long_name):
    if sys.platform != 'win32':
        return long_name
    import ctypes
    from ctypes import wintypes
    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathNameW.restype = wintypes.DWORD
    
    long_name = os.path.abspath(long_name)
    output_buf_size = _GetShortPathNameW(long_name, None, 0)
    if output_buf_size == 0:
        return long_name
    output_buf = ctypes.create_unicode_buffer(output_buf_size)
    needed = _GetShortPathNameW(long_name, output_buf, output_buf_size)
    if output_buf_size >= needed:
        return output_buf.value
    else:
        return long_name

filename = os.path.abspath("test_µ_file_short.ptu")
with open(filename, "w") as f:
    f.write("mock")

short_path = get_short_path_name(filename)
print("Long path:", filename)
print("Short path:", short_path)

try:
    print("\nTrying short path...")
    tt = tttrlib.TTTR(short_path)
    print("Short path succeeded without throwing")
except Exception as e:
    print("Short path threw:", e)
    
os.remove(filename)
