import os
import sys
import pathlib
import ctypes

# Helper to set hidden attribute on Windows

def _set_hidden_on_windows(path: pathlib.Path) -> None:
    if os.name != 'nt':
        return
    try:
        FILE_ATTRIBUTE_HIDDEN = 0x2
        GetFileAttributesW = ctypes.windll.kernel32.GetFileAttributesW
        SetFileAttributesW = ctypes.windll.kernel32.SetFileAttributesW
        GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
        GetFileAttributesW.restype = ctypes.c_uint32
        SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
        SetFileAttributesW.restype = ctypes.c_int
        attrs = GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:
            return
        SetFileAttributesW(str(path), attrs | FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass

# Define the user plugins directory
user_plugins_dir = pathlib.Path.home() / '.chisurf' / 'plugins'
chisurf_user_dir = user_plugins_dir.parent

# Ensure the base ~/.chisurf exists and is hidden if newly created
base_existed = chisurf_user_dir.exists()
chisurf_user_dir.mkdir(parents=True, exist_ok=True)
if not base_existed:
    _set_hidden_on_windows(chisurf_user_dir)

# Ensure the plugins directory exists
user_plugins_dir.mkdir(parents=True, exist_ok=True)

# Add the user plugins directory to the module's __path__
if str(user_plugins_dir) not in __path__:
    __path__.append(str(user_plugins_dir))
