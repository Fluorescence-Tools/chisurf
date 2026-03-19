from __future__ import annotations

import os
import pathlib
import sys
import ctypes


USER_SETTINGS_EXISTED_BEFORE = None


def _set_hidden_on_windows(path: pathlib.Path) -> None:
    """Set the hidden attribute on Windows for the given path.

    This uses WinAPI via ctypes to OR the FILE_ATTRIBUTE_HIDDEN flag without
    clearing existing attributes.
    """
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
        if attrs == 0xFFFFFFFF:  # INVALID_FILE_ATTRIBUTES
            return
        # OR the hidden flag
        SetFileAttributesW(str(path), attrs | FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        # Fail silently if we cannot set the attribute
        pass


def get_path(path_type: str = 'settings') -> pathlib.Path:
    """Get key chisurf paths.

    - For path_type == 'settings': returns the user settings dir '~/.chisurf'.
      Ensures it exists and marks it hidden on Windows (only when newly created).
    - For path_type == 'chisurf': returns the installed chisurf package directory.
      Never modifies attributes of the installed package directory.
    """
    if path_type == 'settings':
        path = pathlib.Path.home() / '.chisurf'
        existed_before = path.exists()
        global USER_SETTINGS_EXISTED_BEFORE
        if USER_SETTINGS_EXISTED_BEFORE is None:
            USER_SETTINGS_EXISTED_BEFORE = bool(existed_before)
        path.mkdir(parents=True, exist_ok=True)
        # Only set hidden for the user settings dir, and only if we created it now
        if not existed_before:
            _set_hidden_on_windows(path)
        return path

    elif path_type == 'chisurf':
        # Return the module root without changing attributes
        return pathlib.Path(__file__).parent.parent
    else:
        # Fallback: return settings dir behavior for unknown types, without risking hiding other paths
        path = pathlib.Path.home() / '.chisurf'
        path.mkdir(parents=True, exist_ok=True)
        return path
