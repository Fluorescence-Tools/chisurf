from __future__ import annotations

import os
import pathlib
import sys
import ctypes


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
    """Get the path of the chisurf settings file.

    This function returns the path of the chisurf settings files in the
    user folder. The default path is '~/.chisurf'. If the path does not
    exist, this function creates the folder. On Windows, when the folder is
    created by this function, it will be marked as hidden.

    :return: pathlib.Path object pointing to the chisurf setting folder
    """
    if path_type == 'settings':
        path = pathlib.Path.home() / '.chisurf'  # Define the settings path
        existed_before = path.exists()
        path.mkdir(parents=True, exist_ok=True)
    elif path_type == 'chisurf':
        path = pathlib.Path(__file__).parent.parent
    _set_hidden_on_windows(path)
    return path