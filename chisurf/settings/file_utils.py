from __future__ import annotations

import pathlib
from typing import Any, Callable, Optional, Union


def safe_open_file(file_path: Union[str, pathlib.Path], 
                  processor: Optional[Callable[[Any], Any]] = None,
                  default_value: Any = None,
                  mode: str = 'r',
                  error_message: Optional[str] = None) -> Any:
    """Safely open and process a file with error handling.

    This function opens a file and processes its content using the provided processor function.
    If any file-related error occurs, it catches the exception, prints an error message,
    and returns the default value.

    Args:
        file_path: Path to the file to open
        processor: Function to process the file content (e.g., json.load, yaml.safe_load)
                  If None, the file content is returned as is
        default_value: Value to return if an error occurs
        mode: File opening mode ('r', 'rb', etc.)
        error_message: Custom error message to print if an error occurs
                      If None, a default message is used

    Returns:
        The processed file content if successful, or the default value if an error occurs
    """
    try:
        with open(str(file_path), mode) as fp:
            if processor:
                return processor(fp)
            else:
                return fp.read()
    except (FileNotFoundError, PermissionError, IOError) as e:
        if error_message:
            print(f"{error_message}: {e}")
        else:
            print(f"Error opening file {file_path}: {e}")
        return default_value