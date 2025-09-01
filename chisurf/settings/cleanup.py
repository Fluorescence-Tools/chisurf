from __future__ import annotations

import os
import shutil

from .path_utils import get_path


def clear_settings_folder():
    """
    Remove settings files and subdirectories inside the settings folder, but preserve log files.

    This function walks through the directory returned by `get_path()` and:
      - Recursively deletes each subdirectory (skipping over any files it cannot remove),
      - Deletes each settings file at the top level (skipping log files),
      - Logs a concise warning via `chisurf.logging.warning()` (max 128 chars)
        for any file or directory that cannot be deleted.

    The root settings folder itself is left intact, even if not empty.

    Raises:
        None. All deletion errors are caught and logged.
    """
    import chisurf

    root = get_path()

    # Helper to warn on failed removals inside rmtree()
    def _handle_remove_error(func, path, exc_info):
        ex = exc_info[1]
        # Only skip PermissionErrors (file-in-use, etc.)
        if isinstance(ex, PermissionError):
            chisurf.logging.warning(f"Could not delete {path}: {ex}. Skipping.")
            return
        # Propagate everything else
        raise ex

    # If the root doesn't even exist, nothing to do
    if not os.path.isdir(root):
        return

    # Iterate through *direct* children of root
    for entry in os.scandir(root):
        path = entry.path
        try:
            if entry.is_dir(follow_symlinks=False):
                # Recursively remove this subfolder entirely (with our onerror)
                shutil.rmtree(path, onerror=_handle_remove_error)
            else:
                # Skip log files (files ending with .log)
                if not str(path).endswith('.log'):
                    # Remove a single file
                    os.unlink(path)
        except PermissionError as e:
            chisurf.logging.warning(f"Skipping locked file or folder: {path}")
        except OSError as e:
            # e.errno==ENOTEMPTY can happen if subdir isn't empty (due to skips)
            chisurf.logging.warning(f"Couldn't remove {path}")


def clear_logging_files():
    """
    Remove only log files inside the logs subfolder of the settings folder.

    This function walks through the logs directory inside the settings folder and:
      - Deletes each log file (files ending with .log or .py),
      - Logs a concise warning via `chisurf.logging.warning()` (max 128 chars)
        for any file that cannot be deleted.

    The logs folder itself is left intact, even if not empty.

    Raises:
        None. All deletion errors are caught and logged.
    """
    import chisurf

    root = get_path()
    logs_folder = root / "logs"

    # If the logs folder doesn't exist, nothing to do
    if not os.path.isdir(logs_folder):
        return

    # Iterate through *direct* children of logs folder
    for entry in os.scandir(logs_folder):
        path = entry.path
        try:
            if not entry.is_dir(follow_symlinks=False):
                # Remove log files (files ending with .log) and session files (.py)
                if str(path).endswith('.log') or str(path).endswith('.py'):
                    os.unlink(path)
        except PermissionError as e:
            chisurf.logging.warning(f"Skipping locked file: {path}")
        except OSError as e:
            chisurf.logging.warning(f"Couldn't remove file: {path}")
