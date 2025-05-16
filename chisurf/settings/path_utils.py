from __future__ import annotations

import pathlib


def get_path(path_type: str = 'settings') -> pathlib.Path:
    """Get the path of the chisurf settings file.

    This function returns the path of the chisurf settings files in the
    user folder. The default path is '~/.chisurf'. If the path does not
    exist, this function creates the folder.

    :return: pathlib.Path object pointing to the chisurf setting folder
    """
    if path_type == 'settings':
        path = pathlib.Path.home() / '.chisurf'  # Define the settings path
        path.mkdir(parents=True, exist_ok=True)
    elif path_type == 'chisurf':
        path = pathlib.Path(__file__).parent.parent
    return path