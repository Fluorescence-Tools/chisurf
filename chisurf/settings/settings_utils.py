from __future__ import annotations

import pathlib
import shutil
import yaml

from .file_utils import safe_open_file
from .path_utils import get_path


def get_chisurf_settings(setting_file: pathlib.Path, use_source_folder: bool = False) -> dict:
    """This function returns the content of a settings file in the user
    settings path. If the settings file does not exist it is copied from
    the package folder to the user settings path.

    :param setting_file: path to settings file
    :param use_source_folder: if true use settings file in source code folder
    :return:
    """
    package_path = pathlib.Path(__file__).parent
    original_settings = package_path / setting_file.parts[-1]
    if use_source_folder:
        setting_file = package_path / setting_file.parts[-1]
    else:
        if not setting_file.is_file():
            shutil.copyfile(original_settings, setting_file)
    return safe_open_file(
        file_path=setting_file,
        processor=yaml.safe_load,
        default_value={},
        error_message=f"Error opening settings file {setting_file}"
    )


def copy_settings_to_user_folder():
    """Copies all settings files from the package directory to the user folder,
    ensuring that existing files are not overwritten."""
    package_path = pathlib.Path(__file__).parent
    user_settings_path = get_path('settings')
    user_settings_path.mkdir(parents=True, exist_ok=True)

    for file in package_path.iterdir():
        if file.is_file():
            destination_file = user_settings_path / file.name
            if not destination_file.exists():  # Avoid overwriting existing files
                shutil.copyfile(file, destination_file)

    # Also copy style files
    copy_styles_to_user_folder()


def copy_styles_to_user_folder():
    """Copies all style files from the gui/styles directory to the user folder,
    ensuring that existing files are not overwritten."""
    package_path = pathlib.Path(__file__).parent.parent / 'gui' / 'styles'
    user_settings_path = get_path('settings') / 'styles'
    user_settings_path.mkdir(parents=True, exist_ok=True)

    for file in package_path.iterdir():
        if file.is_file() and file.suffix == '.qss':
            destination_file = user_settings_path / file.name
            if not destination_file.exists():  # Avoid overwriting existing files
                shutil.copyfile(file, destination_file)