from __future__ import annotations

import os
import pathlib
import datetime
import sys
import yaml
import json
import shutil


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


def get_chisurf_settings(setting_file: pathlib.Path, use_source: bool = False) -> dict:
    """This function returns the content of a settings file in the user
    settings path. If the settings file does not exist it is copied from
    the package folder to the user settings path.

    :param setting_file: path to settings file
    :param use_source: if true use settings file in source code folder
    :return:
    """
    package_path = pathlib.Path(__file__).parent
    original_settings = package_path / setting_file.parts[-1]
    if use_source:
        setting_file = package_path / setting_file.parts[-1]
    else:
        if not setting_file.is_file():
            shutil.copyfile(original_settings, setting_file)
    with open(str(setting_file), 'r') as fp:
        return yaml.safe_load(fp)


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


def clear_settings_folder():
    shutil.rmtree(get_path())


#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################
chisurf_settings_path = get_path('settings')
macro_path = get_path('chisurf') / "macros"
plugin_path = get_path('chisurf') / "plugins"
notebook_path = get_path('chisurf') / "notebooks"

# Copy settings files if not already present
copy_settings_to_user_folder()

# Open chisurf settings file
chisurf_settings_file = chisurf_settings_path / 'settings_chisurf.yaml'
cs_settings = get_chisurf_settings(chisurf_settings_file, True)

anisotropy = dict()
with open(get_path('chisurf') / "settings/anisotropy_corrections.json") as fp:
    anisotropy.update(json.load(fp))

verbose = False
gui = dict()
parameter = dict()
optimization = dict()
fret = dict()
tcspc = dict()
fps = dict()
locals().update(cs_settings)

# Open color settings file
color_settings_file = chisurf_settings_path / 'settings_colors.yaml'
colors = get_chisurf_settings(color_settings_file)

package_directory = pathlib.Path(__file__).parent
style_sheet_file = package_directory / '../gui/styles/' / gui['style_sheet']
style_sheet = open(style_sheet_file, 'r').read()
with open(package_directory / './constants/structure.json') as fp:
    structure_data = json.load(fp)

eps = sys.float_info.epsilon
working_path = ''

session_str = datetime.datetime.now().strftime('session_%H_%M_%d_%m_%Y')
session_file = chisurf_settings_path / str(session_str + ".py")
session_log = chisurf_settings_path / str(session_str + ".log")
