from __future__ import annotations

import datetime
import json
import pathlib
import sys

# Import utility functions
from .file_utils import safe_open_file
from .path_utils import get_path
from .settings_utils import (
    get_chisurf_settings,
    copy_settings_to_user_folder,
    copy_styles_to_user_folder
)
from .cleanup import clear_settings_folder, clear_logging_files

# Path constants
chisurf_settings_path = get_path('settings')
macro_path = get_path('chisurf') / "macros"
plugin_path = get_path('chisurf') / "plugins"
notebook_path = get_path('chisurf') / "notebooks"

# Copy settings files if not already present
copy_settings_to_user_folder()

# Open chisurf settings file
chisurf_settings_file = chisurf_settings_path / 'settings_chisurf.yaml'
# To use the settings in the home folder set to false
# if set to true uses settings in source folder.
cs_settings = get_chisurf_settings(chisurf_settings_file, use_source_folder=False)

anisotropy = dict()
anisotropy_data = safe_open_file(
    file_path=get_path('chisurf') / "settings" / "anisotropy_corrections.json",
    processor=json.load,
    default_value={},
    error_message="Error opening anisotropy corrections file"
)
anisotropy.update(anisotropy_data)

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
style_sheet_file = package_directory / '..' / 'gui' / 'styles' / gui['style_sheet']
style_sheet = safe_open_file(
    file_path=style_sheet_file,
    default_value="",
    error_message=f"Error opening style sheet file {style_sheet_file}"
)
structure_data = safe_open_file(
    file_path=package_directory / 'constants' / 'structure.json',
    processor=json.load,
    default_value={},
    error_message="Error opening structure.json file"
)

eps = sys.float_info.epsilon
working_path = ''

session_str = datetime.datetime.now().strftime('session_%H_%M_%d_%m_%Y')
# Create logs subfolder
logs_folder = chisurf_settings_path / "logs"
logs_folder.mkdir(exist_ok=True)
session_file = logs_folder / str(session_str + ".py")
session_log = logs_folder / str(session_str + ".log")
