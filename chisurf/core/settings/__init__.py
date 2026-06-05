from __future__ import annotations

import os
import datetime
import json
import pathlib
import sys

# Initialize environment (PATH, Qt plugins, vispy, FreeType) as early as possible
from . import env_bootstrap  # noqa: F401

# Import utility functions
from .file_utils import safe_open_file
from .path_utils import get_path
from .settings_utils import (
    get_chisurf_settings,
    copy_settings_to_user_folder,
    copy_styles_to_user_folder
)
from .cleanup import clear_settings_folder, clear_logging_files, clear_user_plugins_folder
from .path_utils import get_path  # Needed early

# Define Chisurf cache path inside user settings folder
_chisurf_user_cache_dir = get_path('settings') / "cache"

# Set environment variables for Numba and Python bytecode cache
os.environ["NUMBA_CACHE_DIR"] = str(_chisurf_user_cache_dir)
os.environ["PYTHONPYCACHEPREFIX"] = str(_chisurf_user_cache_dir)

# Ensure the cache directory exists
_chisurf_user_cache_dir.mkdir(parents=True, exist_ok=True)

# Path constants
chisurf_settings_path = get_path('settings')
chisurf_root = get_path('chisurf')
macro_path = chisurf_root / "macros"
plugin_path = chisurf_root / "plugins"
_notebook_root = chisurf_root.parent / "notebooks"
notebook_path = _notebook_root if _notebook_root.is_dir() else (chisurf_root / "notebooks")

# Copy settings files if not already present
copy_settings_to_user_folder()

import chisurf.core.info
__version__ = chisurf.core.info.__version__

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

# BETA OVERRIDES: Force Jupyter to start even if disabled in user settings
# to ensure connectivity for the Antigravity (v26.1) Beta release.
_gui_overrides = cs_settings.setdefault('gui', {})
_gui_overrides['start_jupyter_on_startup'] = True
# ZMQ server is always auto-started — no setting required.
gui.update(_gui_overrides)


def is_dev_mode() -> bool:
    """Return True if dev mode is enabled (experimental mode).

    Dev mode enables developer features like code badge buttons
    for jumping to source locations in the embedded editor.
    """
    return bool(cs_settings.get('enable_experimental', False))


def dev_mode_settings() -> dict:
    """Return dev_mode settings dict from gui.dev_mode."""
    gui_settings = cs_settings.get('gui', {})
    return gui_settings.get('dev_mode', {})

# Load help mappings from the program's settings folder only. These are not
# intended to be user-editable, so we always read them from the source folder
# and do not look at (or copy into) the user settings directory.
help_settings_file = chisurf_settings_path / 'help_mappings.yaml'
_help_settings = get_chisurf_settings(help_settings_file, use_source_folder=True)
if isinstance(_help_settings, dict):
    help = _help_settings.get('help', _help_settings)
else:
    help = {}

# Open color settings file
color_settings_file = chisurf_settings_path / 'settings_colors.yaml'
colors = get_chisurf_settings(color_settings_file)

package_directory = pathlib.Path(__file__).parent
chisurf_root = package_directory.parent.parent
style_sheet_file = chisurf_root / 'gui' / 'styles' / gui['style_sheet']
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

# Optional registry of fitting-parameter metadata used to enrich parameter
# descriptions in the GUI. This is populated by the command
# ``python dev_tools/export_fitting_parameters.py`` and can be
# edited by the user.
fitting_parameters = safe_open_file(
    file_path=package_directory / 'constants' / 'fitting_parameters.json',
    processor=json.load,
    default_value={},
    error_message="Error opening fitting_parameters.json file"
)

eps = sys.float_info.epsilon
working_path = ''

session_str = datetime.datetime.now().strftime('session_%H_%M_%d_%m_%Y')
# Create logs subfolder
logs_folder = chisurf_settings_path / "logs"
logs_folder.mkdir(exist_ok=True)
session_file = logs_folder / str(session_str + ".py")
session_log = logs_folder / str(session_str + ".log")

try:
    import chisurf as _chisurf
    _chisurf._apply_logging_settings(sys.modules[__name__])
except Exception:
    pass
