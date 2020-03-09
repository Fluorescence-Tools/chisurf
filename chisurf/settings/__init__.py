from __future__ import annotations

import os
import pathlib
import datetime
import sys
import yaml
import json
import shutil

#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################
package_directory =pathlib.Path(__file__).parent

home = pathlib.Path().home()
path = home / '.chisurf'
if not path.exists():
    os.makedirs(str(path.absolute()))

# Open chisurf settings file
chisurf_settings_file = home / '.chisurf/settings_chisurf.yaml'

# If settings file does not exist in user folder
# copy settings file from program directory
if not chisurf_settings_file.is_file():
    original_settings = package_directory / 'settings_chisurf.yaml'
    shutil.copyfile(
        original_settings,
        chisurf_settings_file
    )

cs_settings = dict()
with open(chisurf_settings_file) as fp:
    cs_settings.update(
        yaml.safe_load(fp)
    )
verbose = False
gui = dict()
parameter = dict()
fitting = dict()
fret = dict()
tcspc = dict()
fps = dict()
locals().update(cs_settings)

# Open color settings file
color_settings_file = home / '.chisurf/settings_colors.yaml'

# If settings file does not exist in user folder
# copy settings file from program directory
if not color_settings_file.is_file():
    original_colors = package_directory / 'settings_colors.yaml'
    shutil.copyfile(
        original_colors,
        color_settings_file
    )
with open(color_settings_file) as fp:
    colors = yaml.safe_load(fp)


package_directory = pathlib.Path(__file__).parent
eps = sys.float_info.epsilon
working_path = ''

style_sheet_file = package_directory / './gui/styles/' / gui['style_sheet']

with open(package_directory / './constants/structure.json') as fp:
    structure_data = json.load(fp)

session_str = datetime.datetime.now().strftime('session_%H_%M_%d_%m_%Y')

path = home / '.chisurf'
if not path.exists():
    os.makedirs(str(path))

session_file = path / str(session_str + ".py")
session_log = path / str(session_str + ".log")

url = 'https://github.com/Fluorescence-Tools/chisurf'
