from __future__ import annotations

import json
import os

import yaml
import pyqtgraph as pg

package_directory = os.path.dirname(
    os.path.abspath(__file__)
)
settings_file = os.path.join(
    package_directory, 'chisurf.yaml'
)

with open(settings_file) as fp:
    cs_settings = yaml.safe_load(fp)

gui = cs_settings['gui']

with open(
        os.path.join(
            package_directory,
            './gui/colors.yaml'
        )
) as fp:
    colors = yaml.safe_load(fp)

style_sheet_file = os.path.join(
    package_directory,
    './gui/styles/',
    gui['style_sheet']
)

with open(os.path.join(
        package_directory,
        './constants/structure.json')
) as fp:
    structure_data = json.load(fp)

parameter = cs_settings['parameter']
fitting = cs_settings['fitting']
fret = cs_settings['fret']
tcspc = cs_settings['tcspc']

pyqtgraph_settings = gui['plot']["pyqtgraph"]
for setting in pyqtgraph_settings:
    pg.setConfigOption(setting, gui['plot']['pyqtgraph'][setting])

verbose = cs_settings['verbose']
__version__ = cs_settings['version']
__name__ = cs_settings['name']
working_path = ''

