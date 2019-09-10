from __future__ import annotations

import json
import os

import yaml
import pyqtgraph as pg

package_directory = os.path.dirname(os.path.abspath(__file__))
settings_file = os.path.join(package_directory, 'chisurf.yaml')

cs_settings = yaml.safe_load(
    open(settings_file)
)

colors = yaml.safe_load(
    open(
        os.path.join(package_directory, 'colors.yaml')
    )
)

verbose = cs_settings['verbose']
__version__ = cs_settings['version']
__name__ = cs_settings['name']
working_path = ''

structure_data = json.load(open(os.path.join(package_directory, 'structure.json')))

pyqtgraph_settings = cs_settings['gui']['plot']["pyqtgraph"]
for setting in pyqtgraph_settings:
    pg.setConfigOption(setting, cs_settings['gui']['plot']['pyqtgraph'][setting])
