from __future__ import annotations

import os
import numpy as np
import yaml
import json
import pyqtgraph as pg

#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################

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

parameter = cs_settings['parameter']
fitting = cs_settings['fitting']
fret = cs_settings['fret']
tcspc = cs_settings['tcspc']

pyqtgraph_settings = gui['plot']["pyqtgraph"]
for setting in pyqtgraph_settings:
    pg.setConfigOption(setting, gui['plot']['pyqtgraph'][setting])

verbose = cs_settings['verbose']
working_path = ''

package_directory = os.path.dirname(os.path.abspath(__file__))
eps = np.sqrt(np.finfo(float).eps)

with open(os.path.join(
        package_directory,
        './constants/structure.json')
) as fp:
    structure_data = json.load(fp)


#######################################################
#        LIST OF FITS, DATA, EXPERIMENTS              #
#######################################################


