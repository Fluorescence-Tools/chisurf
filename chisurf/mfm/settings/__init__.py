import os
import yaml
import json

package_directory = os.path.dirname(os.path.abspath(__file__))
settings_file = os.path.join(package_directory, 'chisurf.yaml')

cs_settings = yaml.load(open(settings_file))
colors = yaml.load(open(os.path.join(package_directory, 'colors.yaml')))
verbose = cs_settings['verbose']
__version__ = cs_settings['version']
__name__ = cs_settings['name']
working_path = ''

style_sheet_file = os.path.join(package_directory, cs_settings['gui']['style_sheet'])
structure_data = json.load(open(os.path.join(package_directory, 'structure.json')))
