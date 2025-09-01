import pathlib
import yaml

import chisurf.experiments.experiment
import chisurf.experiments.reader
import chisurf.experiments.fcs
import chisurf.experiments.tcspc
import chisurf.experiments.pda
import chisurf.experiments.globalfit
import chisurf.experiments.modelling
from chisurf.experiments.experiment import Experiment
from chisurf.settings import get_path

# Load experiment types from YAML file
def load_experiment_types():
    # Get the path to the experiment_configs.yaml file
    settings_path = get_path('settings')
    experiment_configs_file = settings_path / 'experiment_configs.yaml'

    # If the file doesn't exist in the user settings, try to load it from the package
    if not experiment_configs_file.is_file():
        package_path = pathlib.Path(__file__).parent.parent / 'settings'
        experiment_configs_file = package_path / 'experiment_configs.yaml'

    # Load the YAML file
    with open(str(experiment_configs_file), 'r') as fp:
        config = yaml.safe_load(fp)

    # Create experiment types from the configuration
    experiment_types = {}
    if 'experiment_types' in config:
        for key, value in config['experiment_types'].items():
            name = value.get('name', '')
            hidden = value.get('hidden', False)
            experiment_types[key] = Experiment(name, hidden)

    return experiment_types

# Load experiment types
types = load_experiment_types()
