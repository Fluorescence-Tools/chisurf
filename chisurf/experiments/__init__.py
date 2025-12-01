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


def load_experiment_types():
    """Load experiment type registry from experiment_configs.yaml.

    The primary source is the user settings file
    ``get_path('settings') / 'experiment_configs.yaml'``; if that file does
    not exist, the packaged default under ``chisurf/settings`` is used.

    In addition to entries under the top-level ``experiment_types`` mapping,
    this function ensures that every top-level experiment section (except
    ``"experiment_types"`` and ``"global"``) receives an
    :class:`Experiment` instance. This makes the registry robust against
    partially updated user configuration files where a new experiment (e.g.
    ``pch``) has been added without a corresponding ``experiment_types``
    entry.
    """

    # Locate the configuration file, preferring the user settings copy.
    settings_path = get_path('settings')
    experiment_configs_file = settings_path / 'experiment_configs.yaml'
    if not experiment_configs_file.is_file():
        package_path = pathlib.Path(__file__).parent.parent / 'settings'
        experiment_configs_file = package_path / 'experiment_configs.yaml'

    # Load the YAML configuration (empty dict on error/empty file).
    try:
        with open(str(experiment_configs_file), 'r', encoding='utf-8') as fp:
            config = yaml.safe_load(fp) or {}
    except Exception:
        config = {}

    experiment_types: dict[str, Experiment] = {}

    # First, honor explicit experiment_types definitions when present.
    type_defs = config.get('experiment_types', {}) or {}
    if isinstance(type_defs, dict):
        for key, value in type_defs.items():
            if not isinstance(value, dict):
                value = {}
            name = value.get('name', key)
            hidden = bool(value.get('hidden', False))
            experiment_types[key] = Experiment(name, hidden)

    # Next, ensure that all top-level experiment sections have an Experiment
    # object, even if they are missing from experiment_types. This prevents
    # KeyError when new experiments are added only under their own section.
    for key in list(config.keys()):
        if key in ('experiment_types', 'global'):
            continue
        if key not in experiment_types:
            experiment_types[key] = Experiment(name=key, hidden=False)

    return experiment_types


# Load experiment types at import time so chisurf.experiments.types is ready
types = load_experiment_types()
