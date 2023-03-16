from __future__ import annotations

try:
    import sys
    import logging
    import pathlib
    import numpy
    import pyopencl
    import typing
except ImportError:
    print("Import issue.")

import chisurf.settings
import chisurf.info

fits: typing.List[chisurf.fitting.fit.FitGroup] = list()
imported_datasets: typing.List[chisurf.data.DataGroup] = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object
experiment: typing.Dict[str, chisurf.experiments.experiment.Experiment] = dict()
fit_windows = list()
working_path = pathlib.Path().home()
verbose = chisurf.settings.verbose


logging.basicConfig(
    filename=settings.session_log,
    level=logging.DEBUG
)


