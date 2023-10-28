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

try:
    if sys.version_info >= (3, 8):
        import typing
    elif sys.version_info >= (3, 7):
        # monkey patch the 3.7 typing system as
        # TypedDict etc. is missing
        import typing_extensions
        import typing
        for key in typing_extensions.__dict__.keys():
            f = typing_extensions.__dict__[key]
            if callable(f):
                typing.__dict__[key] = f
    else:
        import typing_extensions as typing
except ModuleNotFoundError:
    print(
        "WARNING typing_extensions not found",
        file=sys.stderr
    )
    typing = None

import chisurf.settings
import chisurf.info

__version__ = chisurf.info.__version__

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


